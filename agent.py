import asyncio
import os
import time
import av
from dotenv import load_dotenv
import json
import httpx

# LiveKit Imports
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe, stt
from livekit.plugins import deepgram

load_dotenv()

# --- CONFIGURATION ---
AUDIO_FILES = {
    "greeting": "greeting.mp3",
    "continue": "continue.mp3", 
    "closing": "closing.mp3"
}

SAMPLE_RATE = 48000
NUM_CHANNELS = 1
FRAME_SIZE_SAMPLES = 480
BYTES_PER_SAMPLE = 2

grievance_storage = []

async def send_to_pipeline(transcript: str):
    url = "http://localhost:8000/process-grievance"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json={"text": transcript})
            print("[PIPELINE] Transcript sent for processing")
        except Exception as e:
            print(f"[ERROR] Pipeline unreachable: {e}")

class AudioFilePlayer:
    """Streams MP3s in perfect 10ms chunks."""
    def __init__(self, source: rtc.AudioSource):
        self.source = source
        self._current_task = None
        self.is_playing = False

    async def play(self, filename: str):
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        
        if not os.path.exists(filename):
            print(f"[ERROR] File not found: {filename}")
            return

        print(f"[PLAYING] {filename}")
        self.is_playing = True
        self._current_task = asyncio.create_task(self._stream_file(filename))
        await self._current_task
        self.is_playing = False

    async def _stream_file(self, filename: str):
        try:
            container = av.open(filename)
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            buffer = bytearray() 

            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    buffer.extend(resampled_frame.to_ndarray().tobytes())
                    while len(buffer) >= 960:
                        chunk_data = buffer[:960]
                        buffer = buffer[960:]
                        lk_frame = rtc.AudioFrame(
                            data=chunk_data, 
                            sample_rate=SAMPLE_RATE, 
                            num_channels=NUM_CHANNELS, 
                            samples_per_channel=FRAME_SIZE_SAMPLES
                        )
                        await self.source.capture_frame(lk_frame)
                        await asyncio.sleep(0.01) 
        except Exception as e:
            print(f"Error playing file: {e}")
        finally:
            if 'container' in locals(): 
                container.close()

class GrievanceBotLogic:
    def __init__(self):
        self.state = "greeting"
        self.grievance_text = ""
        self.should_disconnect = False
        self.continue_played = False  # NEW: Track if "continue" was already played
        self.last_input_time = time.time()  # NEW: Track timing
        self.total_word_count = 0  # NEW: Track overall input length

    def process_input(self, text: str) -> str:
        if self.should_disconnect: 
            return None
        
        text_lower = text.lower().strip()
        word_count = len(text_lower.split())
        self.total_word_count += word_count
        self.last_input_time = time.time()
        
        print(f"\n[USER SAYS] '{text}' (Words: {word_count}, Total: {self.total_word_count})")

        # 1. Greeting Phase
        if self.state == "greeting":
            if any(w in text_lower for w in ["bye", "exit", "nothing", "no"]):
                self.state = "closing"
                self.should_disconnect = True
                return "closing"
            
            self.state = "listening"
            self.grievance_text += text
            self.continue_played = False  # Reset for listening phase
            return None  # Don't play continue immediately after greeting

        # 2. Listening Phase
        elif self.state == "listening":
            self.grievance_text += " " + text
            
            # EXIT CHECK: Priority #1
            exit_words = ["done", "finished", "that's all", "that's it", "thank you", "thanks", "bye", "goodbye"]
            if any(w in text_lower for w in exit_words):
                print(f"[SAVING REPORT] {self.grievance_text}")
                grievance_storage.append({
                    "timestamp": time.time(),
                    "text": self.grievance_text
                })
                self.state = "closing"
                self.should_disconnect = True
                return "closing"

            # FILTER: Ignore very short fragments (< 5 words)
            if word_count < 5:
                print("   -> Fragment detected (too short). Staying silent.")
                return None 
            
            # NEW LOGIC: Only play "continue" once after substantial input
            # Only prompt if:
            # 1. We haven't played continue yet in this session
            # 2. User has spoken at least 20 words total (substantial grievance started)
            # 3. Current utterance is at least 10 words (complete thought)
            if (not self.continue_played and 
                self.total_word_count >= 20 and 
                word_count >= 5):
                print("   -> Playing continue prompt (first time only)")
                self.continue_played = True
                return "continue"
            
            # Otherwise, stay silent and keep listening
            print("   -> Acknowledged. Staying silent, waiting for more...")
            return None

        return None

async def entrypoint(ctx: JobContext):
    print(f"Room created: {ctx.room.name}. Waiting for user...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("bot_voice", source)
    await ctx.room.local_participant.publish_track(track)
    player = AudioFilePlayer(source)

    # --- TUNING CONFIGURATION ---
    # Increased to 4 seconds to give slow speakers more time
    # This prevents interrupting users who pause to think
    stt_provider = deepgram.STT(
        model="nova-2", 
        language="en", 
        smart_format=True,
        endpointing_ms=4000,  # Increased to 4 seconds for slow speakers
        interim_results=False  # Only process final transcripts
    )
    stt_stream = stt_provider.stream()
    bot = GrievanceBotLogic()

    async def process_stt_events():
        async for event in stt_stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                transcript = event.alternatives[0].text
                if not transcript: 
                    continue
                
                # If bot is speaking, ignore user inputs (prevents interruption)
                if player.is_playing:
                    print("[Bot is speaking - ignoring user input]")
                    continue

                audio_key = bot.process_input(transcript)
                
                if audio_key:
                    await player.play(AUDIO_FILES[audio_key])
                    
                if bot.should_disconnect:
                    asyncio.create_task(send_to_pipeline(bot.grievance_text)) # Fire and forget
                    await asyncio.sleep(2)
                    print("Disconnecting...")
                    await ctx.room.disconnect()
                    break
    
    asyncio.create_task(process_stt_events())

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track, 
        publication: rtc.TrackPublication, 
        participant: rtc.RemoteParticipant
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"User joined: {participant.identity}")
            audio_stream = rtc.AudioStream(track)
            
            async def push_audio_to_stt():
                async for event in audio_stream:
                    # Only push audio to STT when bot is NOT speaking
                    if not player.is_playing:
                        stt_stream.push_frame(event.frame)
            
            asyncio.create_task(push_audio_to_stt())
            asyncio.create_task(play_greeting_after_delay(player))

    async def play_greeting_after_delay(p):
        await asyncio.sleep(1.5) 
        print("Playing Greeting...")
        await p.play(AUDIO_FILES["greeting"])

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))