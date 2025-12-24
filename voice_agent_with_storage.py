import asyncio
import os
import time
import av
from dotenv import load_dotenv
import json

# LiveKit Imports
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe, stt
from livekit.plugins import deepgram

# Import our grievance processor
from grievance_processor import GrievanceProcessor

load_dotenv()

# --- CONFIGURATION ---
AUDIO_FILES = {
    "greeting": "audio/greeting.mp3",
    "continue": "audio/continue.mp3", 
    "closing": "audio/closing.mp3"
    "ack_1": "audio/hmm_hmm.mp3",       
    "ack_2": "audio/i_see.mp3",
    "ack_3": "audio/ohh_isit.mp3"
}

SAMPLE_RATE = 48000
NUM_CHANNELS = 1
FRAME_SIZE_SAMPLES = 480
BYTES_PER_SAMPLE = 2

# Initialize the grievance processor
grievance_processor = GrievanceProcessor(db_path="grievances.db")

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
        self.continue_played = False
        self.last_input_time = time.time()
        self.total_word_count = 0
        self.grievance_timestamp = None

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
            self.grievance_timestamp = time.time()  # Record start time
            self.continue_played = False
            return None

        # 2. Listening Phase
        elif self.state == "listening":
            self.grievance_text += " " + text
            
            # EXIT CHECK: Priority #1
            exit_words = ["done", "finished", "that's all", "that's it", "thank you", "thanks", "bye", "goodbye"]
            if any(w in text_lower for w in exit_words):
                print(f"[SAVING REPORT] {self.grievance_text}")
                
                # Process and store grievance asynchronously
                asyncio.create_task(self._process_grievance())
                
                self.state = "closing"
                self.should_disconnect = True
                return "closing"

            # FILTER: Ignore very short fragments (< 5 words)
            if word_count < 5:
                print("   -> Fragment detected (too short). Staying silent.")
                return None 
            
            # Only play "continue" once after substantial input
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
    
    async def _process_grievance(self):
        """Process grievance with LLM and store in database."""
        try:
            result = await grievance_processor.process_and_store(
                transcript=self.grievance_text,
                timestamp=self.grievance_timestamp or time.time()
            )
            
            print(f"\n{'='*60}")
            print(f"[SUCCESS] Grievance processed and stored")
            print(f"ID: {result['id']}")
            print(f"Category: {result['category']}")
            print(f"Priority: {result['priority']}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Summary: {result['summary']}")
            print(f"Tags: {', '.join(result['tags'])}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to process grievance: {e}")

async def entrypoint(ctx: JobContext):
    print(f"Room created: {ctx.room.name}. Waiting for user...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("bot_voice", source)
    await ctx.room.local_participant.publish_track(track)
    player = AudioFilePlayer(source)

    stt_provider = deepgram.STT(
        model="nova-2", 
        language="en", 
        smart_format=True,
        endpointing_ms=4000,
        interim_results=False
    )
    stt_stream = stt_provider.stream()
    bot = GrievanceBotLogic()

    async def process_stt_events():
        async for event in stt_stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                transcript = event.alternatives[0].text
                if not transcript: 
                    continue
                
                if player.is_playing:
                    print("[Bot is speaking - ignoring user input]")
                    continue

                audio_key = bot.process_input(transcript)
                
                if audio_key:
                    await player.play(AUDIO_FILES[audio_key])
                    
                if bot.should_disconnect:
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