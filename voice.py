import asyncio
import os
import time
import av
from dotenv import load_dotenv
import json
import random

# LiveKit Imports
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe, stt
from livekit.plugins import deepgram

# Import our grievance processor
from grievance_processor import GrievanceProcessor

load_dotenv()

# --- CONFIGURATION ---
AUDIO_FILES = {
    "greeting": "audio/greetingf.mp3",
    "closing": "audio/closingf.mp3",
    "probe_details": "audio/continuef.mp3", # Your long "Thank you for sharing..." script
    "ack_1": "audio/hmm.mp3",              # Short sound (0.5s)
    "ack_2": "audio/i_see.mp3",            # Short sound (0.5s)
    "ack_3": "audio/ohh_isit.mp3"             # Short sound (0.5s)
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
        self.grievance_timestamp = None
        
        # ACTIVE LISTENING STATE
        self.has_played_probe = False  # Tracks if we've played the long empathy script
        self.ack_sounds = ["ack_1", "ack_2", "ack_3"] # Keys from AUDIO_FILES

    def process_input(self, text: str) -> str:
        if self.should_disconnect: 
            return None
        
        text_lower = text.lower().strip()
        word_count = len(text_lower.split())
        
        print(f"\n[USER SAYS] '{text}' (Words: {word_count})")

        # --- 1. GREETING PHASE ---
        if self.state == "greeting":
            if any(w in text_lower for w in ["bye", "exit", "nothing", "no"]):
                self.state = "closing"
                self.should_disconnect = True
                return "closing"
            
            # Transition to listening
            self.state = "listening"
            self.grievance_text += text
            self.grievance_timestamp = time.time()

            # IMPATIENCE CHECK:
            # If the user's *first* utterance is long (e.g. they skipped hello and went straight to the issue),
            # we should play the probe immediately.
            if word_count > 8:
                self.has_played_probe = True
                print("   -> substantial opening detected, playing probe.")
                return "probe_details"
            
            return None

        # --- 2. LISTENING PHASE ---
        elif self.state == "listening":
            self.grievance_text += " " + text
            
            # A. EXIT CHECK
            exit_words = ["done", "finished", "that's all", "that's it", "thank you", "thanks", "bye"]
            if any(w in text_lower for w in exit_words):
                print(f"[SAVING REPORT] {len(self.grievance_text)} chars")
                asyncio.create_task(self._process_grievance()) # Fire and forget DB storage
                self.state = "closing"
                self.should_disconnect = True
                return "closing"

            # B. NOISE FILTER
            # Ignore tiny fragments like "um" or breath sounds (unless it's 'yes'/'no')
            if word_count < 3 and text_lower not in ['yes', 'no', 'yeah']:
                print("   -> Fragment detected. Ignoring.")
                return None 
            
            # C. THE LOGIC SPLIT
            
            # Scenario 1: The "Probe" (Deep Empathy)
            # We play this ONLY ONCE, the first time the user pauses after a real sentence.
            if not self.has_played_probe:
                # Ensure they actually said something substantial (>4 words) before we probe
                if word_count > 5:
                    self.has_played_probe = True
                    print("   -> Playing Empathy Probe (Once only)")
                    return "probe_details" # "Thank you for sharing... tell me more"
                else:
                    return None # Wait for them to say more before probing

            # Scenario 2: The "Backchannel" (Nodding along)
            # We've already probed. Now we just encourage them to keep going.
            else:
                # We don't want to ack EVERY single pause (it sounds robotic).
                # 60% chance to say "hmm", 40% chance to stay silent.
                if random.random() < 0.7:
                    selected_ack = random.choice(self.ack_sounds)
                    print(f"   -> Backchanneling: {selected_ack}")
                    return selected_ack
                
                print("   -> Listening quietly...")
                return None

        return None
    
    async def _process_grievance(self):
        """Process grievance with LLM and store in database."""
        try:
            # Assumes 'grievance_processor' is available in global scope or passed in
            result = await grievance_processor.process_and_store(
                transcript=self.grievance_text,
                timestamp=self.grievance_timestamp or time.time()
            )
            
            print(f"\n{'='*60}")
            print(f"[SUCCESS] Grievance processed and stored")
            print(f"ID: {result.get('id', 'N/A')}")
            print(f"Summary: {result.get('summary', 'N/A')}")
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
        endpointing_ms=1500,
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