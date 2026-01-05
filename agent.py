import asyncio
import os
import time
import av
from dotenv import load_dotenv
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
    "greeting": "audio/greeting_new.mp3",
    "closing": "audio/closing_new.mp3",
    "early_exit": "audio/early_closing.mp3",
    "probe_details": "audio/probe_details_short.mp3",
    "ack_1": "audio/hmm.mp3",
    "ack_2": "audio/i_see.mp3",
    "ack_3": "audio/ohh_isit.mp3"
}

SAMPLE_RATE = 48000
NUM_CHANNELS = 1
FRAME_SIZE_SAMPLES = 480

# Initialize the grievance processor
grievance_processor = GrievanceProcessor(db_path="grievances.db")

class AudioFilePlayer:
    def __init__(self, source: rtc.AudioSource):
        self.source = source
        self._current_task = None
        self.is_playing = False
        self._cache = {}

    def preload(self, files_dict):
        """Pre-decode short audio files into memory."""
        print("[INIT] Pre-loading short audio files...")
        for key, path in files_dict.items():
            if "ack" in key or "greeting" in key: 
                try:
                    self._cache[path] = list(self._decode_file(path))
                    print(f"   -> Cached {path}")
                except Exception as e:
                    print(f"   -> Failed to cache {path}: {e}")

    async def play(self, filename: str):
        # Stop current audio
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

        self.is_playing = True
        
        # Check cache vs disk
        if filename in self._cache:
            self._current_task = asyncio.create_task(self._stream_cached(filename))
        else:
            self._current_task = asyncio.create_task(self._stream_from_disk(filename))
            
        await self._current_task
        self.is_playing = False

    def _decode_file(self, filename):
        """Generator that yields audio frames from a file."""
        container = av.open(filename)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        buffer = bytearray()
        
        for frame in container.decode(stream):
            for resampled_frame in resampler.resample(frame):
                buffer.extend(resampled_frame.to_ndarray().tobytes())
                while len(buffer) >= 960:
                    yield buffer[:960]
                    buffer = buffer[960:]
        container.close()

    async def _stream_cached(self, filename):
        """Streams pre-loaded frames from memory."""
        frames = self._cache[filename]
        for chunk_data in frames:
            lk_frame = rtc.AudioFrame(
                data=chunk_data, sample_rate=SAMPLE_RATE, 
                num_channels=NUM_CHANNELS, samples_per_channel=FRAME_SIZE_SAMPLES
            )
            await self.source.capture_frame(lk_frame)
            await asyncio.sleep(0.01)

    async def _stream_from_disk(self, filename):
        """Streams larger files from disk."""
        try:
            for chunk_data in self._decode_file(filename):
                lk_frame = rtc.AudioFrame(
                    data=chunk_data, sample_rate=SAMPLE_RATE, 
                    num_channels=NUM_CHANNELS, samples_per_channel=FRAME_SIZE_SAMPLES
                )
                await self.source.capture_frame(lk_frame)
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error streaming {filename}: {e}")

class GrievanceBotLogic:
    def __init__(self):
        self.state = "greeting"
        self.grievance_text = ""
        self.should_disconnect = False
        self.grievance_timestamp = None
        self.has_played_probe = False  
        self.ack_sounds = ["ack_1", "ack_2", "ack_3"]
        self.save_data = None  # Store data to save instead of triggering immediately

    def process_input(self, text: str) -> str:
        if self.should_disconnect: 
            return None
        
        text_clean = text.strip()
        text_lower = text_clean.lower()
        words = text_lower.split() 
        word_count = len(words)
        
        print(f"\n[USER SAYS] '{text_clean}' (Words: {word_count})")

        # GLOBAL GUARD: IGNORE QUESTIONS
        if text_clean.endswith("?") or text_lower.startswith(("what", "how", "why", "who", "where")):
            print("   -> Detected question/inquiry. Ignoring exit triggers.")
            if self.state == "listening":
                self.grievance_text += " " + text_clean
            return None

        # --- GREETING PHASE ---
        if self.state == "greeting":
            exit_triggers = ["no", "nothing", "nope", "nah"]
            is_pure_refusal = any(t in text_lower for t in exit_triggers) and word_count < 4
            
            if is_pure_refusal:
                print(f"   -> Greeting refusal detected: '{text_clean}'")
                self.state = "closing"
                self.should_disconnect = True
                return "early_exit"
            
            self.state = "listening"
            self.grievance_text += text_clean
            self.grievance_timestamp = time.time()

            if word_count > 7:
                self.has_played_probe = True
                print("   -> Substantial opening detected, playing probe.")
                return "probe_details"
            
            return None

        # --- LISTENING PHASE ---
        elif self.state == "listening":
            self.grievance_text += " " + text_clean
            
            # STRONG EXIT PHRASES (High Confidence)
            strong_exit_phrases = [
                "that's all", "that is all", "that's it", "that is it",
                "nothing else", "nothing more", "i'm done", "i am done",
                "have a good day", "thank you bye", "thanks bye", "thank you", "Thank you", "That's all"
            ]
            
            for phrase in strong_exit_phrases:
                if phrase in text_lower:
                    if text_lower.endswith(phrase) or text_lower.endswith(phrase + "."):
                        print(f"   -> Strong exit phrase detected: '{phrase}'")
                        return self._prepare_exit()
                    
                    idx = text_lower.find(phrase)
                    remainder = text_lower[idx+len(phrase):].strip()
                    if len(remainder.split()) <= 2:
                        print(f"   -> Strong exit phrase detected (w/ tail): '{phrase}'")
                        return self._prepare_exit()

            # CONTEXTUAL TRIGGERS (Medium Confidence)
            if text_lower.endswith("bye") or text_lower.endswith("goodbye"):
                print(f"   -> Farewell detected: '{text_clean}'")
                return self._prepare_exit()
            
            elif text_lower in ["thank you", "thanks", "thank you.", "thanks."]:
                print(f"   -> Solo gratitude detected")
                return self._prepare_exit()

            # EMPATHY & BACKCHANNEL (avoid spamming on noise)
            if word_count < 2:
                print("   -> Fragment/Noise detected. Ignoring.")
                return None

            # Play probe once on substantial input
            if not self.has_played_probe and word_count > 5:
                self.has_played_probe = True
                print("   -> Playing Empathy Probe (Once only)")
                return "probe_details"

            # Backchannel on longer utterances
            if word_count > 7 and random.random() < 0.7:
                selected_ack = random.choice(self.ack_sounds)
                print(f"   -> Backchanneling: {selected_ack}")
                return selected_ack
            
            print("   -> Listening quietly...")
            return None

        return None
    
    def _prepare_exit(self):
        """Prepare for exit. Stores data for background save, returns immediately."""
        print(f"\n{'='*60}")
        print("[PREPARING EXIT] Grievance collected")
        print(f"Words collected: {len(self.grievance_text.split())}")
        print(f"{'='*60}\n")
        
        # Store the data to be saved (don't start any async operations)
        self.save_data = {
            'transcript': self.grievance_text,
            'timestamp': self.grievance_timestamp or time.time()
        }
        
        # Set flags
        self.state = "closing"
        self.should_disconnect = True
        
        # Return immediately
        return "closing"
    
    async def save_grievance_background(self):
        """Save the grievance in background. Call after playing closing audio."""
        if not self.save_data:
            return
            
        try:
            print("[BACKGROUND] Starting grievance processing...")
            result = await grievance_processor.process_and_store(
                transcript=self.save_data['transcript'],
                timestamp=self.save_data['timestamp']
            )
            
            print(f"\n{'='*60}")
            print(f"[SUCCESS] Grievance processed and stored")
            print(f"ID: {result.get('id', 'N/A')}")
            print(f"Summary: {result.get('summary', 'N/A')[:100]}...")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to process grievance: {e}")
        finally:
            self.save_data = None


async def entrypoint(ctx: JobContext):
    print(f"Room created: {ctx.room.name}. Waiting for user...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("bot_voice", source)
    await ctx.room.local_participant.publish_track(track)
    
    # Initialize player and preload
    player = AudioFilePlayer(source)
    player.preload(AUDIO_FILES)

    # Setup STT
    stt_provider = deepgram.STT(
        model="nova-2", 
        language="en", 
        smart_format=True,
        endpointing_ms=500,
        interim_results=True
    )
    stt_stream = stt_provider.stream()
    bot = GrievanceBotLogic()

    last_processed_transcript = ""
    
    async def process_stt_events():
        nonlocal last_processed_transcript
        
        async for event in stt_stream:
            if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                if player.is_playing and len(event.alternatives[0].text) > 5:
                    print("[BARGE-IN] User speaking, stopping audio.")
                
            elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                transcript = event.alternatives[0].text
                if not transcript or transcript == last_processed_transcript:
                    continue
                
                last_processed_transcript = transcript
                audio_key = bot.process_input(transcript)
                
                if audio_key:
                    # Handle closing/early_exit
                    if audio_key in ("closing", "early_exit"):
                        # Play closing FIRST (immediately)
                        print("[CLOSING] Playing farewell message...")
                        await player.play(AUDIO_FILES["closing"])
                        
                        # THEN start background save (after audio is playing/done)
                        print("[CLOSING] Starting background save...")
                        asyncio.create_task(bot.save_grievance_background())
                        
                        # Brief pause then disconnect
                        await asyncio.sleep(0.5)
                        print("[CLOSING] Disconnecting from room...")
                        await ctx.room.disconnect()
                        break
                    else:
                        # Normal audio playback
                        await player.play(AUDIO_FILES[audio_key])
    
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
            
            async def play_greeting():
                await asyncio.sleep(1.5) 
                print("Playing Greeting...")
                await player.play(AUDIO_FILES["greeting"])
            
            asyncio.create_task(play_greeting())

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))