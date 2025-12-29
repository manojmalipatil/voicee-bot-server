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
    "greeting": "audio/greeting_new.mp3",
    "closing": "audio/closing_new.mp3",
    "early_exit": "audio/early_closing.mp3",
    "probe_details": "audio/probe_details_short.mp3", # Your long "Thank you for sharing..." script
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
    def __init__(self, source: rtc.AudioSource):
        self.source = source
        self._current_task = None
        self.is_playing = False
        self._cache = {}  # Store decoded audio frames here

    def preload(self, files_dict):
        """Pre-decode short audio files into memory."""
        print("[INIT] Pre-loading short audio files...")
        for key, path in files_dict.items():
            # Pre-load only short acks to save RAM, stream long ones
            if "ack" in key or "greeting" in key: 
                try:
                    self._cache[path] = list(self._decode_file(path))
                    print(f"   -> Cached {path}")
                except Exception as e:
                    print(f"   -> Failed to cache {path}: {e}")

    async def play(self, filename: str):
        # 1. Stop current audio (Handle interruption)
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

        self.is_playing = True
        
        # 2. Check Cache
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
        """Streams pre-loaded frames from memory (Zero Latency)."""
        frames = self._cache[filename]
        for chunk_data in frames:
            lk_frame = rtc.AudioFrame(
                data=chunk_data, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS, samples_per_channel=FRAME_SIZE_SAMPLES
            )
            await self.source.capture_frame(lk_frame)
            await asyncio.sleep(0.01) # Maintain timing

    async def _stream_from_disk(self, filename):
        """Streams larger files from disk."""
        # Use the same generator logic but with async sleep
        try:
            for chunk_data in self._decode_file(filename):
                lk_frame = rtc.AudioFrame(
                    data=chunk_data, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS, samples_per_channel=FRAME_SIZE_SAMPLES
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
        
        # ACTIVE LISTENING STATE
        self.has_played_probe = False  
        self.ack_sounds = ["ack_1", "ack_2", "ack_3"] 

    def process_input(self, text: str) -> str:
        if self.should_disconnect: 
            return None
        
        text_clean = text.strip()
        text_lower = text_clean.lower()
        words = text_lower.split() 
        word_count = len(words)
        
        print(f"\n[USER SAYS] '{text_clean}' (Words: {word_count})")

        # --- GLOBAL GUARD: IGNORE QUESTIONS ---
        # If the user asks a question, they are definitely not leaving.
        # e.g., "Are we done?" or "What do you think?"
        if text_clean.endswith("?") or text_lower.startswith(("what", "how", "why", "who", "where")):
            print("   -> Detected question/inquiry. Ignoring exit triggers.")
            if self.state == "listening":
                self.grievance_text += " " + text_clean
            return None

        # --- 1. GREETING PHASE ---
        if self.state == "greeting":
            # Only exit if the response is explicitly negative and short
            exit_triggers = ["no", "nothing", "nope", "nah"]
            
            # Check if the ENTIRE sentence is basically just a refusal
            # e.g., "No." or "Nothing really." -> Exit
            # e.g., "No, I actually have a big problem." -> Stay
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

        # --- 2. LISTENING PHASE ---
        elif self.state == "listening":
            self.grievance_text += " " + text_clean
            
            should_exit_listening = False
            
            # --- SAFE EXIT CHECK ---
            
            # 1. Strong Phrases (High Confidence)
            # These are specific enough that they rarely occur by accident.
            strong_exit_phrases = [
                "that's all", "that is all", "that's it", "that is it",
                "nothing else", "nothing more", "i'm done", "i am done",
                "have a good day", "thank you bye", "thanks bye"
            ]
            
            for phrase in strong_exit_phrases:
                if phrase in text_lower:
                    # Ensure the phrase is at the END of the sentence
                    # e.g. "That's all I have to say" -> OK
                    # e.g. "That's all the money I have" -> NOT OK
                    if text_lower.endswith(phrase) or text_lower.endswith(phrase + "."):
                        should_exit_listening = True
                        print(f"   -> Strong exit phrase detected: '{phrase}'")
                        break
                    
                    # Also check if it's followed by very few words (garbage/fillers)
                    idx = text_lower.find(phrase)
                    remainder = text_lower[idx+len(phrase):].strip()
                    if len(remainder.split()) <= 2:
                        should_exit_listening = True
                        print(f"   -> Strong exit phrase detected (w/ tail): '{phrase}'")
                        break

            # 2. Contextual Triggers (Medium Confidence)
            # Only trigger these if they stand ALONE or are clearly ending the thought.
            if not should_exit_listening:
                # "bye" and "goodbye" are usually safe if they appear at the end
                if text_lower.endswith("bye") or text_lower.endswith("goodbye"):
                    should_exit_listening = True
                    print(f"   -> Farewell detected: '{text_clean}'")

                # "Thanks" / "Thank you" are DANGEROUS. 
                # Only accept if purely isolated: "Okay, thank you." 
                elif text_lower in ["thank you", "thanks", "thank you.", "thanks."]:
                    # We only exit on "thanks" if the PREVIOUS utterance was also short/closing-like
                    # OR if we simply assume a solo "Thank you" is a close.
                    # Safest approach: Treat solo "Thank you" as an exit.
                    should_exit_listening = True
                    print(f"   -> Solo gratitude detected")

            # --- EXECUTE EXIT ---
            if should_exit_listening:
                print("   -> Exiting listening phase")
                self._save_and_exit()
                return "closing"

            # --- B. EMPATHY & BACKCHANNEL ---
            
            # Ignore very short fragments to prevent spamming "hmm" on noise
            if word_count < 2:
                print("   -> Fragment/Noise detected. Ignoring.")
                return None

            # Probe Logic (Once only)
            if not self.has_played_probe:
                if word_count > 5:
                    self.has_played_probe = True
                    print("   -> Playing Empathy Probe (Once only)")
                    return "probe_details"
                return None

            # Backchannel Logic
            else:
                # Only backchannel on longer sentences to avoid interrupting flow
                if word_count > 7:
                    if random.random() < 0.7: # 70% chance
                        selected_ack = random.choice(self.ack_sounds)
                        print(f"   -> Backchanneling: {selected_ack}")
                        return selected_ack
                
                print("   -> Listening quietly...")
                return None

        return None
    
    def _save_and_exit(self):
        """Save the grievance and prepare for exit."""
        print(f"\n{'='*60}")
        print("[SAVING] Grievance collected")
        print(f"Words collected: {len(self.grievance_text.split())}")
        print(f"{'='*60}\n")
        
        # Fire and forget the background task (using the fixed await logic from previous step)
        asyncio.create_task(self._process_grievance_background())
        
        self.state = "closing"
        self.should_disconnect = True
    
    async def _process_grievance_background(self):
        """Process grievance in background without blocking exit."""
        try:
            print("[BACKGROUND] Starting grievance processing...")
            result = await grievance_processor.process_and_store(
                transcript=self.grievance_text,
                timestamp=self.grievance_timestamp or time.time()
            )
            
            print(f"\n{'='*60}")
            print(f"[SUCCESS] Grievance processed and stored")
            print(f"ID: {result.get('id', 'N/A')}")
            print(f"Summary: {result.get('summary', 'N/A')[:100]}...")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to process grievance: {e}")
    
    async def _process_grievance(self):
        """Process grievance with LLM and store in database."""
        try:
            result = await grievance_processor.process_and_store(
                transcript=self.grievance_text,
                timestamp=self.grievance_timestamp or time.time()
            )
            
            print(f"\n{'='*60}")
            print(f"[SUCCESS] Grievance processed and stored")
            print(f"ID: {result.get('id', 'N/A')}")
            print(f"Summary: {result.get('summary', 'N/A')[:100]}...")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to process grievance: {e}")

async def entrypoint(ctx: JobContext):
    print(f"Room created: {ctx.room.name}. Waiting for user...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("bot_voice", source)
    await ctx.room.local_participant.publish_track(track)
    
    # 1. Initialize Player and Preload
    player = AudioFilePlayer(source)
    player.preload(AUDIO_FILES) # <--- Pre-load acks into RAM

    # 2. Optimized Deepgram Config
    stt_provider = deepgram.STT(
        model="nova-2", 
        language="en", 
        smart_format=True,
        endpointing_ms=500,  # <--- CHANGED from 1500 to 500 (Massive speedup)
        interim_results=True # <--- Set to True so you know when they START speaking
    )
    stt_stream = stt_provider.stream()
    bot = GrievanceBotLogic()

    async def process_stt_events():
        async for event in stt_stream:
            if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                if player.is_playing:
                    # If user starts speaking, stop the bot immediately?
                    # Ideally you check if transcript length > 2 chars to avoid noise
                    if len(event.alternatives[0].text) > 5:
                         print("[BARGE-IN] User speaking, stopping audio.")
                         player.stop() # logic to stop player could go here
                    pass
                
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                transcript = event.alternatives[0].text
                if not transcript: continue

                audio_key = bot.process_input(transcript)
                
                if audio_key:
                    await player.play(AUDIO_FILES[audio_key])
                    
                    if audio_key == "closing" or audio_key == "early_exit":
                        print("[CLOSING] Playing farewell message...")
                        await asyncio.sleep(1.0)  # Brief pause after closing audio finishes
                        print("[CLOSING] Disconnecting from room...")
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