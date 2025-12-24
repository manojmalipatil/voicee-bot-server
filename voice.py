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
        words = text_lower.split() 
        word_count = len(words)
        
        print(f"\n[USER SAYS] '{text}' (Words: {word_count})")

        # --- 1. GREETING PHASE ---
        if self.state == "greeting":
            exit_triggers = ["no", "nothing", "nope", "nah", "bye", "exit", "goodbye"]
            should_exit_greeting = False
            
            for i, w in enumerate(words):
                clean_w = w.strip(".,!?")
                if clean_w in exit_triggers:
                    words_after = len(words) - 1 - i
                    if words_after < 2:
                        should_exit_greeting = True
                        print(f"   -> Greeting exit trigger: '{clean_w}'")
                        break

            if should_exit_greeting:
                self.state = "closing"
                self.should_disconnect = True
                return "closing"
            
            self.state = "listening"
            self.grievance_text += text
            self.grievance_timestamp = time.time()

            if word_count > 8:
                self.has_played_probe = True
                print("   -> substantial opening detected, playing probe.")
                return "probe_details"
            
            return None

        # --- 2. LISTENING PHASE ---
        elif self.state == "listening":
            self.grievance_text += " " + text
            
            # A. EXIT CHECK - Multi-layered approach
            should_exit_listening = False
            
            # Layer 1: Exit phrases (high confidence)
            exit_phrases = [
                "that's all", "that is all", "that's it", "that is it",
                "i'm done", "i am done", "im done",
                "that's everything", "that is everything",
                "nothing else", "nothing more",
                "that's all i have", "that's all i wanted to say"
            ]
            
            for phrase in exit_phrases:
                if phrase in text_lower:
                    # Check if phrase is at the end or followed by minimal words
                    phrase_pos = text_lower.rfind(phrase)
                    text_after_phrase = text_lower[phrase_pos + len(phrase):].strip()
                    words_after_phrase = len([w for w in text_after_phrase.split() if w])
                    
                    if words_after_phrase < 3:  # Max 2 words after the phrase (e.g., "that's all, bye")
                        should_exit_listening = True
                        print(f"   -> Exit phrase detected: '{phrase}'")
                        break
            
            # Layer 2: Single word exit triggers (medium confidence)
            if not should_exit_listening:
                finish_triggers = [
                    "done", "finished", "complete", "over",
                    "bye", "goodbye", "thanks", "thank you", 
                    "okay", "ok", "alright"
                ]
                
                for i, w in enumerate(words):
                    clean_w = w.strip(".,!?")
                    if clean_w in finish_triggers:
                        words_after = len(words) - 1 - i
                        
                        # Stricter check for ambiguous words like "okay" and "thanks"
                        if clean_w in ["okay", "ok", "alright", "thanks", "thank you"]:
                            # These need to be at the very end (0-1 words after)
                            if words_after <= 1:
                                should_exit_listening = True
                                print(f"   -> Exit trigger detected: '{clean_w}' (end of speech)")
                                break
                        else:
                            # Other triggers allow up to 2 words after
                            if words_after < 3:
                                should_exit_listening = True
                                print(f"   -> Exit trigger detected: '{clean_w}'")
                                break
            
            # Layer 3: Context-aware exit detection
            # If user repeats similar sentiment (e.g., "that's all" after "thanks")
            if not should_exit_listening and hasattr(self, '_last_utterance'):
                confirmation_words = ["yes", "yeah", "yep", "correct", "right", "exactly"]
                if text_lower in confirmation_words and word_count == 1:
                    # Single confirmation word after a potential exit phrase
                    should_exit_listening = True
                    print(f"   -> Exit confirmation detected: '{text_lower}'")
            
            # Store last utterance for context
            self._last_utterance = text_lower
            
            # CRITICAL: Exit immediately if exit condition is met
            if should_exit_listening:
                print("   -> Exiting listening phase")
                self._save_and_exit()
                return "closing"
                
            # B. NOISE FILTER - Improved
            # Ignore tiny fragments, but allow emotional expressions
            if word_count < 2:
                allowed_short = ['yes', 'no', 'yeah', 'okay', 'ok']
                if text_lower not in allowed_short:
                    print("   -> Fragment detected. Ignoring.")
                    return None
                # If it's "yes" or "no" in isolation, treat cautiously
                elif text_lower in ['yes', 'yeah', 'okay', 'ok']:
                    # Could be confirmation of being done, but we'll wait for more
                    print("   -> Short affirmation, continuing to listen...")
                    return None
            
            # C. EMPATHY & BACKCHANNEL LOGIC - Enhanced
            
            # Scenario 1: The "Probe" (Deep Empathy)
            if not self.has_played_probe:
                # Only probe after substantial content (more than basic greeting)
                if word_count > 5:
                    self.has_played_probe = True
                    print("   -> Playing Empathy Probe (Once only)")
                    return "probe_details"
                else:
                    # Wait for more before probing
                    return None

            # Scenario 2: The "Backchannel" (Active listening)
            else:
                # Intelligent backchanneling based on content length
                # Longer utterances = more likely to acknowledge
                if word_count > 10:
                    # Substantial content, high chance of acknowledgment
                    ack_probability = 0.8
                elif word_count > 5:
                    # Medium content, medium chance
                    ack_probability = 0.6
                else:
                    # Short content, lower chance (might be trailing off)
                    ack_probability = 0.3
                
                if random.random() < ack_probability:
                    selected_ack = random.choice(self.ack_sounds)
                    print(f"   -> Backchanneling: {selected_ack} (prob: {ack_probability})")
                    return selected_ack
                
                print("   -> Listening quietly...")
                return None

        return None
    
    def _save_and_exit(self):
        """Save the grievance and prepare for exit."""
        print(f"\n{'='*60}")
        print("[SAVING] Grievance collected")
        print(f"Words collected: {len(self.grievance_text.split())}")
        print(f"Text preview: {self.grievance_text[:100]}...")
        print(f"{'='*60}\n")
        
        # Schedule async processing
        asyncio.create_task(self._process_grievance())
        
        self.state = "closing"
        self.should_disconnect = True
    
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