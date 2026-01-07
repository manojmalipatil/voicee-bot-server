import asyncio
import os
import sqlite3
import uuid
from datetime import datetime
from dotenv import load_dotenv

# LiveKit Imports
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import sarvam, groq, silero

# Translation Import
from deep_translator import GoogleTranslator

load_dotenv()

# --- Translation Helper ---
class TranslationManager:
    def __init__(self):
        self.translator_to_english = GoogleTranslator(source='kn', target='en')
        self.translator_to_kannada = GoogleTranslator(source='en', target='kn')
    
    def kannada_to_english(self, text: str) -> str:
        """Translate Kannada text to English."""
        try:
            if not text.strip():
                return text
            translated = self.translator_to_english.translate(text)
            return translated
        except Exception as e:
            print(f"[TRANSLATION ERROR] Kannada to English: {e}")
            return text
    
    def english_to_kannada(self, text: str) -> str:
        """Translate English text to Kannada."""
        try:
            if not text.strip():
                return text
            translated = self.translator_to_kannada.translate(text)
            return translated
        except Exception as e:
            print(f"[TRANSLATION ERROR] English to Kannada: {e}")
            return text

# --- Database Manager ---
class DatabaseManager:
    def __init__(self, db_path="grievance.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Added original_language column to track the language
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grievances (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                transcript TEXT NOT NULL,
                original_language TEXT DEFAULT 'kannada',
                category TEXT,
                priority TEXT,
                sentiment TEXT,
                summary TEXT,
                tags TEXT,
                created_at TEXT NOT NULL,
                location TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()
        conn.close()

    def save_grievance(self, transcript: str, language: str = "kannada"):
        """Save the transcript, generating ID and timestamps automatically."""
        if not transcript.strip():
            print("[DB] Transcript is empty, skipping save.")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        record_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        try:
            cursor.execute("""
                INSERT INTO grievances (id, timestamp, transcript, original_language, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (record_id, current_time, transcript, language, current_time))
            
            conn.commit()
            print(f"[DB] Successfully saved grievance ID: {record_id}")
        except Exception as e:
            print(f"[DB] Error saving grievance: {e}")
        finally:
            conn.close()

# --- Kannada System Prompt ---
SYSTEM_INSTRUCTIONS = """ನೀವು "HR Voice Assistant", ಒಂದು ವೃತ್ತಿಪರ ಮತ್ತು ಸಹಾನುಭೂತಿಯುಳ್ಳ ಕುಂದುಕೊರತೆ ಸಂಗ್ರಾಹಕರು. ಸಲಹೆ ಅಥವಾ ಪರಿಹಾರಗಳನ್ನು ನೀಡದೆ ಕುಂದುಕೊರತೆಗಳನ್ನು ಪರಿಣಾಮಕಾರಿಯಾಗಿ ದಾಖಲಿಸುವುದು ನಿಮ್ಮ ಗುರಿ.

ಪ್ರಮುಖ ನಡವಳಿಕೆಗಳು:
- **ಶೈಲಿ:** ಸ್ನೇಹಪರ ಆದರೆ ಸಂಕ್ಷಿಪ್ತ (ಗರಿಷ್ಠ 2 ವಾಕ್ಯಗಳು). ನೈಸರ್ಗಿಕ, ಕನಿಷ್ಠ ಅಂಗೀಕಾರಗಳನ್ನು ಬಳಸಿ ("ಅರ್ಥವಾಯಿತು," "ಮುಂದುವರಿಯಿರಿ").
- **ಅಗತ್ಯ ಮಾಹಿತಿ:** ನೀವು ಸಂಗ್ರಹಿಸಬೇಕಾದದ್ದು: 1) ಕುಂದುಕೊರತೆ, 2) ಸ್ಥಳ, ಮತ್ತು 3) ವಿಭಾಗ.
- **ಕಾಣೆಯಾದ ಮಾಹಿತಿ:** ಬಳಕೆದಾರರು ಸ್ಥಳ ಅಥವಾ ವಿಭಾಗವನ್ನು ಬಿಟ್ಟರೆ, ಅವರನ್ನು ಸ್ವಾಭಾವಿಕವಾಗಿ *ಒಮ್ಮೆ* ಕೇಳಿ.

ಹರಿವು:
1. **ಸ್ವಾಗತ:** ಸಂಕ್ಷಿಪ್ತ ಸ್ವಾಗತ + ಅವರನ್ನು ಮಾತನಾಡಲು ಆಹ್ವಾನಿಸಿ.
2. **ಆಲಿಸಿ:** ಅಡಚಣೆಯಿಲ್ಲದೆ ಅಂಗೀಕರಿಸಿ.
3. **ಮುಕ್ತಾಯ:** ಬಳಕೆದಾರರು ತಾವು ಪೂರ್ಣಗೊಳಿಸಿದ್ದೇನೆ ಎಂದು ಸೂಚಿಸಿದಾಗ (ಉದಾ. "ಅಷ್ಟೇ," "ನಾನು ಮುಗಿಸಿದ್ದೇನೆ," ಅಥವಾ "ಧನ್ಯವಾದಗಳು"):
   - ಪ್ರತಿಕ್ರಿಯಿಸಿ: "ಇದನ್ನು ಹಂಚಿಕೊಂಡಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ಕುಂದುಕೊರತೆಯನ್ನು ದಾಖಲಿಸಲಾಗಿದೆ ಮತ್ತು ಪರಿಶೀಲಿಸಲಾಗುವುದು. ಎಚ್ಚರದಿಂದಿರಿ."
   - **ತಕ್ಷಣ** `end_call` ಕಾರ್ಯವನ್ನು ಕರೆ ಮಾಡಿ.

ನಿರ್ಬಂಧಗಳು:
- ಸಮಸ್ಯೆಗಳನ್ನು ಪರಿಹರಿಸಬೇಡಿ ಅಥವಾ ಸಲಹೆ ನೀಡಬೇಡಿ.
- "ಬೇರೇನಾದರೂ ಇದೆಯೇ?" ಎಂದು ಮತ್ತೆ ಕೇಳಬೇಡಿ.
- ಬಳಕೆದಾರರು ಪೂರ್ಣತೆಯನ್ನು ಸೂಚಿಸಿದಾಗ ಮಾತ್ರ `end_call` ಅನ್ನು ಕರೆ ಮಾಡಿ.

ಮುಕ್ತಾಯ ಅನುಕ್ರಮ (100% ಖಚಿತತೆಗೆ ಮಾತ್ರ):
1. ಹೇಳಿ: "ಇದನ್ನು ಹಂಚಿಕೊಂಡಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ಕುಂದುಕೊರತೆಯನ್ನು ದಾಖಲಿಸಲಾಗಿದೆ ಮತ್ತು ನಮ್ಮ ತಂಡದಿಂದ ಪರಿಶೀಲಿಸಲಾಗುವುದು. ಎಚ್ಚರದಿಂದಿರಿ."
2. "grievance_complete" ಕಾರಣದೊಂದಿಗೆ end_call ಅನ್ನು ಕರೆ ಮಾಡಿ

ನೆನಪಿಡಿ: ನಿಮ್ಮ ಕೆಲಸ ಸಂಪೂರ್ಣ ಮಾಹಿತಿಯನ್ನು ಸಂಗ್ರಹಿಸುವುದು. ತಾಳ್ಮೆ ಮತ್ತು ಸಂಪೂರ್ಣತೆಯಿಂದಿರಿ."""


class GrievanceTracker:
    """Track grievance conversation and manage call state."""
    
    def __init__(self, translator: TranslationManager):
        self.translator = translator
        self.grievance_text_kannada = []  # Original Kannada
        self.grievance_text_english = []  # Translated English
        self.conversation_history = []
        self.word_count = 0
    
    def add_user_message(self, text: str):
        """Add user message in Kannada and translate to English."""
        # Store original Kannada
        self.grievance_text_kannada.append(f"ಉದ್ಯೋಗಿ: {text}")
        
        # Translate to English and store
        english_text = self.translator.kannada_to_english(text)
        self.grievance_text_english.append(f"Employee: {english_text}")
        
        self.conversation_history.append({"role": "user", "content": text})
        self.word_count += len(text.split())
        
        print(f"[TRANSLATION] Kannada: {text[:50]}...")
        print(f"[TRANSLATION] English: {english_text[:50]}...")
    
    def add_agent_message(self, text: str):
        """Add agent message to history."""
        self.conversation_history.append({"role": "assistant", "content": text})
    
    def get_full_grievance_kannada(self) -> str:
        """Get the complete grievance transcript in Kannada."""
        return "\n".join(self.grievance_text_kannada)
    
    def get_full_grievance_english(self) -> str:
        """Get the complete grievance transcript in English (translated)."""
        return "\n".join(self.grievance_text_english)
    
    def get_stats(self) -> dict:
        """Get grievance statistics."""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len([m for m in self.conversation_history if m["role"] == "user"]),
            "word_count": self.word_count,
        }


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice agent."""
    
    print(f"[ROOM] Connecting to room: {ctx.room.name}")
    
    # Initialize Database and Translation
    db_manager = DatabaseManager()
    translator = TranslationManager()

    # Connect to the room
    await ctx.connect()
    
    # Initialize the grievance tracker with translator
    grievance_tracker = GrievanceTracker(translator)
    
    # Flag to track if call should end
    should_end_call = asyncio.Event()
    
    # Define the end_call function tool
    @function_tool
    async def end_call(
        confirmation: str = "yes"
    ):
        """
        ಬಳಕೆದಾರರು ಪೂರ್ಣಗೊಳಿಸಿದ್ದೇನೆ ಎಂದು ಸೂಚಿಸಿದಾಗ ಅಥವಾ ಸಂವಾದ ಮುಗಿದ ನಂತರ ಕುಂದುಕೊರತೆ ಸಂಗ್ರಹಣೆ ಕರೆಯನ್ನು ಮುಕ್ತಾಯಗೊಳಿಸಿ.
        
        Args:
            confirmation: ಕರೆಯನ್ನು ಮುಕ್ತಾಯಗೊಳಿಸಲು ದೃಢೀಕರಣ (ಪೂರ್ವನಿಯೋಜಿತ: "yes")
        """
        print("[FUNCTION] end_call function invoked by LLM")
        should_end_call.set()
        return "ಕರೆ ಮುಕ್ತಾಯ ಪ್ರಾರಂಭಿಸಲಾಗಿದೆ. ಕುಂದುಕೊರತೆ ದಾಖಲಿಸಲಾಗಿದೆ."
    
    # Create the agent with Kannada instructions
    agent = Agent(
        instructions=SYSTEM_INSTRUCTIONS,
        tools=[end_call],
    )
    
    # Create agent session with Kannada language support
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=0.8,
        ),
        stt=groq.STT(
            model="whisper-large-v3-turbo",
            language="kn",
        ),
        llm=groq.LLM(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
        ),
        tts=sarvam.TTS(
            target_language_code="kn-IN",
            speaker="anushka",
        )
    )
    
    # Event handlers for tracking conversation
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        """Called when a conversation item is added (user or agent)."""
        item = event.item
        
        if item.role == "user":
            # User message in Kannada
            text = item.text_content or ""
            if text:
                print(f"\n[USER - ಕನ್ನಡ] {text}")
                grievance_tracker.add_user_message(text)
        elif item.role == "assistant":
            # Agent message - only print complete, non-interrupted messages
            text = item.text_content or ""
            if text and not item.interrupted:
                print(f"[AGENT - ಕನ್ನಡ] {text}")
                grievance_tracker.add_agent_message(text)
    
    @session.on("function_calls_finished")
    def on_function_calls_finished(called_functions):
        """Called when LLM finishes executing function calls."""
        for func in called_functions:
            print(f"[FUNCTION] Completed: {func.call_info.function_info.name}")
    
    print("[SESSION] Starting agent session...")
    
    # Start the session as a background task
    session_task = asyncio.create_task(session.start(agent=agent, room=ctx.room))
    
    print("[SESSION] Waiting for participant to join...")
    
    # Wait for participant to join
    while len(ctx.room.remote_participants) == 0:
        await asyncio.sleep(0.1)
    
    print("[SESSION] Participant joined. Waiting for session to initialize...")
    
    # Wait for session to be ready
    await asyncio.sleep(1.5)
    
    print("[SESSION] Sending initial greeting...")
    
    # Send initial greeting in Kannada
    try:
        await session.generate_reply(
            instructions="ಸಂಕ್ಷಿಪ್ತ, ಸ್ನೇಹಪರ ಸ್ವಾಗತ ನೀಡಿ ಮತ್ತು ಅವರ ಕುಂದುಕೊರತೆಯನ್ನು ಹಂಚಿಕೊಳ್ಳಲು ಹೇಳಿ. ಕೇವಲ ಒಂದು ವಾಕ್ಯ ಮಾತ್ರ."
        )
    except Exception as e:
        print(f"[ERROR] Failed to generate greeting: {e}")
    
    print("[AGENT] Ready to collect grievances in Kannada...")
    
    try:
        # Wait only for the end_call signal
        await should_end_call.wait()
        
        print("[CLOSING] end_call triggered. Waiting for final message...")
        
        # Give enough time for the final message to be generated and spoken
        await asyncio.sleep(6.5)
        
        print("[CLOSING] Proceeding with disconnect")
        
        # Now cancel the session
        session_task.cancel()
        try:
            await session_task
        except asyncio.CancelledError:
            pass
        
    except asyncio.CancelledError:
        print("[SESSION] Session cancelled")
        session_task.cancel()
        try:
            await session_task
        except asyncio.CancelledError:
            pass
    finally:
        # Print final grievance in both languages
        stats = grievance_tracker.get_stats()
        full_grievance_kannada = grievance_tracker.get_full_grievance_kannada()
        full_grievance_english = grievance_tracker.get_full_grievance_english()
        
        print("\n" + "="*70)
        print("[GRIEVANCE COLLECTION COMPLETE]")
        print("="*70)
        print(f"Total Messages: {stats['total_messages']}")
        print(f"User Messages: {stats['user_messages']}")
        print(f"Word Count: {stats['word_count']}")
        print("-"*70)
        print("[ORIGINAL - KANNADA]")
        print(full_grievance_kannada)
        print("-"*70)
        print("[TRANSLATED - ENGLISH]")
        print(full_grievance_english)
        print("="*70 + "\n")
        
        # --- DATABASE STORAGE LOGIC (English translation) ---
        print("[DB] Saving English translation to database...")
        db_manager.save_grievance(full_grievance_english, language="kannada")
        # ---------------------------------------------------
        
        print("[DISCONNECT] Closing connection...")
        await ctx.room.disconnect()
        print("[SESSION] Session ended")


if __name__ == "__main__":
    # Run the agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )