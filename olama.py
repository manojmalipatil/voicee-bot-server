import asyncio
import os
import sqlite3
import uuid
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
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
from livekit.plugins import sarvam, openai, silero, groq

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GrievanceBot")

# --- Load Language Configuration ---
def load_language_config(config_path="language_config.json"):
    """Load language configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"[CONFIG] Successfully loaded language config from {config_path}")
        return config
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in config file: {e}")
        raise

LANGUAGE_CONFIG = load_language_config()
SUPPORTED_LANGUAGES = LANGUAGE_CONFIG["supported_languages"]
LANGUAGE_SELECTION_PROMPT = LANGUAGE_CONFIG["language_selection_prompt"]
GRIEVANCE_PROMPTS = LANGUAGE_CONFIG["grievance_prompts"]

# --- Database Manager (Async Wrapper) ---
class DatabaseManager:
    def __init__(self, db_path="grievance.db"):
        self.db_path = db_path
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.init_db()

    def init_db(self):
        """Initialize the database table synchronously at startup."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grievances (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                transcript TEXT NOT NULL,
                language TEXT,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()
        conn.close()

    async def save_grievance(self, transcript: str, language: str = "en"):
        """Save the transcript asynchronously to avoid blocking voice loop."""
        if not transcript.strip():
            return

        def _save():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            record_id = str(uuid.uuid4())
            current_time = datetime.now().isoformat()
            try:
                cursor.execute("""
                    INSERT INTO grievances (id, timestamp, transcript, language, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (record_id, current_time, transcript, language, current_time))
                conn.commit()
                logger.info(f"[DB] Saved grievance ID: {record_id} (Lang: {language})")
            except Exception as e:
                logger.error(f"[DB] Error saving grievance: {e}")
            finally:
                conn.close()

        # Run blocking SQL in a separate thread
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _save)


class GrievanceTracker:
    """Track grievance conversation."""
    def __init__(self):
        self.grievance_text = []
        self.conversation_history = []
        self.selected_language = None
    
    def set_language(self, language_code: str):
        self.selected_language = language_code
    
    def add_user_message(self, text: str):
        self.grievance_text.append(f"Employee: {text}")
        self.conversation_history.append({"role": "user", "content": text})
    
    def add_agent_message(self, text: str):
        self.grievance_text.append(f"Agent: {text}")
        self.conversation_history.append({"role": "assistant", "content": text})
    
    def get_full_grievance(self) -> str:
        return "\n".join(self.grievance_text)


async def entrypoint(ctx: JobContext):
    logger.info(f"[ROOM] Connecting to room: {ctx.room.name}")
    
    db_manager = DatabaseManager()
    await ctx.connect()
    
    grievance_tracker = GrievanceTracker()
    
    # State flags
    should_end_call = asyncio.Event()
    language_selected = asyncio.Event()
    selected_lang_code = "en"
    
    # --- 1. THE SILENT TOOL ---
    @function_tool
    async def select_language(language: str):
        """
        Select the language. Call this immediately when the user answers.
        """
        nonlocal selected_lang_code
        lang_lower = language.lower().strip()
        
        # Logic to find code
        found_code = "en"
        for name, data in SUPPORTED_LANGUAGES.items():
            if name in lang_lower:
                found_code = data["code"]
                break
        
        selected_lang_code = found_code
        grievance_tracker.set_language(selected_lang_code)
        
        # Trigger the main loop to kill this session
        language_selected.set()
        
        return "TERMINATE_SESSION" 

    @function_tool
    async def end_call(confirmation: str = "yes"):
        should_end_call.set()
        return "Call ending initiated."

    # --- STAGE 1: SELECTION AGENT ---
    logger.info("[START] Stage 1: Language Selection")
    
    session_1 = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=0.5,
        ),
        stt=groq.STT(model="whisper-large-v3-turbo", language="en"),  # Groq Whisper for STT
        llm=openai.LLM.with_ollama(
            model="qwen2.5:7b",
            base_url="http://localhost:11434/v1",
        ),  # Local Ollama LLM
        tts=sarvam.TTS(target_language_code="en-IN", speaker="anushka"),
    )
    
    language_agent = Agent(
        instructions=LANGUAGE_SELECTION_PROMPT,
        tools=[select_language],
    )

    # Start Session 1
    task_1 = asyncio.create_task(session_1.start(agent=language_agent, room=ctx.room))
    
    # Initial Greeting
    await asyncio.sleep(1.0)
    await session_1.generate_reply(
        instructions="Ask the user if they prefer English, Hindi, or Tamil."
    )

    # Wait for the tool to be called
    try:
        await asyncio.wait_for(language_selected.wait(), timeout=60.0)
    except asyncio.TimeoutError:
        selected_lang_code = "en"

    # --- 2. PROPER SESSION SHUTDOWN ---
    logger.info("[SHUTDOWN] Stopping language selection agent...")
    
    # Step 1: Stop the session (this is the key missing piece!)
    try:
        await session_1.aclose()
        logger.info("[SHUTDOWN] Session 1 closed successfully")
    except Exception as e:
        logger.error(f"[SHUTDOWN] Error closing session 1: {e}")
    
    # Step 2: Cancel the task
    task_1.cancel()
    try:
        await task_1
    except asyncio.CancelledError:
        logger.info("[SHUTDOWN] Task 1 cancelled")
    
    # Step 3: Cleanup reference
    del session_1
    del language_agent
    
    # Step 4: CRITICAL BUFFER FLUSH
    logger.info(f"[TRANSITION] Flushing audio buffers... switching to {selected_lang_code}")
    await asyncio.sleep(2.5)  # Increased slightly for safety

    # --- STAGE 2: GRIEVANCE AGENT ---
    logger.info("[START] Stage 2: Grievance Collection")

    # Get the specific prompt for the language
    prompt = GRIEVANCE_PROMPTS.get(selected_lang_code, GRIEVANCE_PROMPTS["en"])
    
    # Lookup TTS code
    tts_code = "en-IN"
    for lang_data in SUPPORTED_LANGUAGES.values():
        if lang_data["code"] == selected_lang_code:
            tts_code = lang_data["sarvam_tts_code"]
            break

    session_2 = AgentSession(
        vad=silero.VAD.load(),
        stt=groq.STT(model="whisper-large-v3-turbo", language=selected_lang_code),  # Groq Whisper
        llm=openai.LLM.with_ollama(
            model="qwen2.5:7b",
            base_url="http://localhost:11434/v1",
        ),  # Local Ollama LLM
        tts=sarvam.TTS(target_language_code=tts_code, speaker="anushka"),
    )

    grievance_agent = Agent(
        instructions=prompt,
        tools=[end_call],
    )
    
    # Event Hook for Session 2 (Logging)
    @session_2.on("conversation_item_added")
    def on_item_added_2(event):
        if event.item.text_content:
            if event.item.role == "user":
                grievance_tracker.add_user_message(event.item.text_content)
            elif event.item.role == "assistant":
                grievance_tracker.add_agent_message(event.item.text_content)

    task_2 = asyncio.create_task(session_2.start(agent=grievance_agent, room=ctx.room))
    
    # Trigger the greeting immediately
    await session_2.generate_reply()

    # Wait for the end
    await should_end_call.wait()
    
    # Cleanup Session 2
    logger.info("[SHUTDOWN] Ending grievance collection...")
    await asyncio.sleep(1.0)
    
    try:
        await session_2.aclose()
    except Exception as e:
        logger.error(f"[SHUTDOWN] Error closing session 2: {e}")
    
    task_2.cancel()
    try:
        await task_2
    except asyncio.CancelledError:
        pass
    
    # Save to DB
    full_log = grievance_tracker.get_full_grievance()
    await db_manager.save_grievance(full_log, selected_lang_code)
    
    await ctx.disconnect()
    logger.info("[END] Disconnected from room")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))