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

load_dotenv()

# --- Database Manager ---
class DatabaseManager:
    def __init__(self, db_path="grievance.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grievances (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                transcript TEXT NOT NULL,
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

    def save_grievance(self, transcript: str):
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
                INSERT INTO grievances (id, timestamp, transcript, created_at)
                VALUES (?, ?, ?, ?)
            """, (record_id, current_time, transcript, current_time))
            
            conn.commit()
            print(f"[DB] Successfully saved grievance ID: {record_id}")
        except Exception as e:
            print(f"[DB] Error saving grievance: {e}")
        finally:
            conn.close()

# --- Tamil System Prompt ---
SYSTEM_INSTRUCTIONS = """நீங்கள் "HR Voice Assistant", ஒரு தொழில்முறை மற்றும் அனுதாபமுள்ள குறைதீர்ப்பு சேகரிப்பாளர். ஆலோசனை அல்லது தீர்வுகளை வழங்காமல் குறைகளை திறமையாக பதிவு செய்வதே உங்கள் இலக்கு.

முக்கிய நடத்தைகள்:
- **பாணி:** அன்பானது ஆனால் சுருக்கமாக (அதிகபட்சம் 2 வாக்கியங்கள்). இயல்பான, குறைந்தபட்ச ஒப்புதல்களைப் பயன்படுத்துங்கள் ("புரிகிறது," "தொடர்ந்து சொல்லுங்கள்").
- **தேவையான தகவல்:** நீங்கள் சேகரிக்க வேண்டும்: 1) குறைதீர்ப்பு, 2) இடம், மற்றும் 3) துறை.
- **தவறிய தகவல்:** பயனர் இடம் அல்லது துறையைத் தவிர்த்தால், அவர்களிடம் இயல்பாக *ஒரு முறை* கேளுங்கள்.

ஓட்டம்:
1. **வாழ்த்து:** சுருக்கமான வரவேற்பு + அவர்களை பேச அழைக்கவும்.
2. **கேளுங்கள்:** இடையூறு இல்லாமல் ஒப்புக்கொள்ளுங்கள்.
3. **முடிவு:** பயனர் அவர்கள் முடித்துவிட்டதாகக் குறிப்பிடும்போது (எ.கா. "அவ்வளவுதான்," "நான் முடித்துவிட்டேன்," அல்லது "நன்றி"):
   - பதிலளிக்கவும்: "இதைப் பகிர்ந்ததற்கு நன்றி. உங்கள் குறைதீர்ப்பு பதிவு செய்யப்பட்டு மதிப்பாய்வு செய்யப்படும். கவனமாக இருங்கள்."
   - **உடனடியாக** `end_call` செயல்பாட்டை அழைக்கவும்.

கட்டுப்பாடுகள்:
- சிக்கல்களைத் தீர்க்காதீர்கள் அல்லது ஆலோசனை வழங்காதீர்கள்.
- "வேறு ஏதாவது உள்ளதா?" என்று மீண்டும் கேட்காதீர்கள்.
- பயனர் முழுமையைக் குறிக்கும்போது மட்டுமே `end_call` ஐ அழைக்கவும்.

முடிவு வரிசை (100% முழுமையானதும் மட்டும்):
1. கூறவும்: "இதைப் பகிர்ந்ததற்கு நன்றி. உங்கள் குறைதீர்ப்பு பதிவு செய்யப்பட்டு எங்கள் குழுவால் மதிப்பாய்வு செய்யப்படும். கவனமாக இருங்கள்."
2. "grievance_complete" காரணத்துடன் end_call ஐ அழைக்கவும்

நினைவில் கொள்ளுங்கள்: உங்கள் வேலை முழுமையான தகவலைச் சேகரிப்பது. பொறுமையாகவும் முழுமையாகவும் இருங்கள்."""


class GrievanceTracker:
    """Track grievance conversation and manage call state."""
    
    def __init__(self):
        self.grievance_text = []  # Tamil transcript
        self.conversation_history = []
        self.word_count = 0
    
    def add_user_message(self, text: str):
        """Add user message in Tamil."""
        self.grievance_text.append(f"பணியாளர்: {text}")
        self.conversation_history.append({"role": "user", "content": text})
        self.word_count += len(text.split())
        print(f"[USER MESSAGE] {text[:100]}...")
    
    def add_agent_message(self, text: str):
        """Add agent message to history."""
        self.conversation_history.append({"role": "assistant", "content": text})
    
    def get_full_grievance(self) -> str:
        """Get the complete grievance transcript in Tamil."""
        return "\n".join(self.grievance_text)
    
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
    
    # Initialize Database
    db_manager = DatabaseManager()

    # Connect to the room
    await ctx.connect()
    
    # Initialize the grievance tracker
    grievance_tracker = GrievanceTracker()
    
    # Flag to track if call should end
    should_end_call = asyncio.Event()
    
    # Define the end_call function tool
    @function_tool
    async def end_call(
        confirmation: str = "yes"
    ):
        """
        பயனர் முடித்துவிட்டதாகக் குறிப்பிடும்போது அல்லது உரையாடல் முடிந்தவுடன் குறைதீர்ப்பு சேகரிப்பு அழைப்பை முடிக்கவும்.
        
        Args:
            confirmation: அழைப்பை முடிக்க உறுதிப்படுத்தல் (இயல்புநிலை: "yes")
        """
        print("[FUNCTION] end_call function invoked by LLM")
        should_end_call.set()
        return "அழைப்பு முடிவடைதல் தொடங்கப்பட்டது. குறைதீர்ப்பு பதிவு செய்யப்பட்டுள்ளது."
    
    # Create the agent with Tamil instructions
    agent = Agent(
        instructions=SYSTEM_INSTRUCTIONS,
        tools=[end_call],
    )
    
    # Create agent session with Tamil language support
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=0.8,
        ),
        stt=groq.STT(
            model="whisper-large-v3-turbo",
            language="ta",
        ),
        llm=groq.LLM(
            model="openai/gpt-oss-20b",
            temperature=0.7,
        ),
        tts=sarvam.TTS(
            target_language_code="ta-IN",
            speaker="anushka",
        )
    )
    
    # Event handlers for tracking conversation
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        """Called when a conversation item is added (user or agent)."""
        item = event.item
        
        if item.role == "user":
            # User message in Tamil
            text = item.text_content or ""
            if text:
                print(f"\n[USER - தமிழ்] {text}")
                grievance_tracker.add_user_message(text)
        elif item.role == "assistant":
            # Agent message
            text = item.text_content or ""
            if text and not item.interrupted:
                print(f"[AGENT - தமிழ்] {text}")
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
    
    # Send initial greeting in Tamil
    try:
        await session.generate_reply(
            instructions="சுருக்கமான, அன்பான வரவேற்பு கொடுத்து அவர்களின் குறைதீர்ப்பைப் பகிர்ந்து கொள்ளச் சொல்லுங்கள். ஒரே ஒரு வாக்கியம் மட்டும்."
        )
    except Exception as e:
        print(f"[ERROR] Failed to generate greeting: {e}")
    
    print("[AGENT] Ready to collect grievances in Tamil...")
    
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
        # Print final grievance
        stats = grievance_tracker.get_stats()
        full_grievance = grievance_tracker.get_full_grievance()
        
        print("\n" + "="*70)
        print("[GRIEVANCE COLLECTION COMPLETE]")
        print("="*70)
        print(f"Total Messages: {stats['total_messages']}")
        print(f"User Messages: {stats['user_messages']}")
        print(f"Word Count: {stats['word_count']}")
        print("-"*70)
        print("[TAMIL TRANSCRIPT]")
        print(full_grievance)
        print("="*70 + "\n")
        
        # Save Tamil transcript to database
        print("[DB] Saving Tamil transcript to database...")
        db_manager.save_grievance(full_grievance)
        
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