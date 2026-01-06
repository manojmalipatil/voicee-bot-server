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
from livekit.plugins import deepgram, cartesia, groq, silero

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
        
        # Create table with specific columns
        # Note: status defaults to 'pending', others are nullable for post-processing
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
            # We only insert id, timestamp, transcript, and created_at
            # 'status' will auto-fill to 'pending'
            # All other columns remain NULL for post-processing
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

# --- Existing System Prompt ---
SYSTEM_INSTRUCTIONS = """You are "HR Voice Assistant," a professional, empathetic grievance collector. Your goal is to record grievances efficiently without offering advice or solutions.

CORE BEHAVIORS:
- **Style:** Warm but concise (max 2 sentences). Use natural, minimal acknowledgments ("I see," "Please go on").
- **Required Data:** You must collect: 1) The Grievance, 2) Location, and 3) Department.
- **Missing Info:** If the user omits Location or Department, ask for them naturally *once* (e.g., "To complete the record, could you share your department and where this happened?").

FLOW:
1. **Greet:** Brief welcome + invite them to speak.
2. **Listen:** Acknowledge without interrupting.
3. **Closing:** When the user indicates they are finished (e.g., "That's all," "I'm done," or "Thank you"):
   - Reply: "Thank you for sharing this. Your grievance has been recorded and will be reviewed. Take care."
   - **IMMEDIATELY** call the `end_call` function.

CONSTRAINTS:
- DO NOT solve problems or give advice.
- DO NOT loop "Is there anything else?"
- ONLY call `end_call` when the user signals completion.

CLOSING SEQUENCE (only when 100% complete):
1. Say: "Thank you for sharing this. Your grievance has been recorded and will be reviewed by our team. Take care."
2. Call end_call with reason "grievance_complete"

Remember: Your job is to collect COMPLETE information. Be patient and thorough."""


class GrievanceTracker:
    """Track grievance conversation and manage call state."""
    
    def __init__(self):
        self.grievance_text = []
        self.conversation_history = []
        self.word_count = 0
    
    def add_user_message(self, text: str):
        """Add user message."""
        self.grievance_text.append(f"Employee: {text}")
        self.conversation_history.append({"role": "user", "content": text})
        self.word_count += len(text.split())
    
    def add_agent_message(self, text: str):
        """Add agent message to history."""
        self.conversation_history.append({"role": "assistant", "content": text})
    
    def get_full_grievance(self) -> str:
        """Get the complete grievance transcript."""
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
        End the grievance collection call when the user indicates they are done 
        or the conversation is complete. Call this when you hear phrases like 
        "that's all", "I'm done", "nothing else", or when the user clearly 
        indicates they have finished sharing their grievance.
        
        Args:
            confirmation: Confirmation to end the call (default: "yes")
        """
        print("[FUNCTION] end_call function invoked by LLM")
        should_end_call.set()
        return "Call ending initiated. The grievance has been recorded."
    
    # Create the agent with enhanced instructions and function calling
    agent = Agent(
        instructions=SYSTEM_INSTRUCTIONS,
        tools=[end_call],  # Pass the function tool to the agent
    )
    
    # Create agent session with optimized settings
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=0.8,
        ),
        stt=deepgram.STT(
            model="nova-2",
            language="en",
            smart_format=True,
            punctuate=True,
        ),
        llm=groq.LLM(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
        ),
        tts=cartesia.TTS(
            voice="248be419-c632-4f23-adf1-5324ed7dbf1d",
        ),
    )
    
    # Event handlers for tracking conversation
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        """Called when a conversation item is added (user or agent)."""
        item = event.item
        
        if item.role == "user":
            # User message
            text = item.text_content or ""
            if text:
                print(f"\n[USER] {text}")
                grievance_tracker.add_user_message(text)
        elif item.role == "assistant":
            # Agent message - only print complete, non-interrupted messages
            text = item.text_content or ""
            if text and not item.interrupted:
                print(f"[AGENT] {text}")
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
    
    # Wait for session to be ready - increased wait time
    await asyncio.sleep(1.5)
    
    print("[SESSION] Sending initial greeting...")
    
    # Send initial greeting
    try:
        await session.generate_reply(
            instructions="Give a brief, warm greeting and ask them to share their grievance. ONE sentence only."
        )
    except Exception as e:
        print(f"[ERROR] Failed to generate greeting: {e}")
    
    print("[AGENT] Ready to collect grievances...")
    
    try:
        # Wait only for the end_call signal
        await should_end_call.wait()
        
        print("[CLOSING] end_call triggered. Waiting for final message...")
        
        # Give enough time for the final message to be generated and spoken
        # Most TTS responses complete within 3-4 seconds
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
        print(full_grievance)
        print("="*70 + "\n")
        
        # --- DATABASE STORAGE LOGIC ---
        print("[DB] Saving grievance to local database...")
        db_manager.save_grievance(full_grievance)
        # ------------------------------
        
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