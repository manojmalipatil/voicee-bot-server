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

# Translation Import
from deep_translator import GoogleTranslator

load_dotenv()

# --- Translation Helper ---
class TranslationManager:
    def __init__(self):
        self.translator_to_english = GoogleTranslator(source='hi', target='en')
        self.translator_to_hindi = GoogleTranslator(source='en', target='hi')
    
    def hindi_to_english(self, text: str) -> str:
        """Translate Hindi text to English."""
        try:
            if not text.strip():
                return text
            translated = self.translator_to_english.translate(text)
            return translated
        except Exception as e:
            print(f"[TRANSLATION ERROR] Hindi to English: {e}")
            return text
    
    def english_to_hindi(self, text: str) -> str:
        """Translate English text to Hindi."""
        try:
            if not text.strip():
                return text
            translated = self.translator_to_hindi.translate(text)
            return translated
        except Exception as e:
            print(f"[TRANSLATION ERROR] English to Hindi: {e}")
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
                original_language TEXT DEFAULT 'hindi',
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

    def save_grievance(self, transcript: str, language: str = "hindi"):
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

# --- Hindi System Prompt ---
SYSTEM_INSTRUCTIONS = """आप "HR Voice Assistant" हैं, एक पेशेवर और सहानुभूतिपूर्ण शिकायत संग्रहकर्ता। आपका लक्ष्य सलाह या समाधान दिए बिना शिकायतों को कुशलतापूर्वक रिकॉर्ड करना है।

मुख्य व्यवहार:
- **शैली:** गर्मजोशी भरा लेकिन संक्षिप्त (अधिकतम 2 वाक्य)। प्राकृतिक, न्यूनतम स्वीकृति का उपयोग करें ("मैं समझता हूं," "कृपया जारी रखें")।
- **आवश्यक डेटा:** आपको यह एकत्र करना होगा: 1) शिकायत, 2) स्थान, और 3) विभाग।
- **छूटी हुई जानकारी:** यदि उपयोगकर्ता स्थान या विभाग छोड़ता है, तो उनसे स्वाभाविक रूप से *एक बार* पूछें।

प्रवाह:
1. **अभिवादन:** संक्षिप्त स्वागत + उन्हें बोलने के लिए आमंत्रित करें।
2. **सुनें:** बिना रुकावट के स्वीकार करें।
3. **समापन:** जब उपयोगकर्ता संकेत करता है कि वे समाप्त कर चुके हैं (जैसे "बस इतना ही," "मैं समाप्त कर चुका हूं," या "धन्यवाद"):
   - उत्तर दें: "आपकी शिकायत साझा करने के लिए धन्यवाद। यह रिकॉर्ड कर ली गई है और इसकी समीक्षा की जाएगी। ध्यान रखें।"
   - **तुरंत** `end_call` फ़ंक्शन को कॉल करें।

बाधाएं:
- समस्याओं को हल न करें या सलाह न दें।
- "कुछ और है?" को दोहराएं नहीं।
- केवल तभी `end_call` को कॉल करें जब उपयोगकर्ता पूर्णता का संकेत देता है।

समापन अनुक्रम (केवल जब 100% पूर्ण हो):
1. कहें: "आपकी शिकायत साझा करने के लिए धन्यवाद। यह रिकॉर्ड कर ली गई है और हमारी टीम द्वारा इसकी समीक्षा की जाएगी। ध्यान रखें।"
2. "grievance_complete" कारण के साथ end_call को कॉल करें

याद रखें: आपका काम पूर्ण जानकारी एकत्र करना है। धैर्यवान और संपूर्ण रहें।"""


class GrievanceTracker:
    """Track grievance conversation and manage call state."""
    
    def __init__(self, translator: TranslationManager):
        self.translator = translator
        self.grievance_text_hindi = []  # Original Hindi
        self.grievance_text_english = []  # Translated English
        self.conversation_history = []
        self.word_count = 0
    
    def add_user_message(self, text: str):
        """Add user message in Hindi and translate to English."""
        # Store original Hindi
        self.grievance_text_hindi.append(f"कर्मचारी: {text}")
        
        # Translate to English and store
        english_text = self.translator.hindi_to_english(text)
        self.grievance_text_english.append(f"Employee: {english_text}")
        
        self.conversation_history.append({"role": "user", "content": text})
        self.word_count += len(text.split())
        
        print(f"[TRANSLATION] Hindi: {text[:50]}...")
        print(f"[TRANSLATION] English: {english_text[:50]}...")
    
    def add_agent_message(self, text: str):
        """Add agent message to history."""
        self.conversation_history.append({"role": "assistant", "content": text})
    
    def get_full_grievance_hindi(self) -> str:
        """Get the complete grievance transcript in Hindi."""
        return "\n".join(self.grievance_text_hindi)
    
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
        जब उपयोगकर्ता यह संकेत करता है कि वे समाप्त कर चुके हैं या बातचीत पूर्ण हो गई है तो शिकायत संग्रह कॉल समाप्त करें।
        
        Args:
            confirmation: कॉल समाप्त करने की पुष्टि (डिफ़ॉल्ट: "yes")
        """
        print("[FUNCTION] end_call function invoked by LLM")
        should_end_call.set()
        return "कॉल समाप्त करना शुरू किया गया। शिकायत रिकॉर्ड कर ली गई है।"
    
    # Create the agent with Hindi instructions
    agent = Agent(
        instructions=SYSTEM_INSTRUCTIONS,
        tools=[end_call],
    )
    
    # Create agent session with Hindi language support
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=0.8,
        ),
        stt=deepgram.STT(
            model="nova-2",
            language="hi",  # Changed to Hindi
            smart_format=True,
            punctuate=True,
        ),
        llm=groq.LLM(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
        ),
        tts=cartesia.TTS(
            voice="faf0731e-dfb9-4cfc-8119-259a79b27e12",  # Use Hindi voice if available
            language="hi",  # Set TTS to Hindi
        ),
    )
    
    # Event handlers for tracking conversation
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        """Called when a conversation item is added (user or agent)."""
        item = event.item
        
        if item.role == "user":
            # User message in Hindi
            text = item.text_content or ""
            if text:
                print(f"\n[USER - हिंदी] {text}")
                grievance_tracker.add_user_message(text)
        elif item.role == "assistant":
            # Agent message - only print complete, non-interrupted messages
            text = item.text_content or ""
            if text and not item.interrupted:
                print(f"[AGENT - हिंदी] {text}")
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
    
    # Send initial greeting in Hindi
    try:
        await session.generate_reply(
            instructions="एक संक्षिप्त, गर्मजोशी भरा स्वागत दें और उन्हें अपनी शिकायत साझा करने के लिए कहें। केवल एक वाक्य।"
        )
    except Exception as e:
        print(f"[ERROR] Failed to generate greeting: {e}")
    
    print("[AGENT] Ready to collect grievances in Hindi...")
    
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
        full_grievance_hindi = grievance_tracker.get_full_grievance_hindi()
        full_grievance_english = grievance_tracker.get_full_grievance_english()
        
        print("\n" + "="*70)
        print("[GRIEVANCE COLLECTION COMPLETE]")
        print("="*70)
        print(f"Total Messages: {stats['total_messages']}")
        print(f"User Messages: {stats['user_messages']}")
        print(f"Word Count: {stats['word_count']}")
        print("-"*70)
        print("[ORIGINAL - HINDI]")
        print(full_grievance_hindi)
        print("-"*70)
        print("[TRANSLATED - ENGLISH]")
        print(full_grievance_english)
        print("="*70 + "\n")
        
        # --- DATABASE STORAGE LOGIC (English translation) ---
        print("[DB] Saving English translation to database...")
        db_manager.save_grievance(full_grievance_english, language="hindi")
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