import asyncio
import os
from dotenv import load_dotenv

# LiveKit Imports
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, cartesia, groq, silero

load_dotenv()

# Enhanced system prompt for grievance collection
SYSTEM_INSTRUCTIONS = """You are a professional and empathetic employee grievance collection agent named "HR Voice Assistant". Your primary goal is to collect employee grievances efficiently and compassionately.

CONVERSATION FLOW:
1. Start with a warm, brief greeting (1 sentence)
2. Invite them to share their grievance
3. Listen actively - let them speak without interruption
4. Use minimal acknowledgments: "I understand", "I see", "Please go on"
5. Only ask clarifying questions if critical details are missing
6. When they finish, thank them and confirm the grievance is recorded

RESPONSE STYLE:
- Keep responses SHORT (1-2 sentences maximum)
- Be warm but professional
- Never give advice or try to solve the problem
- Don't ask unnecessary questions
- Use natural conversational language
- Show empathy through tone, not lengthy responses

DETECTING CONVERSATION END:
When the user says phrases like:
- "That's all" / "That's it"
- "Nothing else" / "Nothing more"
- "I'm done" / "That's everything"
- "Thank you" (as a closing)
- Any clear indication they're finished

Respond with: "Thank you for sharing this with me. Your grievance has been recorded and will be reviewed by our team. Take care."

Then STOP generating further responses - the call will end automatically.

IMPORTANT RULES:
- NO lengthy explanations or procedures
- NO asking "Is there anything else?" repeatedly
- NO offering solutions or advice
- Keep it natural, brief, and human-like
- Let silence be okay - don't fill every gap"""


class GrievanceTracker:
    """Track grievance conversation and manage call state."""
    
    def __init__(self):
        self.grievance_text = []
        self.conversation_history = []
        self.should_end_call = False
        self.word_count = 0
        
        # Closing phrases that indicate conversation end
        self.closing_phrases = [
            "that's all", "that's it", "thats all", "thats it",
            "nothing else", "nothing more",
            "i'm done", "im done", "i am done",
            "that is all", "that is it",
            "thank you", "thanks",
            "goodbye", "bye", "good bye",
        ]
    
    def add_user_message(self, text: str):
        """Add user message and check for closing signals."""
        self.grievance_text.append(f"Employee: {text}")
        self.conversation_history.append({"role": "user", "content": text})
        self.word_count += len(text.split())
        
        # Check if user is indicating they're done
        text_lower = text.lower().strip()
        
        # Check for closing phrases
        for phrase in self.closing_phrases:
            if phrase in text_lower:
                # Check if it's at the end or standalone
                if text_lower.endswith(phrase) or text_lower == phrase:
                    print(f"[DETECTOR] Closing phrase detected: '{phrase}'")
                    self.should_end_call = True
                    return True
                
                # Check if phrase is followed by minimal words
                idx = text_lower.find(phrase)
                if idx != -1:
                    remainder = text_lower[idx + len(phrase):].strip()
                    if len(remainder.split()) <= 2:
                        print(f"[DETECTOR] Closing phrase with minimal tail: '{phrase}'")
                        self.should_end_call = True
                        return True
        
        return False
    
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
    
    # Connect to the room
    await ctx.connect()
    
    # Initialize the grievance tracker
    grievance_tracker = GrievanceTracker()
    
    # Create the agent with enhanced instructions
    agent = Agent(
        instructions=SYSTEM_INSTRUCTIONS,
    )
    
    # Create agent session with optimized settings
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.3,  # Detect speech after 300ms
            min_silence_duration=0.8,  # Consider silence after 800ms
        ),
        stt=deepgram.STT(
            model="nova-2",
            language="en",
            smart_format=True,
            punctuate=True,
        ),
        llm=groq.LLM(
            model="llama-3.3-70b-versatile",
            temperature=0.7,  # Balanced between consistent and natural
        ),
        tts=cartesia.TTS(
            voice="248be419-c632-4f23-adf1-5324ed7dbf1d",  # Friendly female voice
            speed="normal",
            emotion=["positivity:high", "curiosity:medium"],
        ),
    )
    
    # Flag to track if we're closing
    is_closing = False
    
    # Event handlers for tracking conversation
    @session.on("user_speech_committed")
    def on_user_speech_committed(message):
        """Called when user finishes speaking."""
        nonlocal is_closing
        
        print(f"\n[USER] {message.content}")
        
        # Add message and check if it's a closing signal
        is_closing_signal = grievance_tracker.add_user_message(message.content)
        
        if is_closing_signal:
            print("[SIGNAL] User indicated they're done talking")
            is_closing = True
    
    @session.on("agent_speech_committed")
    def on_agent_speech_committed(message):
        """Called when agent finishes speaking."""
        print(f"[AGENT] {message.content}")
        grievance_tracker.add_agent_message(message.content)
        
        # If this was the closing message, prepare to disconnect
        if is_closing:
            print("[CLOSING] Agent finished farewell. Scheduling disconnect...")
            asyncio.create_task(disconnect_gracefully())
    
    @session.on("agent_speech_interrupted")
    def on_agent_speech_interrupted(message):
        """Called when user interrupts the agent."""
        print(f"[INTERRUPTED] User interrupted agent")
    
    async def disconnect_gracefully():
        """Disconnect from the room after a brief delay."""
        await asyncio.sleep(2)  # Wait 2 seconds after closing message
        
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
        
        # TODO: Add storage logic here
        # await store_grievance(full_grievance)
        
        print("[DISCONNECT] Closing connection...")
        await ctx.room.disconnect()
    
    # Start the session
    await session.start(agent=agent, room=ctx.room)
    
    print("[SESSION] Agent session started. Waiting for participant...")
    
    # Wait for participant to join before greeting
    async def wait_for_participant():
        """Wait for a participant to join before starting."""
        while len(ctx.room.remote_participants) == 0:
            await asyncio.sleep(0.1)
        
        print("[SESSION] Participant joined. Sending greeting...")
        
        # Send initial greeting after participant joins
        await session.generate_reply(
            instructions="Give a brief, warm greeting and ask them to share their grievance. ONE sentence only."
        )
    
    asyncio.create_task(wait_for_participant())
    
    print("[AGENT] Ready to collect grievances...")
    
    # Monitor for manual disconnection
    try:
        # Keep running until disconnect is called
        while ctx.room.connection_state == "connected":
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    finally:
        print("[SESSION] Session ended")


if __name__ == "__main__":
    # Run the agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )