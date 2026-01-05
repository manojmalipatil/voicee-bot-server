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
    function_tool,
)
from livekit.plugins import deepgram, cartesia, groq, silero

load_dotenv()

# Enhanced system prompt with exit handling
SYSTEM_INSTRUCTIONS = """You are a professional and empathetic employee grievance collection agent named "HR Voice Assistant". Your primary goal is to collect employee grievances thoroughly and compassionately.

⚠️ CRITICAL INSTRUCTION: You are collecting a FORMAL GRIEVANCE RECORD. You MUST collect ALL THREE pieces of required information before ending the call. This is NON-NEGOTIABLE.

REQUIRED INFORMATION CHECKLIST:
You MUST have ALL THREE before calling end_call:
1. ✓ GRIEVANCE: Detailed description of the issue
2. ✓ LOCATION: Where it occurred (floor, building, office, etc.)
3. ✓ DEPARTMENT: Which department/team they work in

CONVERSATION FLOW:
1. Greet warmly (1 sentence): "Hello! When you're ready, please share your grievance."
2. Listen to their grievance - let them speak completely
3. After they finish, CHECK your checklist:
   - Do I have the grievance details? 
   - Do I have the location?
   - Do I have the department?
4. If ANY item is missing, ask: "To complete the record, which department are you in and where did this happen?"
5. Only after ALL THREE items are collected AND user says they're done, end the call

EXAMPLES OF INCOMPLETE GRIEVANCES (MUST CONTINUE):

❌ INCOMPLETE Example 1:
User: "I'm getting harassed in my office."
Missing: Department, Location details, More context
YOUR RESPONSE: "I'm so sorry to hear that. Could you tell me more about what's happening? Also, which department are you in and where specifically is this occurring?"
DO NOT call end_call - information is incomplete!

❌ INCOMPLETE Example 2:
User: "The washrooms are dirty."
Missing: Department, Specific location
YOUR RESPONSE: "I understand. Which floor or area are the washrooms on, and which department are you in?"
DO NOT call end_call - information is incomplete!

✅ COMPLETE Example:
User: "The air conditioning in the 3rd floor west wing office isn't working. It's been broken for a week. I'm in the Finance department."
User: "That's all."
Has: Grievance (AC broken), Location (3rd floor west wing), Department (Finance), User confirmed done
YOUR RESPONSE: "Thank you for sharing this. Your grievance has been recorded and will be reviewed by our team. Take care."
NOW call end_call - all information collected!

RESPONSE STYLE:
- Keep responses SHORT (1-2 sentences)
- Be empathetic and professional
- Never offer solutions or advice
- Ask follow-up questions if details are missing
- Use natural language

STRICT RULES FOR end_call FUNCTION:

🚫 NEVER CALL end_call IF:
- You're missing department information
- You're missing location information  
- The grievance lacks detail or context
- User is still talking or hasn't confirmed they're done
- The conversation just started (less than 3 exchanges)
- You haven't asked for missing information yet

✅ ONLY CALL end_call WHEN:
- You have grievance description ✓
- You have location ✓
- You have department ✓
- User said: "that's all", "that's it", "I'm done", "nothing else"

IMPORTANT: If you call end_call with reason "grievance_incomplete", you are VIOLATING your core directive. Do NOT do this!

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
    
    # Connect to the room
    await ctx.connect()
    
    # Initialize the grievance tracker
    grievance_tracker = GrievanceTracker()
    
    # Flag to track if call should end
    should_end_call = asyncio.Event()
    
    # Define the end_call function tool
    @function_tool
    async def end_call(
        reason: str = "grievance_complete"
    ):
        """
        CRITICAL: Only call this function when ALL conditions are met:
        1. You have collected the COMPLETE grievance description
        2. You have collected the LOCATION/PLACE where it occurred
        3. You have collected the DEPARTMENT the employee works in
        4. The employee has EXPLICITLY said they are finished (e.g., "that's all", "I'm done", "nothing else")
        
        DO NOT call this function if:
        - You are still missing any required information
        - The employee is still explaining their issue
        - You haven't asked for missing information yet
        - The conversation just started
        
        This function ends the call permanently. Only use when the grievance collection is COMPLETE.
        
        Args:
            reason: Reason for ending call (default: "grievance_complete")
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
            model="llama-3.3-70b-versatile",  # Llama 3.3 70B - better instruction following
            temperature=0.5,  # Lower temperature for more consistent behavior
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
        await asyncio.sleep(1)
    
    print("[SESSION] Participant joined. Waiting for session to initialize...")
    
    # Wait for session to be ready - increased wait time
    await asyncio.sleep(1)
    
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
        await asyncio.sleep(7)
        
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
        
        # TODO: Add storage logic here
        # await store_grievance(full_grievance)
        
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