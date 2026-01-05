import sqlite3
import json
import os
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, Optional

# Load environment variables
load_dotenv()

# Configure Gemini API
# Make sure GOOGLE_API_KEY is in your .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GrievanceProcessor:
    def __init__(self, db_path="grievance.db"):
        self.db_path = db_path
        self.ensure_schema_updates()

    def ensure_schema_updates(self):
        """
        Checks if 'department' column exists. If not, adds it.
        This ensures backward compatibility with your existing DB.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get list of columns
        cursor.execute("PRAGMA table_info(grievances)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "department" not in columns:
            print("[SCHEMA] 'department' column missing. Adding it now...")
            try:
                cursor.execute("ALTER TABLE grievances ADD COLUMN department TEXT")
                conn.commit()
                print("[SCHEMA] Column 'department' added successfully.")
            except Exception as e:
                print(f"[SCHEMA] Error updating schema: {e}")
        
        conn.close()

    async def categorize_grievance(self, transcript: str) -> Optional[Dict]:
        """Use Gemini to categorize, analyze, extract location AND department."""
        
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""Analyze this customer grievance and provide a structured categorization.

Grievance transcript:
"{transcript}"

Please provide:
1. Category (e.g., POSH, Managerial, Data, Hygiene, Compensation, Workplace Environment, Conflict, Career, Attendance)
2. Priority (High, Medium, Low)
3. Sentiment (Positive, Neutral, Negative, Very Negative)
4. Brief Summary (1-2 sentences)
5. Tags (up to 5 relevant keywords)
6. Location (Extract specific branch, city, or office. If NOT found, return "Undisclosed Location")
7. Department (Extract specific department e.g., Sales, IT, HR, Logistics. If NOT found, return "General")

Respond strictly in VALID JSON format without markdown code blocks:
{{
    "category": "...",
    "priority": "...",
    "sentiment": "...",
    "summary": "...",
    "tags": ["tag1", "tag2"],
    "location": "...",
    "department": "..."
}}"""

        try:
            response = await model.generate_content_async(prompt)
            text_response = response.text
            
            # Clean up potential markdown formatting (```json ... ```)
            clean_json = text_response.replace("```json", "").replace("```", "").strip()
            
            return json.loads(clean_json)
        except Exception as e:
            print(f"[AI ERROR] Failed to process transcript: {e}")
            return None

    def get_pending_grievances(self):
        """Fetch all grievances with status 'pending'."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, transcript FROM grievances WHERE status = 'pending'")
        rows = cursor.fetchall()
        
        conn.close()
        return rows

    def update_grievance(self, g_id: str, data: Dict):
        """Update the database with the AI analysis results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Convert tags list to string for storage
            tags_str = ", ".join(data.get("tags", []))

            cursor.execute("""
                UPDATE grievances 
                SET category = ?,
                    priority = ?,
                    sentiment = ?,
                    summary = ?,
                    tags = ?,
                    location = ?,
                    department = ?,
                    status = 'processed'
                WHERE id = ?
            """, (
                data.get("category"),
                data.get("priority"),
                data.get("sentiment"),
                data.get("summary"),
                tags_str,
                data.get("location"),
                data.get("department"),
                g_id
            ))
            conn.commit()
            print(f"[DB] Updated grievance {g_id} successfully.")
        except Exception as e:
            print(f"[DB] Error updating grievance {g_id}: {e}")
        finally:
            conn.close()

async def main():
    processor = GrievanceProcessor()
    
    print("--- Starting Grievance Processor ---")
    pending_rows = processor.get_pending_grievances()
    
    if not pending_rows:
        print("No pending grievances found.")
        return

    print(f"Found {len(pending_rows)} pending grievances. Processing...")

    for row in pending_rows:
        g_id = row['id']
        transcript = row['transcript']

        print(f"Processing ID: {g_id}...")
        
        # 1. Analyze with Gemini
        analysis_result = await processor.categorize_grievance(transcript)
        
        if analysis_result:
            # 2. Update Database
            processor.update_grievance(g_id, analysis_result)
        else:
            print(f"Skipping update for {g_id} due to AI failure.")

    print("--- Processing Complete ---")

if __name__ == "__main__":
    asyncio.run(main())