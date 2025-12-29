import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, Optional
import google.generativeai as genai
import os

class GrievanceProcessor:
    """Handles LLM categorization and SQLite storage of grievances."""
    
    def __init__(self, db_path: str = "grievances.db", api_key: Optional[str] = None):
        self.db_path = db_path
        # Initialize Gemini
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database and ensure location column exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
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
                location TEXT
            )
        """)
        
        # Migration Check: Check if 'location' column exists (for existing databases)
        cursor.execute("PRAGMA table_info(grievances)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "location" not in columns:
            print("[DB] Migrating database: Adding 'location' column...")
            cursor.execute("ALTER TABLE grievances ADD COLUMN location TEXT")
        
        conn.commit()
        conn.close()
        print(f"[DB] Database initialized at {self.db_path}")
    
    async def categorize_grievance(self, transcript: str) -> Dict:
        """Use Gemini to categorize, analyze, and extract location."""
        prompt = f"""Analyze this customer grievance and provide a structured categorization.

Grievance transcript:
{transcript}

Please provide:
1. Category (e.g., POSH, Managerial, Data, Hygiene, Compensation, Workplace Environment, Conflict, Career, Attendance)
2. Priority (High, Medium, Low)
3. Sentiment (Positive, Neutral, Negative, Very Negative)
4. Brief Summary (1-2 sentences)
5. Tags (up to 5 relevant keywords)
6. Location (Extract the specific branch, city, or office location mentioned. If NO location is found, return "Undisclosed Location")

Respond strictly in JSON format:
{{
    "category": "...",
    "priority": "...",
    "sentiment": "...",
    "summary": "...",
    "tags": ["tag1", "tag2"],
    "location": "..."
}}"""

        try:
            # Generate content using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            
            analysis = json.loads(response.text)
            
            # Fallback if LLM forgets the key
            if "location" not in analysis:
                analysis["location"] = "Undisclosed Location"
                
            print(f"[LLM] Categorized: {analysis['category']} | Loc: {analysis['location']}")
            return analysis
            
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            # Return default analysis if LLM fails
            return {
                "category": "Uncategorized",
                "priority": "Medium",
                "sentiment": "Neutral",
                "summary": transcript[:200] + "..." if len(transcript) > 200 else transcript,
                "tags": [],
                "location": "Undisclosed Location"
            }
    
    def store_grievance(
        self, 
        transcript: str, 
        timestamp: float,
        analysis: Dict
    ) -> str:
        """Store grievance in SQLite database with unique ID."""
        grievance_id = str(uuid.uuid4())
        created_at = datetime.fromtimestamp(timestamp).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO grievances 
            (id, timestamp, transcript, category, priority, sentiment, summary, tags, created_at, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            grievance_id,
            str(timestamp),
            transcript,
            analysis.get("category", "Uncategorized"),
            analysis.get("priority", "Medium"),
            analysis.get("sentiment", "Neutral"),
            analysis.get("summary", ""),
            json.dumps(analysis.get("tags", [])),
            created_at,
            analysis.get("location", "Undisclosed Location")
        ))
        
        conn.commit()
        conn.close()
        
        print(f"[DB] Stored grievance {grievance_id} (Loc: {analysis.get('location')})")
        return grievance_id
    
    async def process_and_store(self, transcript: str, timestamp: float) -> Dict:
        """Complete pipeline: categorize with LLM and store in database."""
        print(f"\n[PROCESSING] Analyzing grievance...")
        
        # Step 1: Categorize with LLM
        analysis = await self.categorize_grievance(transcript)
        
        # Step 2: Store in database
        grievance_id = self.store_grievance(transcript, timestamp, analysis)
        
        return {
            "id": grievance_id,
            "transcript": transcript,
            "timestamp": timestamp,
            **analysis
        }
    
    def get_grievance(self, grievance_id: str) -> Optional[Dict]:
        """Retrieve a grievance by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM grievances WHERE id = ?", (grievance_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # Handle potential missing column if row is from very old schema version 
            # (though _init_database handles the migration, this is extra safety)
            location = row[9] if len(row) > 9 else "Undisclosed Location"
            
            return {
                "id": row[0],
                "timestamp": float(row[1]),
                "transcript": row[2],
                "category": row[3],
                "priority": row[4],
                "sentiment": row[5],
                "summary": row[6],
                "tags": json.loads(row[7]),
                "created_at": row[8],
                "location": location
            }
        return None
    
    def get_all_grievances(self, limit: int = 100) -> list:
        """Retrieve all grievances, most recent first."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM grievances 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            location = row[9] if len(row) > 9 else "Undisclosed Location"
            results.append({
                "id": row[0],
                "timestamp": float(row[1]),
                "transcript": row[2],
                "category": row[3],
                "priority": row[4],
                "sentiment": row[5],
                "summary": row[6],
                "tags": json.loads(row[7]),
                "created_at": row[8],
                "location": location
            })
        return results
    
    def get_statistics(self) -> Dict:
        """Get summary statistics of all grievances."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM grievances")
        total = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM grievances 
            GROUP BY category 
            ORDER BY count DESC
        """)
        by_category = dict(cursor.fetchall())
        
        cursor.execute("""
            SELECT priority, COUNT(*) as count 
            FROM grievances 
            GROUP BY priority
        """)
        by_priority = dict(cursor.fetchall())
        
        # New: Stats by location
        # Check if column exists first to be safe during migration edge cases
        try:
            cursor.execute("""
                SELECT location, COUNT(*) as count 
                FROM grievances 
                GROUP BY location
                ORDER BY count DESC
            """)
            by_location = dict(cursor.fetchall())
        except sqlite3.OperationalError:
            by_location = {}

        conn.close()
        
        return {
            "total_grievances": total,
            "by_category": by_category,
            "by_priority": by_priority,
            "by_location": by_location
        }