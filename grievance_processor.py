import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, Optional
import google.generativeai as genai  # Changed from anthropic
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
        """Initialize SQLite database with grievances table."""
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
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"[DB] Database initialized at {self.db_path}")
    
    async def categorize_grievance(self, transcript: str) -> Dict:
        """Use Gemini to categorize and analyze the grievance."""
        prompt = f"""Analyze this customer grievance and provide a structured categorization.

Grievance transcript:
{transcript}

Please provide:
1. Category (e.g., POSH, Managerial, Data, Hygiene, Compensation, Workplace Environment, Conflict, Career, Attendance)
2. Priority (High, Medium, Low)
3. Sentiment (Positive, Neutral, Negative, Very Negative)
4. Brief Summary (1-2 sentences)
5. Tags (up to 5 relevant keywords)

Respond strictly in JSON format:
{{
    "category": "...",
    "priority": "...",
    "sentiment": "...",
    "summary": "...",
    "tags": ["tag1", "tag2", "tag3"]
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
            print(f"[LLM] Categorized as: {analysis['category']} | Priority: {analysis['priority']}")
            return analysis
            
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            # Return default analysis if LLM fails
            return {
                "category": "Uncategorized",
                "priority": "Medium",
                "sentiment": "Neutral",
                "summary": transcript[:200] + "..." if len(transcript) > 200 else transcript,
                "tags": []
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
            (id, timestamp, transcript, category, priority, sentiment, summary, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            grievance_id,
            str(timestamp),
            transcript,
            analysis.get("category", "Uncategorized"),
            analysis.get("priority", "Medium"),
            analysis.get("sentiment", "Neutral"),
            analysis.get("summary", ""),
            json.dumps(analysis.get("tags", [])),
            created_at
        ))
        
        conn.commit()
        conn.close()
        
        print(f"[DB] Stored grievance with ID: {grievance_id}")
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
            return {
                "id": row[0],
                "timestamp": float(row[1]),
                "transcript": row[2],
                "category": row[3],
                "priority": row[4],
                "sentiment": row[5],
                "summary": row[6],
                "tags": json.loads(row[7]),
                "created_at": row[8]
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
        
        return [{
            "id": row[0],
            "timestamp": float(row[1]),
            "transcript": row[2],
            "category": row[3],
            "priority": row[4],
            "sentiment": row[5],
            "summary": row[6],
            "tags": json.loads(row[7]),
            "created_at": row[8]
        } for row in rows]
    
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
        
        conn.close()
        
        return {
            "total_grievances": total,
            "by_category": by_category,
            "by_priority": by_priority
        }