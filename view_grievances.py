#!/usr/bin/env python3
"""
Utility script to view and manage stored grievances.
Usage: python view_grievances.py [command]
"""

import sys
import json
from grievance_processor import GrievanceProcessor
from datetime import datetime

def print_grievance(grievance: dict, detailed: bool = True):
    """Pretty print a grievance."""
    print(f"\n{'='*80}")
    print(f"ID: {grievance['id']}")
    print(f"Created: {grievance['created_at']}")
    print(f"Category: {grievance['category']} | Priority: {grievance['priority']} | Sentiment: {grievance['sentiment']}")
    
    if detailed:
        print(f"\nSummary:")
        print(f"  {grievance['summary']}")
        print(f"\nTags: {', '.join(grievance['tags'])}")
        print(f"\nFull Transcript:")
        print(f"  {grievance['transcript']}")
    else:
        # Truncate transcript for list view
        transcript_preview = grievance['transcript'][:100] + "..." if len(grievance['transcript']) > 100 else grievance['transcript']
        print(f"Preview: {transcript_preview}")
    
    print(f"{'='*80}")

def main():
    processor = GrievanceProcessor()
    
    if len(sys.argv) < 2:
        command = "list"
    else:
        command = sys.argv[1]
    
    if command == "list":
        # List all grievances
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print(f"\n📋 Showing last {limit} grievances:\n")
        
        grievances = processor.get_all_grievances(limit=limit)
        if not grievances:
            print("No grievances found in database.")
            return
        
        for grievance in grievances:
            print_grievance(grievance, detailed=False)
        
        print(f"\nTotal: {len(grievances)} grievances")
    
    elif command == "view":
        # View specific grievance by ID
        if len(sys.argv) < 3:
            print("Usage: python view_grievances.py view <grievance_id>")
            return
        
        grievance_id = sys.argv[2]
        grievance = processor.get_grievance(grievance_id)
        
        if grievance:
            print_grievance(grievance, detailed=True)
        else:
            print(f"Grievance with ID {grievance_id} not found.")
    
    elif command == "stats":
        # Show statistics
        stats = processor.get_statistics()
        
        print(f"\n📊 Grievance Statistics\n")
        print(f"{'='*80}")
        print(f"Total Grievances: {stats['total_grievances']}")
        
        print(f"\nBy Category:")
        for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}")
        
        print(f"\nBy Priority:")
        priority_order = {"High": 3, "Medium": 2, "Low": 1}
        for priority, count in sorted(stats['by_priority'].items(), key=lambda x: priority_order.get(x[0], 0), reverse=True):
            print(f"  {priority}: {count}")
        
        print(f"{'='*80}\n")
    
    elif command == "export":
        # Export all grievances to JSON
        output_file = sys.argv[2] if len(sys.argv) > 2 else "grievances_export.json"
        
        grievances = processor.get_all_grievances(limit=10000)
        
        with open(output_file, 'w') as f:
            json.dump(grievances, f, indent=2)
        
        print(f"✅ Exported {len(grievances)} grievances to {output_file}")
    
    elif command == "help":
        print("""
Grievance Management Utility

Commands:
  list [limit]      - List recent grievances (default: 10)
  view <id>        - View detailed grievance by ID
  stats            - Show statistics about all grievances
  export [file]    - Export all grievances to JSON file
  help             - Show this help message

Examples:
  python view_grievances.py list 20
  python view_grievances.py view abc123-def456-...
  python view_grievances.py stats
  python view_grievances.py export my_grievances.json
        """)
    
    else:
        print(f"Unknown command: {command}")
        print("Run 'python view_grievances.py help' for usage information.")

if __name__ == "__main__":
    main()