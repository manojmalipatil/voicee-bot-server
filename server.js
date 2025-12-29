const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public')); // Serve static files

// Database connection
const db = new sqlite3.Database('./grievances.db', (err) => {
    if (err) {
        console.error('Error opening database:', err.message);
    } else {
        console.log('Connected to SQLite database');
    }
});

// API Routes

// Get all grievances
app.get('/api/grievances', (req, res) => {
    const query = `
        SELECT * FROM grievances 
        ORDER BY datetime(timestamp) DESC
    `;
    
    db.all(query, [], (err, rows) => {
        if (err) {
            res.status(500).json({ error: err.message });
            return;
        }
        res.json(rows);
    });
});

// Get single grievance by ID
app.get('/api/grievances/:id', (req, res) => {
    const query = 'SELECT * FROM grievances WHERE id = ?';
    
    db.get(query, [req.params.id], (err, row) => {
        if (err) {
            res.status(500).json({ error: err.message });
            return;
        }
        res.json(row);
    });
});

// Get dashboard statistics
app.get('/api/stats', (req, res) => {
    const stats = {};
    
    // Total count
    db.get('SELECT COUNT(*) as total FROM grievances', [], (err, row) => {
        if (err) {
            res.status(500).json({ error: err.message });
            return;
        }
        stats.total = row.total;
        
        // Category distribution
        db.all(`
            SELECT category, COUNT(*) as count 
            FROM grievances 
            GROUP BY category 
            ORDER BY count DESC
        `, [], (err, rows) => {
            if (err) {
                res.status(500).json({ error: err.message });
                return;
            }
            stats.categoryDistribution = rows;
            
            // Priority distribution
            db.all(`
                SELECT priority, COUNT(*) as count 
                FROM grievances 
                GROUP BY priority
            `, [], (err, rows) => {
                if (err) {
                    res.status(500).json({ error: err.message });
                    return;
                }
                stats.priorityDistribution = rows;
                
                // Sentiment distribution
                db.all(`
                    SELECT sentiment, COUNT(*) as count 
                    FROM grievances 
                    GROUP BY sentiment
                `, [], (err, rows) => {
                    if (err) {
                        res.status(500).json({ error: err.message });
                        return;
                    }
                    stats.sentimentDistribution = rows;
                    
                    // Weekly trend (last 4 weeks)
                    db.all(`
                        SELECT 
                            strftime('%W', timestamp) as week,
                            COUNT(*) as count
                        FROM grievances
                        WHERE datetime(timestamp) >= datetime('now', '-30 days')
                        GROUP BY week
                        ORDER BY week
                    `, [], (err, rows) => {
                        if (err) {
                            res.status(500).json({ error: err.message });
                            return;
                        }
                        stats.weeklyTrend = rows;
                        res.json(stats);
                    });
                });
            });
        });
    });
});

// Update grievance status
app.put('/api/grievances/:id/status', (req, res) => {
    const { status, notes } = req.body;
    const query = `
        UPDATE grievances 
        SET priority = ?, summary = ? 
        WHERE id = ?
    `;
    
    db.run(query, [status, notes, req.params.id], function(err) {
        if (err) {
            res.status(500).json({ error: err.message });
            return;
        }
        res.json({ 
            message: 'Updated successfully',
            changes: this.changes 
        });
    });
});

// Add new grievance
app.post('/api/grievances', (req, res) => {
    const { 
        id, timestamp, transcript, category, 
        priority, sentiment, summary, tags, location 
    } = req.body;
    
    const query = `
        INSERT INTO grievances 
        (id, timestamp, transcript, category, priority, sentiment, summary, tags, created_at, location)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
    `;
    
    db.run(query, [
        id, timestamp, transcript, category, 
        priority, sentiment, summary, tags, location
    ], function(err) {
        if (err) {
            res.status(500).json({ error: err.message });
            return;
        }
        res.json({ 
            message: 'Grievance added successfully',
            id: this.lastID 
        });
    });
});

// SSE endpoint for real-time updates
app.get('/api/stream', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    
    // Send initial connection message
    res.write('data: {"type":"connected"}\n\n');
    
    // Set up database trigger listener (polling approach)
    const interval = setInterval(() => {
        db.get('SELECT COUNT(*) as count FROM grievances', [], (err, row) => {
            if (!err) {
                res.write(`data: ${JSON.stringify({type: 'update', count: row.count})}\n\n`);
            }
        });
    }, 5000); // Check every 5 seconds
    
    req.on('close', () => {
        clearInterval(interval);
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});

// Graceful shutdown
process.on('SIGINT', () => {
    db.close((err) => {
        if (err) {
            console.error(err.message);
        }
        console.log('Database connection closed');
        process.exit(0);
    });
});