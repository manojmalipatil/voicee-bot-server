// Configuration
const API_BASE_URL = 'http://localhost:3000/api';

// State management
let grievances = [];
let selectedGrievance = null;
let stats = null;

// Category color mapping
const categoryColors = {
    'POSH': 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200',
    'Managerial': 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    'Data': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    'Hygiene': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'Compensation': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    'Workplace Environment': 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    'Conflict': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    'Career': 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200',
    'Attendance': 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
};

// Priority color mapping
const priorityColors = {
    'Critical': 'text-red-700 dark:text-red-300',
    'High': 'text-red-600 dark:text-red-400',
    'Medium': 'text-amber-600 dark:text-amber-400',
    'Low': 'text-gray-500 dark:text-gray-400'
};

// Sentiment color mapping
const sentimentColors = {
    'Angry': 'bg-red-500',
    'Concerned': 'bg-orange-400',
    'Frustrated': 'bg-yellow-500',
    'Neutral': 'bg-blue-400',
    'Suggestion': 'bg-emerald-500',
    'Positive': 'bg-green-500'
};

// Initialize dashboard
async function initDashboard() {
    try {
        await loadStats();
        await loadGrievances();
        setupEventListeners();
        setupRealtimeUpdates();
        console.log('Dashboard initialized successfully');
    } catch (error) {
        console.error('Error initializing dashboard:', error);
        showError('Failed to load dashboard data');
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        stats = await response.json();
        updateStatsUI();
    } catch (error) {
        console.error('Error loading stats:', error);
        throw error;
    }
}

// Load grievances
async function loadGrievances() {
    try {
        const response = await fetch(`${API_BASE_URL}/grievances`);
        grievances = await response.json();
        updateGrievanceTable();
        
        // Load first grievance details
        if (grievances.length > 0) {
            await loadGrievanceDetails(grievances[0].id);
        }
    } catch (error) {
        console.error('Error loading grievances:', error);
        throw error;
    }
}

// Update statistics UI
function updateStatsUI() {
    if (!stats) return;
    
    // Update total count
    const totalElement = document.querySelector('[class*="text-4xl"]');
    if (totalElement) {
        totalElement.textContent = stats.total;
    }
    
    // Update top category
    if (stats.categoryDistribution && stats.categoryDistribution.length > 0) {
        const topCategory = stats.categoryDistribution[0];
        const categoryNameElement = document.querySelector('.text-3xl');
        if (categoryNameElement) {
            categoryNameElement.innerHTML = topCategory.category.replace(' ', '<br/>');
        }
        
        const categoryCountElement = categoryNameElement?.nextElementSibling?.querySelector('.text-sm');
        if (categoryCountElement) {
            categoryCountElement.textContent = `${topCategory.count} Active Cases`;
        }
    }
    
    // Update resolution status (mock calculation)
    const openCount = stats.priorityDistribution?.find(p => p.priority === 'High')?.count || 0;
    const investigatingCount = stats.priorityDistribution?.find(p => p.priority === 'Medium')?.count || 0;
    const resolvedCount = stats.priorityDistribution?.find(p => p.priority === 'Low')?.count || 0;
    
    const statusElements = document.querySelectorAll('.text-sm.font-bold.text-gray-900');
    if (statusElements.length >= 3) {
        statusElements[0].textContent = openCount || 42;
        statusElements[1].textContent = investigatingCount || 35;
        statusElements[2].textContent = resolvedCount || 21;
    }
}

// Update grievance table
function updateGrievanceTable() {
    const tbody = document.querySelector('tbody');
    if (!tbody) return;
    
    tbody.innerHTML = grievances.map((grievance, index) => {
        const categoryClass = categoryColors[grievance.category] || categoryColors['Attendance'];
        const priorityClass = priorityColors[grievance.priority] || priorityColors['Low'];
        const sentimentClass = sentimentColors[grievance.sentiment] || sentimentColors['Neutral'];
        const isHighlighted = index === 0 ? 'bg-blue-50/50 dark:bg-blue-900/10' : '';
        
        // Format timestamp
        const date = new Date(grievance.timestamp);
        const formattedDate = date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric' 
        }) + ', ' + date.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: false 
        });
        
        // Determine status based on priority (mock logic)
        let status = 'Open';
        let statusClass = 'bg-primary/10 text-primary border-primary/20';
        
        if (grievance.priority === 'High' || grievance.priority === 'Critical') {
            status = 'Investigating';
            statusClass = 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 border-amber-200 dark:border-amber-800';
        } else if (grievance.priority === 'Low') {
            status = 'Resolved';
            statusClass = 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200 border-emerald-200 dark:border-emerald-800';
        }
        
        return `
            <tr class="${isHighlighted} cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors" data-id="${grievance.id}">
                <td class="px-6 py-4">
                    <span class="font-mono text-sm ${index === 0 ? 'font-bold text-gray-900 dark:text-white bg-white dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded px-2 py-1' : 'font-medium text-gray-600 dark:text-gray-400'}">${grievance.id}</span>
                </td>
                <td class="px-6 py-4">
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded text-xs font-bold ${categoryClass}">
                        ${grievance.category || 'Uncategorized'}
                    </span>
                </td>
                <td class="px-6 py-4">
                    <span class="inline-flex items-center gap-1 text-xs font-bold ${priorityClass}">
                        <span class="size-1.5 rounded-full ${grievance.priority === 'Critical' ? 'animate-pulse' : ''}" style="background-color: currentColor"></span> 
                        ${grievance.priority || 'Medium'}
                    </span>
                </td>
                <td class="px-6 py-4 text-sm text-gray-600 dark:text-gray-300 font-medium">
                    ${grievance.location || 'Unknown'}
                </td>
                <td class="px-6 py-4">
                    <div class="flex items-center gap-2">
                        <div class="size-2.5 rounded-full ${sentimentClass}"></div>
                        <span class="text-sm font-medium text-gray-700 dark:text-gray-200">${grievance.sentiment || 'Neutral'}</span>
                    </div>
                </td>
                <td class="px-6 py-4 text-sm text-gray-600 dark:text-gray-300">
                    ${formattedDate}
                </td>
                <td class="px-6 py-4">
                    <div class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold ${statusClass} border">
                        ${status}
                    </div>
                </td>
                <td class="px-6 py-4 text-right">
                    <span class="material-symbols-outlined text-gray-400">chevron_right</span>
                </td>
            </tr>
        `;
    }).join('');
    
    // Add click listeners to rows
    tbody.querySelectorAll('tr').forEach(row => {
        row.addEventListener('click', () => {
            const id = row.getAttribute('data-id');
            loadGrievanceDetails(id);
        });
    });
}

// Load grievance details
async function loadGrievanceDetails(id) {
    try {
        const response = await fetch(`${API_BASE_URL}/grievances/${id}`);
        selectedGrievance = await response.json();
        updateDetailPanel();
    } catch (error) {
        console.error('Error loading grievance details:', error);
        showError('Failed to load grievance details');
    }
}

// Update detail panel
function updateDetailPanel() {
    if (!selectedGrievance) return;
    
    // Update case reference
    const refElement = document.querySelector('aside h2 + p');
    if (refElement) {
        refElement.textContent = `REF: ${selectedGrievance.id}`;
    }
    
    // Update tags
    const tagsContainer = document.querySelector('.flex.flex-wrap.gap-2');
    if (tagsContainer && selectedGrievance.tags) {
        const tags = selectedGrievance.tags.split(',').map(tag => tag.trim());
        tagsContainer.innerHTML = tags.map(tag => 
            `<span class="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300">${tag}</span>`
        ).join('');
    }
    
    // Update summary
    const summaryElement = document.querySelector('.bg-blue-50 p');
    if (summaryElement && selectedGrievance.summary) {
        summaryElement.innerHTML = selectedGrievance.summary;
    }
    
    // Update transcript
    const transcriptElement = document.querySelector('.font-serif');
    if (transcriptElement && selectedGrievance.transcript) {
        // Split transcript into paragraphs and add redaction styling
        const paragraphs = selectedGrievance.transcript.split('\n').filter(p => p.trim());
        transcriptElement.innerHTML = paragraphs.map(p => `<p class="mb-4">${p}</p>`).join('');
    }
}

// Setup event listeners
function setupEventListeners() {
    // Save button
    const saveButton = document.querySelector('button[class*="bg-primary"]');
    if (saveButton) {
        saveButton.addEventListener('click', saveGrievanceUpdate);
    }
    
    // Filter button
    const filterButton = document.querySelector('button:has(.material-symbols-outlined:contains("filter_list"))');
    if (filterButton) {
        filterButton.addEventListener('click', () => {
            alert('Filter functionality - Coming soon!');
        });
    }
    
    // Export button
    const exportButton = document.querySelector('button:has(.material-symbols-outlined:contains("download"))');
    if (exportButton) {
        exportButton.addEventListener('click', exportData);
    }
}

// Save grievance update
async function saveGrievanceUpdate() {
    if (!selectedGrievance) return;
    
    const statusSelect = document.querySelector('select');
    const notesTextarea = document.querySelector('textarea');
    
    const updateData = {
        status: statusSelect.value,
        notes: notesTextarea.value
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}/grievances/${selectedGrievance.id}/status`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updateData)
        });
        
        if (response.ok) {
            showSuccess('Case updated successfully');
            await loadGrievances();
            await loadStats();
        } else {
            throw new Error('Update failed');
        }
    } catch (error) {
        console.error('Error updating grievance:', error);
        showError('Failed to update case');
    }
}

// Export data
function exportData() {
    const csv = convertToCSV(grievances);
    downloadCSV(csv, 'grievances_export.csv');
    showSuccess('Data exported successfully');
}

// Convert to CSV
function convertToCSV(data) {
    const headers = ['ID', 'Timestamp', 'Category', 'Priority', 'Sentiment', 'Location', 'Summary'];
    const rows = data.map(g => [
        g.id,
        g.timestamp,
        g.category,
        g.priority,
        g.sentiment,
        g.location,
        `"${(g.summary || '').replace(/"/g, '""')}"`
    ]);
    
    return [headers, ...rows].map(row => row.join(',')).join('\n');
}

// Download CSV
function downloadCSV(csv, filename) {
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

// Setup real-time updates
function setupRealtimeUpdates() {
    const eventSource = new EventSource(`${API_BASE_URL}/stream`);
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'update') {
            console.log('Database updated, reloading...');
            loadGrievances();
            loadStats();
        }
    };
    
    eventSource.onerror = (error) => {
        console.error('SSE Error:', error);
        eventSource.close();
        // Retry connection after 5 seconds
        setTimeout(setupRealtimeUpdates, 5000);
    };
}

// Utility functions
function showSuccess(message) {
    console.log('✓ ' + message);
    // You can implement a toast notification here
}

function showError(message) {
    console.error('✗ ' + message);
    // You can implement a toast notification here
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDashboard);
} else {
    initDashboard();
}