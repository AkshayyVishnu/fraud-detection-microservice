/**
 * The Merchant Shield - Dashboard JavaScript
 * Real-time data fetching and UI updates
 */

// ============================================================================
// CONFIGURATION
// ============================================================================
const CONFIG = {
    refreshInterval: 30000, // 30 seconds
    apiEndpoints: {
        stats: '/api/stats',
        transactions: '/api/transactions',
        temporal: '/api/temporal-data'
    }
};

// ============================================================================
// STATE
// ============================================================================
let state = {
    transactions: [],
    stats: null,
    temporalData: null,
    lastUpdate: null
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatDate(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
    });
}

function getRiskClass(probability) {
    if (probability > 0.7) return 'high';
    if (probability > 0.4) return 'medium';
    return 'low';
}

// ============================================================================
// API FUNCTIONS
// ============================================================================
async function fetchStats() {
    try {
        const response = await fetch(CONFIG.apiEndpoints.stats);
        const data = await response.json();
        state.stats = data;
        updateStatsUI(data);
        updateThreatIndicator(data.threat_level);
    } catch (error) {
        console.error('Failed to fetch stats:', error);
    }
}

async function fetchTransactions() {
    try {
        const response = await fetch(`${CONFIG.apiEndpoints.transactions}?limit=10`);
        const data = await response.json();
        state.transactions = data.transactions;
        updateTransactionListUI(data.transactions);
        updateRiskDistribution(data.transactions);
    } catch (error) {
        console.error('Failed to fetch transactions:', error);
    }
}

async function fetchTemporalData() {
    try {
        const response = await fetch(CONFIG.apiEndpoints.temporal);
        const data = await response.json();
        state.temporalData = data;
        renderHeatmap(data.buckets);
    } catch (error) {
        console.error('Failed to fetch temporal data:', error);
    }
}

// ============================================================================
// UI UPDATE FUNCTIONS
// ============================================================================
function updateStatsUI(stats) {
    document.getElementById('statTotal').textContent = stats.total_transactions;
    document.getElementById('statFlagged').textContent = stats.flagged_count;
    document.getElementById('statBlocked').textContent = stats.blocked_count;
    document.getElementById('statAmountAtRisk').textContent = formatCurrency(stats.amount_at_risk);
    
    // Update last update time
    state.lastUpdate = new Date();
    document.getElementById('lastUpdate').textContent = `Last updated: ${formatTime(state.lastUpdate)}`;
}

function updateThreatIndicator(level) {
    const indicator = document.getElementById('threatIndicator');
    const statusText = indicator.querySelector('.status-text');
    
    if (level === 'ELEVATED') {
        indicator.classList.add('elevated');
        statusText.textContent = 'Elevated';
    } else {
        indicator.classList.remove('elevated');
        statusText.textContent = 'Monitoring';
    }
}

function updateTransactionListUI(transactions) {
    const container = document.getElementById('transactionList');
    
    if (!transactions || transactions.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
                </svg>
                <div class="empty-state-title">No transactions yet</div>
                <div class="empty-state-text">Transactions will appear here as they are analyzed</div>
            </div>
        `;
        return;
    }
    
    container.innerHTML = transactions.map(tx => `
        <div class="transaction-item" data-id="${tx.id}">
            <div class="transaction-info">
                <span class="transaction-id">${tx.id}</span>
                <div class="transaction-meta">
                    <span>${formatTime(tx.timestamp)}</span>
                    <span>•</span>
                    <span>${tx.risk_level} Risk</span>
                </div>
            </div>
            <div class="transaction-status">
                <span class="transaction-amount">${formatCurrency(tx.amount)}</span>
                <span class="status-badge ${tx.status}">${tx.status}</span>
            </div>
        </div>
    `).join('');
}

function updateRiskDistribution(transactions) {
    if (!transactions || transactions.length === 0) return;
    
    const total = transactions.length;
    const low = transactions.filter(t => t.risk_level === 'LOW').length;
    const medium = transactions.filter(t => t.risk_level === 'MEDIUM').length;
    const high = transactions.filter(t => t.risk_level === 'HIGH').length;
    
    const lowPercent = Math.round((low / total) * 100);
    const mediumPercent = Math.round((medium / total) * 100);
    const highPercent = Math.round((high / total) * 100);
    
    // Update bars
    document.getElementById('riskBarLow').style.width = `${lowPercent}%`;
    document.getElementById('riskBarMedium').style.width = `${mediumPercent}%`;
    document.getElementById('riskBarHigh').style.width = `${highPercent}%`;
    
    // Update percentages
    document.getElementById('riskPercentLow').textContent = `${lowPercent}%`;
    document.getElementById('riskPercentMedium').textContent = `${mediumPercent}%`;
    document.getElementById('riskPercentHigh').textContent = `${highPercent}%`;
}

// ============================================================================
// HEATMAP RENDERING
// ============================================================================
function renderHeatmap(buckets) {
    const container = document.getElementById('temporalHeatmap');
    
    // Create 6 rows of 16 cells each (96 buckets total = 48 hours / 30 min each)
    const rows = [];
    for (let i = 0; i < 6; i++) {
        const rowBuckets = buckets.slice(i * 16, (i + 1) * 16);
        const cells = rowBuckets.map(bucket => {
            let riskLevel = 'low';
            if (bucket.fraud_rate > 0.05) riskLevel = 'critical';
            else if (bucket.fraud_rate > 0.03) riskLevel = 'high';
            else if (bucket.fraud_rate > 0.01) riskLevel = 'medium';
            
            return `
                <div class="heatmap-cell" 
                     data-risk="${riskLevel}"
                     data-bucket="${bucket.bucket_index}"
                     data-time="${bucket.time_label}"
                     data-transactions="${bucket.transaction_count}"
                     data-frauds="${bucket.fraud_count}"
                     data-rate="${(bucket.fraud_rate * 100).toFixed(2)}">
                </div>
            `;
        }).join('');
        
        rows.push(`<div class="heatmap-row">${cells}</div>`);
    }
    
    container.innerHTML = rows.join('');
    
    // Add hover interactions
    addHeatmapInteractions();
}

function addHeatmapInteractions() {
    const cells = document.querySelectorAll('.heatmap-cell');
    let tooltip = null;
    
    cells.forEach(cell => {
        cell.addEventListener('mouseenter', (e) => {
            const data = {
                time: cell.dataset.time,
                transactions: cell.dataset.transactions,
                frauds: cell.dataset.frauds,
                rate: cell.dataset.rate
            };
            
            showTooltip(e, data);
        });
        
        cell.addEventListener('mouseleave', () => {
            hideTooltip();
        });
    });
}

function showTooltip(event, data) {
    hideTooltip(); // Remove any existing tooltip
    
    const tooltip = document.createElement('div');
    tooltip.className = 'heatmap-tooltip';
    tooltip.innerHTML = `
        <div class="tooltip-time">${data.time}</div>
        <div class="tooltip-stats">
            <div class="tooltip-stat">
                <span>Transactions</span>
                <span class="tooltip-stat-value">${data.transactions}</span>
            </div>
            <div class="tooltip-stat">
                <span>Fraud Cases</span>
                <span class="tooltip-stat-value">${data.frauds}</span>
            </div>
            <div class="tooltip-stat">
                <span>Fraud Rate</span>
                <span class="tooltip-stat-value">${data.rate}%</span>
            </div>
        </div>
    `;
    
    document.body.appendChild(tooltip);
    
    const rect = event.target.getBoundingClientRect();
    tooltip.style.left = `${rect.left + rect.width / 2}px`;
    tooltip.style.top = `${rect.top}px`;
}

function hideTooltip() {
    const existing = document.querySelector('.heatmap-tooltip');
    if (existing) {
        existing.remove();
    }
}

// ============================================================================
// GLOBAL FUNCTIONS
// ============================================================================
function refreshData() {
    fetchStats();
    fetchTransactions();
    fetchTemporalData();
}

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Initial data fetch
    refreshData();
    
    // Set up auto-refresh
    setInterval(refreshData, CONFIG.refreshInterval);
});
