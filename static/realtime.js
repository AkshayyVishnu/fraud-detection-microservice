/**
 * Real-time WebSocket Client
 * Handles live transaction updates and fraud alerts
 */

class RealtimeClient {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;
    }

    connect() {
        // Initialize Socket.IO client
        if (typeof io === 'undefined') {
            console.error('Socket.IO library not loaded');
            return;
        }

        this.socket = io();

        // Connection event
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.connected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus(true);
            this.socket.emit('request_transactions');
        });

        // Disconnection event
        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
            this.connected = false;
            this.updateConnectionStatus(false);
            this.attemptReconnect();
        });

        // Status update
        this.socket.on('status', (data) => {
            console.log('WebSocket status:', data);
            if (data.connected) {
                this.updateConnectionStatus(true);
            }
        });

        // New transaction event
        this.socket.on('new_transaction', (transaction) => {
            console.log('New transaction received:', transaction);
            this.handleNewTransaction(transaction);
        });

        // Fraud alert event
        this.socket.on('fraud_alert', (alert) => {
            console.log('Fraud alert received:', alert);
            this.handleFraudAlert(alert);
        });

        // Transactions update
        this.socket.on('transactions_update', (data) => {
            console.log('Transactions update:', data);
            if (data.transactions && Array.isArray(data.transactions)) {
                this.updateTransactionList(data.transactions);
            }
        });
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay);
        } else {
            console.error('Max reconnection attempts reached');
            this.updateConnectionStatus(false, 'Connection lost');
        }
    }

    updateConnectionStatus(connected, message = null) {
        const statusIndicator = document.getElementById('wsConnectionStatus');
        if (!statusIndicator) return;

        if (connected) {
            statusIndicator.className = 'ws-status ws-status-connected';
            statusIndicator.innerHTML = `
                <span class="ws-status-dot"></span>
                <span class="ws-status-text">Live</span>
            `;
        } else {
            statusIndicator.className = 'ws-status ws-status-disconnected';
            statusIndicator.innerHTML = `
                <span class="ws-status-dot"></span>
                <span class="ws-status-text">${message || 'Disconnected'}</span>
            `;
        }
    }

    handleNewTransaction(transaction) {
        // Add to transaction list with animation
        this.addTransactionToList(transaction);
        
        // Update stats
        this.updateStats();
        
        // Add pulse animation to transaction card
        const transactionElement = document.querySelector(`[data-transaction-id="${transaction.id}"]`);
        if (transactionElement) {
            transactionElement.classList.add('transaction-pulse');
            setTimeout(() => {
                transactionElement.classList.remove('transaction-pulse');
            }, 1000);
        }
    }

    handleFraudAlert(alert) {
        // Show toast notification
        this.showFraudAlertToast(alert);
        
        // Play alert sound (if available)
        this.playAlertSound();
        
        // Update threat indicator
        this.updateThreatIndicator('HIGH');
        
        // Scroll to transaction if visible
        const transactionElement = document.querySelector(`[data-transaction-id="${alert.transaction_id}"]`);
        if (transactionElement) {
            transactionElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            transactionElement.classList.add('fraud-alert-highlight');
            setTimeout(() => {
                transactionElement.classList.remove('fraud-alert-highlight');
            }, 3000);
        }
    }

    showFraudAlertToast(alert) {
        const toast = document.createElement('div');
        toast.className = 'toast toast-fraud';
        toast.innerHTML = `
            <div class="toast-icon">
                <svg viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                </svg>
            </div>
            <div class="toast-content">
                <div class="toast-title">🚨 Fraud Detected</div>
                <div class="toast-message">
                    Transaction ${alert.transaction_id} flagged with ${(alert.fraud_probability * 100).toFixed(1)}% confidence
                </div>
                <div class="toast-meta">Amount: $${alert.amount.toFixed(2)}</div>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">×</button>
        `;

        const toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            // Create toast container if it doesn't exist
            const container = document.createElement('div');
            container.id = 'toastContainer';
            container.className = 'toast-container';
            document.body.appendChild(container);
            container.appendChild(toast);
        } else {
            toastContainer.appendChild(toast);
        }

        // Auto-remove after 5 seconds
        setTimeout(() => {
            toast.classList.add('toast-exit');
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    playAlertSound() {
        // Try to play alert sound if available
        const audio = document.getElementById('alertSound');
        if (audio) {
            audio.play().catch(e => console.log('Could not play alert sound:', e));
        }
    }

    updateThreatIndicator(level) {
        const indicator = document.getElementById('threatIndicator');
        if (!indicator) return;

        const levelClasses = {
            'LOW': 'status-safe',
            'MEDIUM': 'status-warning',
            'HIGH': 'status-danger'
        };

        indicator.className = `status-indicator ${levelClasses[level] || 'status-warning'}`;
        const statusText = indicator.querySelector('.status-text');
        if (statusText) {
            statusText.textContent = level === 'HIGH' ? 'Threat Detected' : 'Monitoring';
        }
    }

    addTransactionToList(transaction) {
        const transactionList = document.getElementById('transactionList');
        if (!transactionList) return;

        // Create transaction element
        const transactionElement = this.createTransactionElement(transaction);
        
        // Add to top of list
        if (transactionList.firstChild) {
            transactionList.insertBefore(transactionElement, transactionList.firstChild);
        } else {
            transactionList.appendChild(transactionElement);
        }

        // Limit list to 20 items
        const items = transactionList.querySelectorAll('.transaction-item');
        if (items.length > 20) {
            items[items.length - 1].remove();
        }

        // Auto-scroll if at top
        if (transactionList.scrollTop === 0) {
            transactionList.scrollTop = 0;
        }
    }

    createTransactionElement(transaction) {
        const riskClass = transaction.risk_level?.toLowerCase() || 'low';
        const statusClass = transaction.status || 'approved';
        
        const element = document.createElement('div');
        element.className = `transaction-item transaction-item-${riskClass} transaction-new`;
        element.setAttribute('data-transaction-id', transaction.id);
        
        element.innerHTML = `
            <div class="transaction-header">
                <span class="transaction-id">${transaction.id}</span>
                <span class="transaction-badge transaction-badge-${statusClass}">${statusClass}</span>
            </div>
            <div class="transaction-body">
                <div class="transaction-amount">$${transaction.amount?.toFixed(2) || '0.00'}</div>
                <div class="transaction-meta">
                    <span class="transaction-risk risk-${riskClass}">${transaction.risk_level || 'UNKNOWN'}</span>
                    <span class="transaction-prob">${(transaction.fraud_probability * 100).toFixed(1)}%</span>
                </div>
            </div>
            <div class="transaction-time">${new Date(transaction.timestamp).toLocaleTimeString()}</div>
        `;

        // Remove 'new' class after animation
        setTimeout(() => {
            element.classList.remove('transaction-new');
        }, 1000);

        return element;
    }

    updateTransactionList(transactions) {
        const transactionList = document.getElementById('transactionList');
        if (!transactionList) return;

        // Clear loading state
        transactionList.innerHTML = '';

        // Add transactions
        transactions.forEach(transaction => {
            const element = this.createTransactionElement(transaction);
            transactionList.appendChild(element);
        });
    }

    updateStats() {
        // Trigger stats refresh
        if (typeof refreshData === 'function') {
            refreshData();
        }
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
            this.connected = false;
            this.updateConnectionStatus(false);
        }
    }
}

// Initialize realtime client when DOM is ready
let realtimeClient;

document.addEventListener('DOMContentLoaded', () => {
    realtimeClient = new RealtimeClient();
    realtimeClient.connect();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (realtimeClient) {
        realtimeClient.disconnect();
    }
});

