/**
 * Fraud Time Machine - Temporal Analysis Visualization
 * Displays fraud patterns over time with interactive heatmap and session detection
 */

class FraudTimeMachine {
    constructor() {
        this.temporalData = null;
        this.sessions = [];
        this.velocityLevel = 'low';

        this.elements = {
            heatmap: document.getElementById('temporalHeatmap'),
            sessionsList: document.getElementById('sessionsList'),
            sessionsCount: document.getElementById('sessionsCount'),
            velocityFill: document.getElementById('velocityFill'),
            velocityValue: document.getElementById('velocityValue')
        };
    }

    async loadData() {
        try {
            // Load temporal data
            const temporalResponse = await fetch('/api/temporal-data');
            const temporalData = await temporalResponse.json();
            this.temporalData = temporalData.buckets || [];

            // Load fraud sessions from network endpoint
            const networkResponse = await fetch('/api/fraud-network');
            const networkData = await networkResponse.json();
            this.sessions = networkData.sessions || [];

            this.render();
        } catch (error) {
            console.error('Failed to load temporal data:', error);
            this.showError();
        }
    }

    render() {
        this.renderHeatmap();
        this.renderSessions();
        this.updateVelocity();
    }

    renderHeatmap() {
        if (!this.elements.heatmap || !this.temporalData.length) return;

        // Group buckets into rows (8 hours per row = 16 buckets per row)
        const bucketsPerRow = 16;
        const rows = [];

        for (let i = 0; i < this.temporalData.length; i += bucketsPerRow) {
            const rowBuckets = this.temporalData.slice(i, i + bucketsPerRow);
            const startHour = Math.floor((i * 30) / 60);
            rows.push({ startHour, buckets: rowBuckets });
        }

        let html = '';

        rows.forEach(row => {
            const endHour = row.startHour + 8;
            const label = `${String(row.startHour).padStart(2, '0')}:00`;

            const cells = row.buckets.map(bucket => {
                const riskLevel = this.getRiskLevel(bucket.fraud_rate);
                return `
                    <div class="heatmap-cell-enhanced" 
                         data-risk="${riskLevel}"
                         data-bucket="${bucket.bucket_index}"
                         data-time="${bucket.time_label}"
                         data-transactions="${bucket.transaction_count}"
                         data-frauds="${bucket.fraud_count}"
                         data-rate="${(bucket.fraud_rate * 100).toFixed(2)}"
                         title="${bucket.time_label} - ${bucket.fraud_count} frauds">
                    </div>
                `;
            }).join('');

            html += `
                <div class="heatmap-row-labeled">
                    <span class="heatmap-label">${label}</span>
                    <div class="heatmap-cells">${cells}</div>
                </div>
            `;
        });

        this.elements.heatmap.innerHTML = html;
        this.attachHeatmapInteractions();
    }

    getRiskLevel(fraudRate) {
        if (fraudRate > 0.05) return 'critical';
        if (fraudRate > 0.03) return 'high';
        if (fraudRate > 0.01) return 'medium';
        if (fraudRate > 0.001) return 'low';
        return 'none';
    }

    attachHeatmapInteractions() {
        const cells = this.elements.heatmap.querySelectorAll('.heatmap-cell-enhanced');

        // Create tooltip element if it doesn't exist
        if (!this.tooltip) {
            this.tooltip = document.createElement('div');
            this.tooltip.className = 'temporal-tooltip';
            document.body.appendChild(this.tooltip);
        }


        cells.forEach(cell => {
            cell.addEventListener('mouseenter', (e) => {
                const time = cell.dataset.time;
                const txns = cell.dataset.transactions;
                const frauds = cell.dataset.frauds;
                const rate = cell.dataset.rate;
                const risk = cell.dataset.risk;

                const riskColors = {
                    critical: '#f85149',
                    high: '#f0883e',
                    medium: '#e3b341',
                    low: '#3fb950',
                    none: '#8b949e'
                };

                this.tooltip.innerHTML = `
                    <div style="font-weight:600; margin-bottom:8px; color:var(--color-text-primary);">
                        ${time}
                    </div>
                    <div style="display:flex; flex-direction:column; gap:4px; font-size:12px;">
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:var(--color-text-secondary);">Transactions</span>
                            <span style="font-weight:500;">${txns}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:var(--color-text-secondary);">Frauds Detected</span>
                            <span style="font-weight:500; color:#f85149;">${frauds}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:var(--color-text-secondary);">Fraud Rate</span>
                            <span style="font-weight:500;">${rate}%</span>
                        </div>
                        <div style="margin-top:6px; padding-top:6px; border-top:1px solid var(--color-border);">
                            <span style="display:inline-block; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:600; background:${riskColors[risk]}20; color:${riskColors[risk]};">
                                ${risk.toUpperCase()} RISK
                            </span>
                        </div>
                    </div>
                `;

                this.tooltip.style.opacity = '1';
                this.tooltip.style.left = (e.clientX + 12) + 'px';
                this.tooltip.style.top = (e.clientY + 12) + 'px';
            });

            cell.addEventListener('mousemove', (e) => {
                this.tooltip.style.left = (e.clientX + 12) + 'px';
                this.tooltip.style.top = (e.clientY + 12) + 'px';
            });

            cell.addEventListener('mouseleave', () => {
                this.tooltip.style.opacity = '0';
            });

            cell.addEventListener('click', () => {
                const bucket = cell.dataset.bucket;
                this.highlightBucket(bucket);
            });
        });
    }

    renderSessions() {
        if (!this.elements.sessionsList) return;

        if (!this.sessions.length) {
            this.elements.sessionsList.innerHTML = `
                <div class="session-card">
                    <div class="session-icon">
                        <svg viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                        </svg>
                    </div>
                    <div class="session-info">
                        <div class="session-title">No active attack windows detected</div>
                        <div class="session-meta">
                            <span class="session-stat">System is monitoring for fraud patterns</span>
                        </div>
                    </div>
                </div>
            `;
            this.elements.sessionsCount.textContent = '0';
            return;
        }

        const html = this.sessions.slice(0, 5).map((session, index) => {
            // Calculate velocity (transactions per minute)
            const velocity = (session.count / Math.max(1, session.duration_minutes)).toFixed(1);
            const riskLevel = velocity > 0.5 ? 'High Velocity' : velocity > 0.2 ? 'Moderate' : 'Normal';

            return `
            <div class="session-card" 
                 data-session="${session.id}" 
                 data-velocity="${velocity}"
                 data-risk="${riskLevel}"
                 onclick="window.timeMachine.highlightSession('${session.id}')"
                 onmouseenter="window.timeMachine.showSessionTooltip(event, '${session.id}', ${index})"
                 onmouseleave="window.timeMachine.hideSessionTooltip()">
                <div class="session-icon">
                    <svg width="18" height="18" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clip-rule="evenodd"/>
                    </svg>
                </div>
                <div class="session-info">
                    <div class="session-title">Attack Window ${index + 1}</div>
                    <div class="session-meta">
                        <span class="session-stat">
                            <svg width="12" height="12" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd"/>
                            </svg>
                            ${session.start_time} - ${session.end_time}
                        </span>
                        <span class="session-stat">
                            <svg width="12" height="12" viewBox="0 0 20 20" fill="currentColor">
                                <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/>
                                <path fill-rule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clip-rule="evenodd"/>
                            </svg>
                            ${session.count} txns
                        </span>
                        <span class="session-stat velocity-badge ${velocity > 0.5 ? 'high' : ''}">${velocity}/min</span>
                    </div>
                </div>
                <span class="session-amount">$${session.total_amount.toFixed(0)}</span>
            </div>
        `}).join('');

        this.elements.sessionsList.innerHTML = html;
        this.elements.sessionsCount.textContent = this.sessions.length;
    }

    showSessionTooltip(event, sessionId, index) {
        const session = this.sessions[index];
        if (!session) return;

        const velocity = (session.count / Math.max(1, session.duration_minutes)).toFixed(2);
        const avgAmount = (session.total_amount / session.count).toFixed(2);

        // Determine attack pattern type
        let patternType = 'Standard Fraud';
        if (velocity > 0.5) patternType = 'Rapid Burst Attack';
        else if (session.total_amount > 5000) patternType = 'High Value Fraud';
        else if (session.count > 10) patternType = 'Volume Attack';

        this.tooltip.innerHTML = `
            <div style="font-weight:600; margin-bottom:8px; border-bottom:1px solid var(--color-border); padding-bottom:6px;">
                Attack Window ${index + 1}
            </div>
            <div style="display:flex; flex-direction:column; gap:6px; font-size:12px;">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--color-text-secondary);">Pattern Type</span>
                    <span style="font-weight:500; color:#f85149;">${patternType}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--color-text-secondary);">Velocity</span>
                    <span style="font-weight:500;">${velocity} txns/min</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--color-text-secondary);">Avg. Amount</span>
                    <span style="font-weight:500;">$${avgAmount}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--color-text-secondary);">Duration</span>
                    <span style="font-weight:500;">${session.duration_minutes} min</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--color-text-secondary);">Total at Risk</span>
                    <span style="font-weight:600; color:#f0883e;">$${session.total_amount.toFixed(2)}</span>
                </div>
            </div>
        `;

        this.tooltip.style.opacity = '1';
        this.tooltip.style.left = (event.clientX + 12) + 'px';
        this.tooltip.style.top = (event.clientY + 12) + 'px';
    }

    hideSessionTooltip() {
        if (this.tooltip) {
            this.tooltip.style.opacity = '0';
        }
    }

    highlightBucket(bucketIndex) {
        // Find transactions in this time bucket and highlight in network
        if (window.fraudNetwork) {
            console.log('Highlighting bucket:', bucketIndex);
        }
    }

    highlightSession(sessionId) {
        if (window.fraudNetwork) {
            window.fraudNetwork.highlightSession(sessionId);
        }
    }

    updateVelocity() {
        if (!this.elements.velocityFill || !this.elements.velocityValue) return;

        // Calculate fraud velocity based on recent data
        const recentBuckets = this.temporalData.slice(-10);
        const avgFraudRate = recentBuckets.reduce((sum, b) => sum + b.fraud_rate, 0) / recentBuckets.length;

        let level = 'low';
        let velocityMultiplier = 0.2;

        if (avgFraudRate > 0.04) {
            level = 'critical';
            velocityMultiplier = 4.5;
        } else if (avgFraudRate > 0.025) {
            level = 'high';
            velocityMultiplier = 2.8;
        } else if (avgFraudRate > 0.01) {
            level = 'moderate';
            velocityMultiplier = 1.5;
        }

        this.velocityLevel = level;
        this.elements.velocityFill.className = `velocity-bar-fill ${level}`;

        this.elements.velocityValue.textContent = `${velocityMultiplier.toFixed(1)}x`;
    }

    showError() {
        if (this.elements.heatmap) {
            this.elements.heatmap.innerHTML = `
                <div class="heatmap-row-labeled">
                    <span class="heatmap-label">Error</span>
                    <div class="heatmap-cells">
                        <div style="color: var(--color-text-tertiary); font-size: 0.875rem; padding: 2rem;">
                            Failed to load temporal data
                        </div>
                    </div>
                </div>
            `;
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('temporalHeatmap')) {
        window.timeMachine = new FraudTimeMachine();
        window.timeMachine.loadData();
    }
});
