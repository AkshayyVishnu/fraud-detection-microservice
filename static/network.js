/**
 * Fraud Network Graph - D3.js Force-Directed Visualization
 * Professional network graph showing transaction relationships
 */

class FraudNetworkGraph {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);

        // Get actual container dimensions
        const rect = this.container.getBoundingClientRect();
        const containerWidth = rect.width || this.container.clientWidth || 800;
        const containerHeight = rect.height || this.container.clientHeight || 500;

        console.log('[FraudNetwork] Container dimensions:', containerWidth, 'x', containerHeight);

        this.options = {
            width: options.width || containerWidth || 800,
            height: options.height || containerHeight || 500,
            nodeRadius: options.nodeRadius || { min: 6, max: 24 },
            linkDistance: options.linkDistance || 80,
            chargeStrength: options.chargeStrength || -150,
            ...options
        };

        console.log('[FraudNetwork] Using dimensions:', this.options.width, 'x', this.options.height);

        this.nodes = [];
        this.edges = [];
        this.sessions = [];

        this.svg = null;
        this.simulation = null;
        this.tooltip = null;
        this.nodeElements = null;
        this.edgeElements = null;

        this.init();
    }

    init() {
        // Clear container
        this.container.innerHTML = '';

        // Use fixed dimensions that we know work
        const width = Math.max(this.options.width, 600);
        const height = Math.max(this.options.height, 400);

        console.log('[FraudNetwork] Creating SVG:', width, 'x', height);

        // Create SVG with explicit dimensions
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('class', 'fraud-network-svg')
            .style('display', 'block');

        // Update options with actual used dimensions
        this.options.width = width;
        this.options.height = height;

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.3, 3])
            .on('zoom', (event) => {
                this.graphGroup.attr('transform', event.transform);
            });

        this.svg.call(zoom);

        // Create main group for graph elements
        this.graphGroup = this.svg.append('g')
            .attr('class', 'graph-group');

        // Create groups for edges and nodes (edges first so nodes appear on top)
        this.edgesGroup = this.graphGroup.append('g').attr('class', 'edges-group');
        this.nodesGroup = this.graphGroup.append('g').attr('class', 'nodes-group');

        // Create tooltip
        this.createTooltip();

        // Add legend
        this.createLegend();
    }

    createTooltip() {
        this.tooltip = d3.select('body')
            .append('div')
            .attr('class', 'network-tooltip')
            .style('opacity', 0)
            .style('position', 'fixed')
            .style('pointer-events', 'none')
            .style('z-index', 1000);
    }

    createLegend() {
        const legendData = [
            { label: 'Fraud (Blocked)', color: '#f85149' },
            { label: 'High Risk', color: '#d29922' },
            { label: 'Low Risk', color: '#3fb950' }
        ];

        const legend = this.svg.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(16, 16)`);

        legend.append('rect')
            .attr('x', -8)
            .attr('y', -8)
            .attr('width', 140)
            .attr('height', legendData.length * 24 + 16)
            .attr('rx', 6)
            .attr('fill', 'rgba(22, 27, 34, 0.9)')
            .attr('stroke', 'rgba(255,255,255,0.1)');

        legendData.forEach((item, i) => {
            const g = legend.append('g')
                .attr('transform', `translate(0, ${i * 24})`);

            g.append('circle')
                .attr('r', 5)
                .attr('fill', item.color);

            g.append('text')
                .attr('x', 14)
                .attr('y', 4)
                .attr('fill', '#8b949e')
                .attr('font-size', '11px')
                .text(item.label);
        });
    }

    async loadData() {
        console.log('[FraudNetwork] Starting loadData...');
        try {
            this.showLoading();

            console.log('[FraudNetwork] Fetching /api/fraud-network...');
            const response = await fetch('/api/fraud-network');
            console.log('[FraudNetwork] Response status:', response.status);

            const data = await response.json();
            console.log('[FraudNetwork] Data received:', {
                nodes: data.nodes?.length || 0,
                edges: data.edges?.length || 0,
                sessions: data.sessions?.length || 0,
                hasError: !!data.error
            });

            if (data.error) {
                console.error('[FraudNetwork] Server error:', data.error);
                this.showError(data.error);
                return;
            }

            this.nodes = data.nodes || [];
            this.edges = data.edges || [];
            this.sessions = data.sessions || [];

            if (this.nodes.length === 0) {
                console.warn('[FraudNetwork] No nodes received, showing placeholder');
                this.showError('No network data available');
                return;
            }

            console.log('[FraudNetwork] Calling render()...');
            this.render();
            console.log('[FraudNetwork] Render complete');
            this.updateStats(data.stats);

        } catch (error) {
            console.error('[FraudNetwork] Error loading:', error);
            this.showError('Failed to load network data: ' + error.message);
        }
    }

    showLoading() {
        this.graphGroup.selectAll('*').remove();

        this.graphGroup.append('text')
            .attr('x', this.options.width / 2)
            .attr('y', this.options.height / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#8b949e')
            .attr('font-size', '14px')
            .text('Loading network data...');
    }

    showError(message) {
        this.graphGroup.selectAll('*').remove();

        this.graphGroup.append('text')
            .attr('x', this.options.width / 2)
            .attr('y', this.options.height / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#f85149')
            .attr('font-size', '14px')
            .text(message);
    }

    render() {
        if (this.nodes.length === 0) {
            this.showError('No network data available');
            return;
        }

        console.log('[FraudNetwork] Re-building SVG groups...');

        // CRITICAL FIX: showLoading() removes all children including edges/nodes groups
        // We must re-create them here to ensure they exist in the DOM
        this.graphGroup.selectAll('*').remove(); // Clear loading text/spinner

        // Re-create groups (edges first so they are behind nodes)
        this.edgesGroup = this.graphGroup.append('g').attr('class', 'edges-group');
        this.nodesGroup = this.graphGroup.append('g').attr('class', 'nodes-group');

        // Initialize node positions at center with random offset
        const centerX = this.options.width / 2;
        const centerY = this.options.height / 2;
        this.nodes.forEach((node, i) => {
            if (node.x === undefined) node.x = centerX + (Math.random() - 0.5) * 200;
            if (node.y === undefined) node.y = centerY + (Math.random() - 0.5) * 200;
        });
        console.log('[FraudNetwork] Initialized node positions. First node at:', this.nodes[0]?.x, this.nodes[0]?.y);

        // Scale for node sizes based on amount
        const amountExtent = d3.extent(this.nodes, d => d.amount);
        console.log('[FraudNetwork] Amount extent:', amountExtent);
        const radiusScale = d3.scaleSqrt()
            .domain(amountExtent)
            .range([this.options.nodeRadius.min, this.options.nodeRadius.max]);

        // Create force simulation with optimized settings
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.edges)
                .id(d => d.id)
                .distance(this.options.linkDistance)
                .strength(d => d.strength * 0.3))
            .force('charge', d3.forceManyBody()
                .strength(this.options.chargeStrength)
                .distanceMax(300))
            .force('center', d3.forceCenter(
                this.options.width / 2,
                this.options.height / 2))
            .force('collision', d3.forceCollide()
                .radius(d => radiusScale(d.amount) + 4))
            .alphaDecay(0.05)
            .velocityDecay(0.4);

        // Create edges
        const edges = this.edgesGroup.selectAll('line')
            .data(this.edges)
            .enter()
            .append('line')
            .attr('class', 'network-edge')
            .attr('stroke', d => this.getEdgeColor(d))
            .attr('stroke-width', d => Math.max(1, d.strength * 3))
            .attr('stroke-opacity', d => 0.3 + d.strength * 0.4)
            .attr('stroke-dasharray', d =>
                d.types.includes('amount_pattern') ? '4,2' : 'none');

        // Create nodes
        const nodes = this.nodesGroup.selectAll('g')
            .data(this.nodes)
            .enter()
            .append('g')
            .attr('class', 'network-node')
            .attr('transform', d => `translate(${d.x}, ${d.y})`)
            .call(this.drag(this.simulation));

        console.log('[FraudNetwork] Created node groups:', nodes.size());

        // Node circles with explicit styling
        nodes.append('circle')
            .attr('r', d => Math.max(6, radiusScale(d.amount)))
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', d => d.is_fraud ? '#f85149' : 'rgba(255,255,255,0.3)')
            .attr('stroke-width', d => d.is_fraud ? 2 : 1)
            .style('cursor', 'pointer')
            .style('opacity', 1);

        console.log('[FraudNetwork] Added circles to nodes');

        // Add glow effect for fraud nodes
        nodes.filter(d => d.is_fraud)
            .select('circle')
            .style('filter', 'drop-shadow(0 0 6px rgba(248, 81, 73, 0.5))');

        // Node labels for large amounts
        nodes.filter(d => d.amount > 1000)
            .append('text')
            .attr('dy', d => radiusScale(d.amount) + 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#8b949e')
            .attr('font-size', '10px')
            .text(d => `$${d.amount.toFixed(0)}`);

        // Hover interactions
        nodes.on('mouseenter', (event, d) => {
            this.showTooltip(event, d);
            this.highlightConnected(d);
        })
            .on('mouseleave', () => {
                this.hideTooltip();
                this.resetHighlight();
            })
            .on('click', (event, d) => this.onNodeClick(d));

        // Add pulse animation to fraud nodes
        this.addFraudPulse();

        // Store references
        this.nodeElements = nodes;
        this.edgeElements = edges;

        // Update positions on tick
        this.simulation.on('tick', () => {
            edges
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            nodes.attr('transform', d => `translate(${d.x}, ${d.y})`);
        });
    }

    addFraudPulse() {
        // Add animated pulse ring to fraud nodes
        const fraudNodes = this.nodesGroup.selectAll('.network-node')
            .filter(d => d.is_fraud);

        fraudNodes.insert('circle', 'circle')
            .attr('class', 'pulse-ring')
            .attr('r', d => 8)
            .attr('fill', 'none')
            .attr('stroke', '#f85149')
            .attr('stroke-width', 2)
            .attr('opacity', 0.6);
    }

    highlightConnected(node) {
        // Find connected node IDs
        const connectedIds = new Set([node.id]);
        this.edges.forEach(edge => {
            const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
            const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
            if (sourceId === node.id) connectedIds.add(targetId);
            if (targetId === node.id) connectedIds.add(sourceId);
        });

        // Dim non-connected nodes
        this.nodeElements
            .transition().duration(200)
            .style('opacity', d => connectedIds.has(d.id) ? 1 : 0.2);

        // Highlight connected edges
        this.edgeElements
            .transition().duration(200)
            .style('opacity', d => {
                const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
                const targetId = typeof d.target === 'object' ? d.target.id : d.target;
                return (sourceId === node.id || targetId === node.id) ? 1 : 0.1;
            })
            .attr('stroke-width', d => {
                const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
                const targetId = typeof d.target === 'object' ? d.target.id : d.target;
                return (sourceId === node.id || targetId === node.id) ? 4 : 1;
            });
    }

    resetHighlight() {
        this.nodeElements
            .transition().duration(300)
            .style('opacity', 1);

        this.edgeElements
            .transition().duration(300)
            .style('opacity', d => 0.3 + d.strength * 0.4)
            .attr('stroke-width', d => Math.max(1, d.strength * 3));
    }

    highlightSession(sessionId) {
        // Find session
        const session = this.sessions.find(s => s.id === sessionId);
        if (!session) return;

        const sessionNodeIds = new Set(session.transaction_ids);

        // Highlight session nodes
        this.nodeElements
            .transition().duration(300)
            .style('opacity', d => sessionNodeIds.has(d.id) ? 1 : 0.15)
            .select('circle')
            .attr('stroke-width', d => sessionNodeIds.has(d.id) ? 3 : 1);

        // After 3 seconds, reset
        setTimeout(() => this.resetHighlight(), 3000);
    }

    getNodeColor(node) {
        if (node.is_fraud) {
            return '#f85149';
        }
        if (node.risk_score > 0.7) {
            return '#d29922';
        }
        if (node.risk_score > 0.4) {
            return '#58a6ff';
        }
        return '#3fb950';
    }

    getEdgeColor(edge) {
        if (edge.types.includes('confirmed_fraud')) {
            return '#f85149';
        }
        if (edge.types.includes('temporal_strong')) {
            return '#58a6ff';
        }
        return '#484f58';
    }

    showTooltip(event, node) {
        const content = `
            <div class="tooltip-header ${node.is_fraud ? 'fraud' : ''}">
                <span class="tooltip-id">${node.id}</span>
                <span class="tooltip-status">${node.is_fraud ? 'FRAUD' : 'OK'}</span>
            </div>
            <div class="tooltip-body">
                <div class="tooltip-row">
                    <span>Amount</span>
                    <strong>$${node.amount.toFixed(2)}</strong>
                </div>
                <div class="tooltip-row">
                    <span>Time</span>
                    <strong>${node.time_label}</strong>
                </div>
                <div class="tooltip-row">
                    <span>Risk Score</span>
                    <strong>${(node.risk_score * 100).toFixed(0)}%</strong>
                </div>
                <div class="tooltip-row">
                    <span>V14 Value</span>
                    <strong>${node.v14.toFixed(2)}</strong>
                </div>
            </div>
        `;

        this.tooltip
            .html(content)
            .style('left', (event.clientX + 16) + 'px')
            .style('top', (event.clientY - 16) + 'px')
            .style('opacity', 1);
    }

    hideTooltip() {
        this.tooltip.style('opacity', 0);
    }

    // Filter to show only fraud nodes (High/Medium risk)
    filterFraudOnly() {
        // Dim low risk (green) nodes
        this.nodesGroup.selectAll('.network-node')
            .transition().duration(500)
            .style('opacity', d => d.is_fraud || d.risk_score > 0.3 ? 1 : 0.1);

        // Dim edges connected to low risk nodes
        this.edgesGroup.selectAll('.network-edge')
            .transition().duration(500)
            .style('opacity', d => {
                const s = d.source;
                const t = d.target;
                const sKeep = s.is_fraud || s.risk_score > 0.3;
                const tKeep = t.is_fraud || t.risk_score > 0.3;
                return (sKeep && tKeep) ? (0.3 + d.strength * 0.4) : 0.05;
            });
    }

    resetFilter() {
        // Reset all opacities
        this.nodesGroup.selectAll('.network-node')
            .transition().duration(500)
            .style('opacity', 1);

        this.edgesGroup.selectAll('.network-edge')
            .transition().duration(500)
            .style('opacity', d => 0.3 + d.strength * 0.4);
    }

    async showClusterExplanation(sessionId) {
        console.log('[FraudNetwork] Fetching explanation for:', sessionId);
        try {
            const response = await fetch(`/api/cluster-explanation?session_id=${sessionId}`);
            const data = await response.json();

            if (data.explanation) {
                this.renderExplanationPanel(sessionId, data.explanation);
            }
        } catch (e) {
            console.error('[FraudNetwork] Failed to get explanation:', e);
        }
    }

    renderExplanationPanel(sessionId, explanation) {
        // Remove existing panel
        d3.select('.explanation-panel').remove();

        const panel = d3.select('body').append('div')
            .attr('class', 'explanation-panel')
            .style('position', 'fixed')
            .style('bottom', '20px')
            .style('right', '20px')
            .style('width', '300px')
            .style('background', 'var(--color-surface-1)')
            .style('border', '1px solid var(--color-border)')
            .style('border-radius', 'var(--radius-lg)')
            .style('padding', '16px')
            .style('box-shadow', '0 10px 25px rgba(0,0,0,0.5)')
            .style('z-index', 100);

        panel.html(`
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <h3 style="font-size:14px;font-weight:600;margin:0;">Cluster Insights (SHAP)</h3>
                <button onclick="this.parentElement.parentElement.remove()" style="background:none;border:none;color:var(--color-text-tertiary);cursor:pointer;">&times;</button>
            </div>
            <div style="display:flex;flex-direction:column;gap:8px;">
                ${explanation.map(item => `
                    <div style="display:flex;align-items:center;gap:8px;font-size:12px;">
                        <span style="flex:1;color:var(--color-text-secondary);">${item.feature}</span>
                        <div style="flex:2;height:6px;background:var(--color-surface-3);border-radius:4px;overflow:hidden;">
                            <div style="width:${item.importance * 100}%;height:100%;background:${item.contribution === 'positive' ? 'var(--color-danger)' : 'var(--color-success)'}"></div>
                        </div>
                        <span style="width:30px;text-align:right;font-family:monospace;">${item.value > 0 ? '+' : ''}${item.value.toFixed(1)}</span>
                    </div>
                `).join('')}
            </div>
            <div style="margin-top:10px;font-size:10px;color:var(--color-text-tertiary);text-align:center;">
                Top factors driving risk score
            </div>
        `);
    }

    onNodeClick(node) {
        console.log('Node clicked:', node);
        this.highlightConnected(node);
        // Clean session ID check - assume TXN nodes might belong to a session
        // For now, simpler: just show explanation for this "node cluster"
        this.showClusterExplanation(node.id);
    }

    drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }

    updateStats(stats) {
        if (!stats) return;

        // Update stats display if elements exist
        const statsEl = document.getElementById('networkStats');
        if (statsEl) {
            statsEl.innerHTML = `
                <span class="stat-item">
                    <span class="stat-value">${stats.total_nodes || 0}</span>
                    <span class="stat-label">Nodes</span>
                </span>
                <span class="stat-item">
                    <span class="stat-value">${stats.total_edges || 0}</span>
                    <span class="stat-label">Connections</span>
                </span>
                <span class="stat-item">
                    <span class="stat-value">${stats.fraud_count || 0}</span>
                    <span class="stat-label">Fraud Cases</span>
                </span>
                <span class="stat-item">
                    <span class="stat-value">${stats.sessions_detected || 0}</span>
                    <span class="stat-label">Sessions</span>
                </span>
            `;
        }
    }

    // Filter to show only fraud nodes and their connections
    filterFraudOnly() {
        const fraudIds = new Set(this.nodes.filter(n => n.is_fraud).map(n => n.id));

        this.nodesGroup.selectAll('.network-node')
            .style('opacity', d => fraudIds.has(d.id) ? 1 : 0.2);

        this.edgesGroup.selectAll('.network-edge')
            .style('opacity', d =>
                (fraudIds.has(d.source.id) || fraudIds.has(d.target.id)) ? 0.6 : 0.05);
    }

    // Reset filters
    resetFilter() {
        this.nodesGroup.selectAll('.network-node')
            .style('opacity', 1);

        this.edgesGroup.selectAll('.network-edge')
            .style('opacity', d => 0.3 + d.strength * 0.4);
    }

    // Highlight a specific session
    highlightSession(sessionId) {
        const session = this.sessions.find(s => s.id === sessionId);
        if (!session) return;

        const txnIds = new Set(session.transaction_ids);

        this.nodesGroup.selectAll('.network-node')
            .style('opacity', d => txnIds.has(d.id) ? 1 : 0.2);

        this.edgesGroup.selectAll('.network-edge')
            .style('opacity', d =>
                (txnIds.has(d.source.id) && txnIds.has(d.target.id)) ? 0.8 : 0.05);

        // Also show explanation
        this.showClusterExplanation(sessionId);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Check if network container exists
    const container = document.getElementById('fraudNetwork');
    if (container) {
        window.fraudNetwork = new FraudNetworkGraph('fraudNetwork');
        window.fraudNetwork.loadData();
    }
});
