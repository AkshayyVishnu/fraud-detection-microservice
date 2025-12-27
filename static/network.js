/**
 * Fraud Network Graph - D3.js Force-Directed Visualization
 * Professional network graph showing transaction relationships
 */

class FraudNetworkGraph {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);

        this.options = {
            width: options.width || this.container.clientWidth || 800,
            height: options.height || 500,
            nodeRadius: options.nodeRadius || { min: 6, max: 24 },
            linkDistance: options.linkDistance || 80,
            chargeStrength: options.chargeStrength || -200,
            ...options
        };

        this.nodes = [];
        this.edges = [];
        this.sessions = [];

        this.svg = null;
        this.simulation = null;
        this.tooltip = null;

        this.init();
    }

    init() {
        // Clear container
        this.container.innerHTML = '';

        // Create SVG
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', '100%')
            .attr('height', this.options.height)
            .attr('viewBox', `0 0 ${this.options.width} ${this.options.height}`)
            .attr('class', 'fraud-network-svg');

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
        try {
            this.showLoading();

            const response = await fetch('/api/fraud-network');
            const data = await response.json();

            if (data.error) {
                this.showError(data.error);
                return;
            }

            this.nodes = data.nodes || [];
            this.edges = data.edges || [];
            this.sessions = data.sessions || [];

            this.render();
            this.updateStats(data.stats);

        } catch (error) {
            console.error('Failed to load network data:', error);
            this.showError('Failed to load network data');
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

        // Clear existing
        this.edgesGroup.selectAll('*').remove();
        this.nodesGroup.selectAll('*').remove();

        // Scale for node sizes based on amount
        const amountExtent = d3.extent(this.nodes, d => d.amount);
        const radiusScale = d3.scaleSqrt()
            .domain(amountExtent)
            .range([this.options.nodeRadius.min, this.options.nodeRadius.max]);

        // Create force simulation
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.edges)
                .id(d => d.id)
                .distance(this.options.linkDistance)
                .strength(d => d.strength * 0.5))
            .force('charge', d3.forceManyBody()
                .strength(this.options.chargeStrength))
            .force('center', d3.forceCenter(
                this.options.width / 2,
                this.options.height / 2))
            .force('collision', d3.forceCollide()
                .radius(d => radiusScale(d.amount) + 4));

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
            .call(this.drag(this.simulation));

        // Node circles
        nodes.append('circle')
            .attr('r', d => radiusScale(d.amount))
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', d => d.is_fraud ? '#f85149' : 'rgba(255,255,255,0.2)')
            .attr('stroke-width', d => d.is_fraud ? 2 : 1)
            .style('cursor', 'pointer');

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
        nodes.on('mouseenter', (event, d) => this.showTooltip(event, d))
            .on('mouseleave', () => this.hideTooltip())
            .on('click', (event, d) => this.onNodeClick(d));

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

    onNodeClick(node) {
        console.log('Node clicked:', node);
        // Could open a detail panel or navigate to transaction view
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
