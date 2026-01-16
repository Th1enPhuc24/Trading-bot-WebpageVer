/**
 * Trading Dashboard v3.0 - TradingView Lightweight Charts
 * Features: Candlestick Chart, Markers, Price Lines, Trade Navigation, Replay Mode
 */

// ============================================
// GLOBAL STATE
// ============================================
let socket = null;

// TradingView Lightweight Charts
let tvChart = null;
let candlestickSeries = null;
let markers = [];
let priceLines = [];

// Chart.js for mini charts
let equityChart = null;
let pnlChart = null;

// Data storage
let ohlcData = [];
let tradeEvents = [];
let allTrades = [];
let currentSelectedTrade = null;

// Replay mode state
let replayMode = false;
let replayData = null;
let replayIndex = 0;
let replayPlaying = false;
let replaySpeed = 5;
let replayInterval = null;

// Statistics
let stats = {
    totalTrades: 0,
    longTrades: 0,
    shortTrades: 0,
    winningTrades: 0,
    losingTrades: 0,
    winRate: 0,
    totalPnl: 0,
    currentBalance: 10000,
    startingBalance: 10000,
    maxDrawdown: 0,
    sharpeRatio: 0,
    equityCurve: [],
    pnlHistory: []
};

let lastPrice = null;

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initializeTradingViewChart();
    initializeEquityChart();
    initializePnlChart();
    initializeSocket();
    initializePlaybackControls();
    initializeDownloadButton();
    initializeResizeHandler();
    initializeTrainingControls();
});

// ============================================
// TRADINGVIEW LIGHTWEIGHT CHARTS
// ============================================
function initializeTradingViewChart() {
    const container = document.getElementById('tradingview-chart');
    if (!container) {
        console.error('TradingView chart container not found');
        return;
    }

    // Get container dimensions
    const chartContainer = document.getElementById('price-chart-container');
    const width = chartContainer.clientWidth;
    const height = chartContainer.clientHeight || 600;

    // Create chart with dark theme
    tvChart = LightweightCharts.createChart(container, {
        width: width,
        height: height,
        layout: {
            background: { type: 'solid', color: '#0a0b0d' },
            textColor: '#9ca3af'
        },
        grid: {
            vertLines: { color: 'rgba(107, 114, 128, 0.1)' },
            horzLines: { color: 'rgba(107, 114, 128, 0.1)' }
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Magnet,
            vertLine: {
                color: 'rgba(59, 130, 246, 0.5)',
                width: 1,
                style: LightweightCharts.LineStyle.Dashed,
                labelBackgroundColor: '#3b82f6'
            },
            horzLine: {
                color: 'rgba(59, 130, 246, 0.5)',
                width: 1,
                style: LightweightCharts.LineStyle.Dashed,
                labelBackgroundColor: '#3b82f6'
            }
        },
        timeScale: {
            borderColor: 'rgba(107, 114, 128, 0.3)',
            timeVisible: true,
            secondsVisible: false
        },
        rightPriceScale: {
            borderColor: 'rgba(107, 114, 128, 0.3)'
        }
    });

    // Create candlestick series
    candlestickSeries = tvChart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderUpColor: '#10b981',
        borderDownColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444'
    });

    // Subscribe to crosshair move for tooltip
    tvChart.subscribeCrosshairMove(handleCrosshairMove);

    console.log('TradingView chart initialized');
}

function handleCrosshairMove(param) {
    const tooltip = document.getElementById('chart-tooltip');
    if (!tooltip) return;

    if (!param.time || !param.point) {
        tooltip.style.display = 'none';
        return;
    }

    const price = param.seriesData.get(candlestickSeries);
    if (!price) {
        tooltip.style.display = 'none';
        return;
    }

    // Format date
    const date = new Date(param.time * 1000);
    const dateStr = date.toLocaleString();

    // Find trade events at this exact time (for marker tooltip)
    let tradeInfo = '';
    const eventsAtTime = tradeEvents.filter(e => e.time === param.time);

    if (eventsAtTime.length > 0) {
        eventsAtTime.forEach(event => {
            // Find corresponding trade for full details
            const trade = allTrades.find(t => t.id === event.trade_id || t.trade_id === event.trade_id);

            if (event.type === 'BUY' || event.type === 'SELL') {
                // Entry marker tooltip
                const tradeNum = event.trade_id || '?';
                const tradeType = event.type === 'BUY' ? 'Long' : 'Short';
                tradeInfo += `
                    <div class="tooltip-trade entry">
                        <div class="trade-header">📈 Trade #${tradeNum} - ${tradeType}</div>
                        <div class="trade-detail">Entry: $${event.price?.toFixed(2) || '--'}</div>
                        <div class="trade-detail">TP: $${event.tp_price?.toFixed(2) || '--'}</div>
                        <div class="trade-detail">SL: $${event.sl_price?.toFixed(2) || '--'}</div>
                    </div>
                `;
            } else if (event.type.startsWith('CLOSE')) {
                // Exit marker tooltip
                const tradeNum = event.trade_id || '?';
                let exitReason = event.reason || 'UNKNOWN';
                if (exitReason === 'TAKE_PROFIT') exitReason = 'TP Hit';
                else if (exitReason === 'STOP_LOSS') exitReason = 'SL Hit';
                else if (exitReason === 'TIMEOUT') exitReason = 'Timeout';

                const pnl = event.pnl || 0;
                const pnlClass = pnl >= 0 ? 'profit' : 'loss';
                const pnlSign = pnl >= 0 ? '+' : '';

                tradeInfo += `
                    <div class="tooltip-trade exit ${pnlClass}">
                        <div class="trade-header">📊 Trade #${tradeNum} - Exit</div>
                        <div class="trade-detail">Type: ${event.type === 'CLOSE_LONG' ? 'Long' : 'Short'}</div>
                        <div class="trade-detail">Exit: $${event.price?.toFixed(2) || '--'}</div>
                        <div class="trade-detail">Reason: ${exitReason}</div>
                        <div class="trade-detail pnl ${pnlClass}">P&L: ${pnlSign}$${pnl.toFixed(2)}</div>
                    </div>
                `;
            }
        });
    }

    tooltip.innerHTML = `
        <div class="tooltip-time">${dateStr}</div>
        <div class="tooltip-prices">
            <span>O: ${price.open?.toFixed(2) || '--'}</span>
            <span>H: ${price.high?.toFixed(2) || '--'}</span>
            <span>L: ${price.low?.toFixed(2) || '--'}</span>
            <span>C: ${price.close?.toFixed(2) || '--'}</span>
        </div>
        ${tradeInfo}
    `;

    // Position tooltip
    const chartRect = document.getElementById('tradingview-chart').getBoundingClientRect();
    let left = param.point.x + 20;
    let top = param.point.y - 20;

    if (left + 250 > chartRect.width) {
        left = param.point.x - 270;
    }

    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
    tooltip.style.display = 'block';
}

function initializeResizeHandler() {
    const resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
            if (tvChart) {
                tvChart.applyOptions({
                    width: entry.contentRect.width,
                    height: entry.contentRect.height || 600
                });
            }
        }
    });

    const container = document.getElementById('price-chart-container');
    if (container) {
        resizeObserver.observe(container);
    }

    // Also handle window resize
    window.addEventListener('resize', () => {
        if (tvChart) {
            const container = document.getElementById('price-chart-container');
            tvChart.applyOptions({
                width: container.clientWidth,
                height: container.clientHeight || 600
            });
        }
    });
}

// ============================================
// LOAD DATA INTO CHART
// ============================================
function loadOHLCData(data) {
    if (!data || data.length === 0) return;

    console.log('loadOHLCData called with', data.length, 'candles');
    console.log('Sample candle:', data[0]);

    // Convert timestamps and filter valid data
    const processedData = data
        .map(d => {
            // Convert time to Unix timestamp if needed
            let time = d.time;
            if (typeof time === 'string') {
                // Parse ISO string to Unix timestamp (seconds)
                time = Math.floor(new Date(time).getTime() / 1000);
            }
            return {
                time: time,
                open: parseFloat(d.open),
                high: parseFloat(d.high),
                low: parseFloat(d.low),
                close: parseFloat(d.close)
            };
        })
        .filter(d => d.time && !isNaN(d.time) && d.open && d.high && d.low && d.close)
        .sort((a, b) => a.time - b.time)
        .filter((d, i, arr) => i === 0 || d.time !== arr[i - 1].time);

    ohlcData = processedData;

    console.log('Processed candles:', processedData.length);
    if (processedData.length > 0) {
        console.log('First candle:', processedData[0]);
        console.log('Last candle:', processedData[processedData.length - 1]);
    }

    if (candlestickSeries && processedData.length > 0) {
        candlestickSeries.setData(processedData);
        tvChart.timeScale().fitContent();
        console.log(`Chart updated with ${processedData.length} candles`);
    } else {
        console.warn('Could not update chart: candlestickSeries =', !!candlestickSeries, 'data length =', processedData.length);
    }
}

function loadTradeEvents(events) {
    if (!events || events.length === 0) return;

    tradeEvents = events;
    updateMarkers();
}

function updateMarkers() {
    if (!candlestickSeries || tradeEvents.length === 0) return;

    // Build markers from trade events using createMarkerFromEvent
    const chartMarkers = tradeEvents
        .map(event => createMarkerFromEvent(event))
        .filter(m => m !== null);

    candlestickSeries.setMarkers(chartMarkers);
    markers = chartMarkers;
}

// ============================================
// PRICE LINES (Entry, TP, SL)
// ============================================
function showTradeLines(trade) {
    // Clear existing price lines
    clearPriceLines();

    if (!trade || !candlestickSeries) return;

    const entryEvent = tradeEvents.find(e => e.trade_id === trade.trade_id && (e.type === 'BUY' || e.type === 'SELL'));
    const exitEvent = tradeEvents.find(e => e.trade_id === trade.trade_id && e.type.startsWith('CLOSE'));

    if (entryEvent) {
        // Entry line (solid)
        const entryLine = candlestickSeries.createPriceLine({
            price: entryEvent.price,
            color: '#3b82f6',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            axisLabelVisible: true,
            title: 'Entry'
        });
        priceLines.push(entryLine);

        // TP line (dashed green)
        if (entryEvent.tp_price) {
            const tpLine = candlestickSeries.createPriceLine({
                price: entryEvent.tp_price,
                color: '#10b981',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: 'TP'
            });
            priceLines.push(tpLine);
        }

        // SL line (dashed red)
        if (entryEvent.sl_price) {
            const slLine = candlestickSeries.createPriceLine({
                price: entryEvent.sl_price,
                color: '#ef4444',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: 'SL'
            });
            priceLines.push(slLine);
        }
    }

    if (exitEvent) {
        // Exit line
        const exitLine = candlestickSeries.createPriceLine({
            price: exitEvent.price,
            color: exitEvent.pnl >= 0 ? '#a855f7' : '#06b6d4',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: true,
            title: 'Exit'
        });
        priceLines.push(exitLine);
    }
}

function clearPriceLines() {
    priceLines.forEach(line => {
        if (candlestickSeries) {
            candlestickSeries.removePriceLine(line);
        }
    });
    priceLines = [];
}

// ============================================
// TRADE NAVIGATION
// ============================================
function navigateToTrade(trade) {
    if (!trade || !tvChart) return;

    currentSelectedTrade = trade;

    // Find trade events
    const entryEvent = tradeEvents.find(e => e.trade_id === trade.trade_id && (e.type === 'BUY' || e.type === 'SELL'));
    const exitEvent = tradeEvents.find(e => e.trade_id === trade.trade_id && e.type.startsWith('CLOSE'));

    if (entryEvent) {
        // Scroll and zoom to trade timeframe
        const startTime = entryEvent.time - 300; // 5 minutes before
        const endTime = exitEvent ? exitEvent.time + 300 : entryEvent.time + 600;

        tvChart.timeScale().setVisibleRange({
            from: startTime,
            to: endTime
        });

        // Show price lines for this trade
        showTradeLines({ trade_id: entryEvent.trade_id });
    }

    // Highlight in trade table
    highlightTradeRow(trade);
}

function highlightTradeRow(trade) {
    // Remove previous highlights
    document.querySelectorAll('.trades-table tbody tr').forEach(row => {
        row.classList.remove('selected');
    });

    // Find and highlight the row
    const rows = document.querySelectorAll('.trades-table tbody tr');
    rows.forEach((row, idx) => {
        if (allTrades[allTrades.length - 1 - idx]?.trade_id === trade.trade_id) {
            row.classList.add('selected');
            row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    });
}

// ============================================
// MINI CHARTS (Equity & P&L)
// ============================================
function initializeEquityChart() {
    const ctx = document.getElementById('equityChart');
    if (!ctx) return;

    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Equity',
                data: [],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: {
                    grid: { color: 'rgba(107, 114, 128, 0.1)' },
                    ticks: { color: '#9ca3af' }
                }
            }
        }
    });
}

function initializePnlChart() {
    const ctx = document.getElementById('pnlChart');
    if (!ctx) return;

    pnlChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'P&L',
                data: [],
                backgroundColor: [],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: {
                    grid: { color: 'rgba(107, 114, 128, 0.1)' },
                    ticks: { color: '#9ca3af' }
                }
            }
        }
    });
}

function updateEquityChart(equity) {
    if (!equityChart) return;
    stats.equityCurve.push(equity);
    equityChart.data.labels = stats.equityCurve.map((_, i) => i + 1);
    equityChart.data.datasets[0].data = stats.equityCurve;
    equityChart.update('none');
}

function updatePnlChart(pnl) {
    if (!pnlChart) return;
    stats.pnlHistory.push(pnl);
    pnlChart.data.labels = stats.pnlHistory.map((_, i) => i + 1);
    pnlChart.data.datasets[0].data = stats.pnlHistory;
    pnlChart.data.datasets[0].backgroundColor = stats.pnlHistory.map(p => p >= 0 ? '#10b981' : '#ef4444');
    pnlChart.update('none');
}

// ============================================
// SOCKET.IO CONNECTION
// ============================================
function initializeSocket() {
    socket = io();

    socket.on('connect', () => {
        console.log('Connected to dashboard server');
        updateConnectionStatus(true);
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from dashboard server');
        updateConnectionStatus(false);
    });

    socket.on('initial_state', (data) => {
        console.log('Received initial state:', data);

        // Load OHLC data for candlestick chart
        if (data.ohlc && data.ohlc.length > 0) {
            loadOHLCData(data.ohlc);
        }

        // Load trade events for markers
        if (data.trade_events && data.trade_events.length > 0) {
            loadTradeEvents(data.trade_events);
        }

        // Load equity curve
        if (data.equity && data.equity.length > 0) {
            stats.equityCurve = data.equity;
            if (equityChart) {
                equityChart.data.labels = data.equity.map((_, i) => i + 1);
                equityChart.data.datasets[0].data = data.equity;
                equityChart.update('none');
            }
        }

        // Load trades for table display (but don't recalculate stats)
        if (data.trades && data.trades.length > 0) {
            data.trades.forEach((trade, idx) => {
                addTradeToTable(trade);
                allTrades.push(trade);

                // Build P&L history for chart only
                const pnl = trade.pnl || 0;
                stats.pnlHistory.push(pnl);
            });

            // Update P&L chart
            if (pnlChart && stats.pnlHistory.length > 0) {
                pnlChart.data.labels = stats.pnlHistory.map((_, i) => i + 1);
                pnlChart.data.datasets[0].data = stats.pnlHistory;
                pnlChart.data.datasets[0].backgroundColor = stats.pnlHistory.map(p => p >= 0 ? '#10b981' : '#ef4444');
                pnlChart.update('none');
            }
        }

        // Load ALL stats from server (this is the source of truth to prevent double-counting)
        if (data.stats) {
            stats.totalTrades = data.stats.total_trades || 0;
            stats.winningTrades = data.stats.winning_trades || 0;
            stats.losingTrades = data.stats.losing_trades || 0;
            stats.winRate = data.stats.win_rate || 0;
            stats.totalPnl = data.stats.total_pnl || 0;
            stats.startingBalance = data.stats.starting_balance || 10000;
            stats.currentBalance = data.stats.current_balance || (stats.startingBalance + stats.totalPnl);
            stats.maxDrawdown = data.stats.max_drawdown || 0;
            stats.sharpeRatio = data.stats.sharpe_ratio || 0;
            stats.longTrades = data.stats.long_trades || 0;
            stats.shortTrades = data.stats.short_trades || 0;
        }

        updateStatsDisplay();

        // Show playback controls when we have backtest data (OHLC + trades)
        if ((data.ohlc && data.ohlc.length > 0) || (data.trades && data.trades.length > 0)) {
            const controls = document.getElementById('playback-controls');
            if (controls) {
                controls.style.display = 'flex';
            }

            // Store data for scrubbing
            replayMode = true;
            replayData = {
                ohlc: data.ohlc || [],
                trade_events: data.trade_events || [],
                trades: data.trades || [],
                equity: data.equity || []
            };

            // Setup scrubber
            const scrubber = document.getElementById('scrubber');
            if (scrubber && replayData.ohlc.length > 0) {
                scrubber.max = replayData.ohlc.length - 1;
                scrubber.value = replayData.ohlc.length - 1;  // Start at end
                replayIndex = replayData.ohlc.length - 1;
            }

            updateReplayStatus('Ready');
        }

        // Check replay mode (legacy)
        if (data.replay_mode) {
            replayMode = true;
            replayData = {
                ohlc: data.ohlc || [],
                trade_events: data.trade_events || [],
                trades: data.trades || [],
                equity: data.equity || []
            };
            startReplayMode();
        }
    });

    socket.on('price_update', (data) => {
        // For live updates - add new candle
        if (data.ohlc && candlestickSeries) {
            candlestickSeries.update(data.ohlc);
        }
        if (data.price) {
            updateCurrentPrice(data.price);
        }
    });

    socket.on('trade_closed', (data) => {
        if (data.trade) {
            addTradeToTable(data.trade);
            allTrades.push(data.trade);
            updateStatsFromTrade(data.trade);
        }
        if (data.stats) {
            updateStats(data.stats);
        }
    });

    // Handle chart data update (from training)
    socket.on('chart_data', (data) => {
        console.log('Received chart_data update');
        if (data.ohlc && data.ohlc.length > 0) {
            loadOHLCData(data.ohlc);
        }
    });

    // Handle equity updates (balance, max_drawdown, total_pnl)
    socket.on('equity_update', (data) => {
        if (data.balance !== undefined) {
            stats.currentBalance = data.balance;
        }
        if (data.max_drawdown !== undefined) {
            stats.maxDrawdown = data.max_drawdown;
        }
        if (data.total_pnl !== undefined) {
            stats.totalPnl = data.total_pnl;
        }
        updateStatsDisplay();
        updateEquityChart(stats.currentBalance);
    });

    // Handle full state refresh after training
    socket.on('refresh_state', (data) => {
        console.log('Received refresh_state:', data);

        // Reload OHLC
        if (data.ohlc && data.ohlc.length > 0) {
            loadOHLCData(data.ohlc);
        }

        // Reload trade events
        if (data.trade_events && data.trade_events.length > 0) {
            loadTradeEvents(data.trade_events);
        }

        // Reload trades
        if (data.trades && data.trades.length > 0) {
            // Clear existing trades table
            const tbody = document.getElementById('trades-tbody');
            if (tbody) {
                tbody.innerHTML = '';
            }
            allTrades = [];

            data.trades.forEach(trade => {
                addTradeToTable(trade);
                allTrades.push(trade);
            });
        }

        // Update equity chart
        if (data.equity && data.equity.length > 0) {
            stats.equityCurve = data.equity;
            if (equityChart) {
                equityChart.data.labels = data.equity.map((_, i) => i + 1);
                equityChart.data.datasets[0].data = data.equity;
                equityChart.update('none');
            }
        }

        // Update stats
        if (data.stats) {
            updateStats(data.stats);
        }
    });
}

// ============================================
// REPLAY MODE
// ============================================
function initializePlaybackControls() {
    const controls = document.getElementById('playback-controls');
    const playBtn = document.getElementById('play-pause-btn');
    const speedSelect = document.getElementById('speed-select');
    const scrubber = document.getElementById('scrubber');
    const backwardBtn = document.getElementById('backward-btn');
    const forwardBtn = document.getElementById('forward-btn');

    if (playBtn) {
        playBtn.addEventListener('click', togglePlayPause);
    }

    if (speedSelect) {
        speedSelect.addEventListener('change', (e) => {
            replaySpeed = parseInt(e.target.value);
            if (replayPlaying) {
                stopReplayInterval();
                startReplayInterval();
            }
        });
    }

    if (scrubber) {
        scrubber.addEventListener('input', (e) => {
            seekToPosition(parseInt(e.target.value));
        });
    }

    if (backwardBtn) {
        backwardBtn.addEventListener('click', () => seekRelative(-60));
    }

    if (forwardBtn) {
        forwardBtn.addEventListener('click', () => seekRelative(60));
    }
}

function startReplayMode() {
    const controls = document.getElementById('playback-controls');
    if (controls) {
        controls.style.display = 'flex';
    }

    if (!replayData || !replayData.ohlc || replayData.ohlc.length === 0) {
        console.log('No replay data available');
        return;
    }

    // Data is already loaded from initial_state, just set up playback position
    // Don't reset stats or clear chart - they already have the data
    replayIndex = replayData.ohlc.length - 1; // Start at end (all data loaded)

    // Setup scrubber
    const scrubber = document.getElementById('scrubber');
    if (scrubber) {
        scrubber.max = replayData.ohlc.length - 1;
        scrubber.value = replayIndex;
    }

    updateReplayStatus('Completed');
}

function togglePlayPause() {
    if (replayPlaying) {
        stopReplayInterval();
        updateReplayStatus('Paused');
    } else {
        startReplayInterval();
        updateReplayStatus('Playing');
    }

    const icon = document.getElementById('play-icon');
    if (icon) {
        icon.textContent = replayPlaying ? '⏸️' : '▶️';
    }
}

function startReplayInterval() {
    if (replayInterval) return;

    replayPlaying = true;
    const interval = Math.max(10, 100 / replaySpeed);

    replayInterval = setInterval(() => {
        if (replayIndex < replayData.ohlc.length) {
            playNextFrame();
        } else {
            stopReplayInterval();
            updateReplayStatus('Completed');
        }
    }, interval);
}

function stopReplayInterval() {
    replayPlaying = false;
    if (replayInterval) {
        clearInterval(replayInterval);
        replayInterval = null;
    }
}

function playNextFrame() {
    if (!replayData || replayIndex >= replayData.ohlc.length) return;

    const candle = replayData.ohlc[replayIndex];

    // Add candle to chart
    if (candlestickSeries) {
        candlestickSeries.update(candle);
    }

    // Update price display
    updateCurrentPrice(candle.close);

    // Check for trade events at this time
    const currentEvents = replayData.trade_events.filter(e => e.time === candle.time);
    currentEvents.forEach(event => {
        processTradeEvent(event);
    });

    // Update scrubber
    updateScrubber();

    replayIndex++;
}

function processTradeEvent(event) {
    // Find corresponding trade
    const trade = replayData.trades.find(t => {
        const entryEvent = replayData.trade_events.find(e => e.trade_id === event.trade_id && (e.type === 'BUY' || e.type === 'SELL'));
        return entryEvent && entryEvent.time === event.time;
    });

    if (event.type === 'BUY' || event.type === 'SELL') {
        // Entry event - will add marker on next update
    } else if (event.type.startsWith('CLOSE')) {
        // Exit event - update stats and add to table
        const fullTrade = replayData.trades.find(t => {
            const exitEvent = replayData.trade_events.find(e => e.trade_id === event.trade_id && e.type.startsWith('CLOSE'));
            return exitEvent && exitEvent.time === event.time;
        });

        if (fullTrade && !allTrades.includes(fullTrade)) {
            addTradeToTable(fullTrade);
            allTrades.push(fullTrade);
            updateStatsFromTrade(fullTrade);
            updateEquityChart(stats.currentBalance);
            updatePnlChart(fullTrade.pnl || event.pnl || 0);
        }
    }

    // Update markers
    const visibleEvents = replayData.trade_events.filter(e => e.time <= replayData.ohlc[replayIndex].time);
    const chartMarkers = visibleEvents.map(e => createMarkerFromEvent(e)).filter(m => m);
    if (candlestickSeries) {
        candlestickSeries.setMarkers(chartMarkers);
    }
}

function createMarkerFromEvent(event) {
    let color, shape, position, text;

    // Entry markers - Square shape
    if (event.type === 'BUY') {
        color = '#10b981';  // Green
        shape = 'square';
        position = 'belowBar';
        text = 'L';
    } else if (event.type === 'SELL') {
        color = '#ef4444';  // Red
        shape = 'square';
        position = 'aboveBar';
        text = 'S';
    } else if (event.type === 'CLOSE_LONG' || event.type === 'CLOSE_SHORT' || event.type.startsWith('CLOSE')) {
        // Exit markers - determine TP/SL/TO based on reason
        const reason = event.reason || '';

        if (reason === 'TAKE_PROFIT' || event.pnl > 0) {
            color = '#a855f7';  // Purple for TP
            text = 'TP';
            position = 'aboveBar';
        } else if (reason === 'STOP_LOSS' || event.pnl < 0) {
            color = '#06b6d4';  // Blue for SL
            text = 'SL';
            position = 'belowBar';
        } else if (reason === 'TIMEOUT') {
            color = '#eab308';  // Yellow for Timeout
            text = 'TO';
            position = 'aboveBar';
        } else {
            // Default based on P&L
            if (event.pnl > 0) {
                color = '#a855f7'; text = 'TP'; position = 'aboveBar';
            } else if (event.pnl < 0) {
                color = '#06b6d4'; text = 'SL'; position = 'belowBar';
            } else {
                color = '#eab308'; text = 'TO'; position = 'aboveBar';
            }
        }
        shape = 'square';
    } else {
        return null;
    }

    // Convert time to Unix timestamp if needed
    let time = event.time;
    if (typeof time === 'string') {
        time = Math.floor(new Date(time).getTime() / 1000);
    }

    return {
        time: time,
        position: position,
        color: color,
        shape: shape,
        text: text,
        size: 2,  // Larger size for visibility
        id: event.trade_id || null  // Store trade_id for tooltip lookup
    };
}

function seekToPosition(position) {
    if (!replayData || !replayData.ohlc) return;

    replayIndex = Math.min(position, replayData.ohlc.length - 1);

    // Rebuild chart up to this point
    const visibleCandles = replayData.ohlc.slice(0, replayIndex + 1);
    if (candlestickSeries) {
        candlestickSeries.setData(visibleCandles);
    }

    // Update markers
    const visibleEvents = replayData.trade_events.filter(e => e.time <= replayData.ohlc[replayIndex].time);
    const chartMarkers = visibleEvents.map(e => createMarkerFromEvent(e)).filter(m => m);
    if (candlestickSeries) {
        candlestickSeries.setMarkers(chartMarkers);
    }

    // Auto-scale chart to fit visible content (fixes blank chart during replay)
    if (tvChart) {
        tvChart.timeScale().fitContent();
    }

    updateScrubber();
}

function seekRelative(bars) {
    const newPosition = Math.max(0, Math.min(replayIndex + bars, replayData.ohlc.length - 1));
    seekToPosition(newPosition);
}

function updateScrubber() {
    const scrubber = document.getElementById('scrubber');
    const timeDisplay = document.getElementById('scrubber-time');

    if (scrubber) {
        scrubber.value = replayIndex;
    }

    if (timeDisplay && replayData && replayData.ohlc[replayIndex]) {
        const time = new Date(replayData.ohlc[replayIndex].time * 1000);
        timeDisplay.textContent = time.toLocaleTimeString();
    }
}

function updateReplayStatus(status) {
    const statusEl = document.getElementById('replay-status');
    if (statusEl) {
        statusEl.textContent = status;
        statusEl.className = 'replay-status ' + status.toLowerCase();
    }
}

function clearChart() {
    if (candlestickSeries) {
        candlestickSeries.setData([]);
        candlestickSeries.setMarkers([]);
    }
    allTrades = [];
    document.getElementById('trades-tbody').innerHTML = '<tr class="empty-row"><td colspan="7">Waiting for trades...</td></tr>';
}

// ============================================
// STATISTICS & DISPLAY
// ============================================
function updateStatsFromTrade(trade) {
    stats.totalTrades++;

    const tradeType = (trade.type || trade.direction || 'BUY').toUpperCase();
    if (tradeType === 'BUY' || tradeType === 'LONG') {
        stats.longTrades++;
    } else if (tradeType === 'SELL' || tradeType === 'SHORT') {
        stats.shortTrades++;
    }

    const pnl = trade.pnl || 0;
    if (pnl > 0) stats.winningTrades++;
    else stats.losingTrades++;

    stats.totalPnl += pnl;
    stats.currentBalance = stats.startingBalance + stats.totalPnl;

    if (stats.totalTrades > 0) {
        stats.winRate = (stats.winningTrades / stats.totalTrades) * 100;
    }

    updateStatsDisplay();
}

function updateStats(statsData) {
    stats.totalTrades = statsData.total_trades || 0;
    stats.winningTrades = statsData.winning_trades || 0;
    stats.losingTrades = statsData.losing_trades || 0;
    stats.winRate = statsData.win_rate || 0;
    stats.totalPnl = statsData.total_pnl || 0;
    stats.currentBalance = statsData.current_balance || 10000;
    stats.startingBalance = statsData.starting_balance || 10000;
    stats.maxDrawdown = statsData.max_drawdown || 0;
    stats.sharpeRatio = statsData.sharpe_ratio || 0;
    stats.longTrades = statsData.long_trades || 0;
    stats.shortTrades = statsData.short_trades || 0;

    updateStatsDisplay();
}

function updateStatsDisplay() {
    const setElement = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };

    setElement('stat-total-trades', stats.totalTrades);
    setElement('stat-long-trades', stats.longTrades);
    setElement('stat-short-trades', stats.shortTrades);
    setElement('stat-win-rate', stats.winRate.toFixed(1) + '%');
    setElement('stat-winning', stats.winningTrades);
    setElement('stat-losing', stats.losingTrades);
    setElement('stat-reward', formatCurrency(stats.totalPnl));
    setElement('stat-reward-pct', ((stats.totalPnl / stats.startingBalance) * 100).toFixed(2) + '%');
    setElement('stat-max-dd', stats.maxDrawdown.toFixed(2) + '%');
    setElement('stat-sharpe', stats.sharpeRatio.toFixed(2));
    setElement('stat-starting', formatCurrency(stats.startingBalance));
    setElement('stat-current-balance', formatCurrency(stats.currentBalance));

    // Update header displays
    setElement('current-balance', formatCurrency(stats.currentBalance));
    setElement('total-pnl', formatCurrency(stats.totalPnl));

    const pnlDisplay = document.getElementById('pnl-display');
    if (pnlDisplay) {
        pnlDisplay.className = 'pnl-display ' + (stats.totalPnl >= 0 ? 'positive' : 'negative');
    }
}

function resetStatsDisplay() {
    stats = {
        totalTrades: 0, longTrades: 0, shortTrades: 0,
        winningTrades: 0, losingTrades: 0, winRate: 0,
        totalPnl: 0, currentBalance: stats.startingBalance,
        startingBalance: stats.startingBalance, maxDrawdown: 0,
        sharpeRatio: 0, equityCurve: [], pnlHistory: []
    };
    updateStatsDisplay();
}

// ============================================
// TRADE TABLE
// ============================================
function addTradeToTable(trade) {
    const tbody = document.getElementById('trades-tbody');

    // Remove empty row
    const emptyRow = tbody.querySelector('.empty-row');
    if (emptyRow) emptyRow.remove();

    const row = document.createElement('tr');
    const tradeNum = trade.id || trade.trade_id || trade.tradeNumber || allTrades.length + 1;

    // Determine trade type and display text
    const tradeType = (trade.type || trade.direction || 'N/A').toUpperCase();
    const isLong = (tradeType === 'BUY' || tradeType === 'LONG');
    const typeDisplay = isLong ? 'LONG' : (tradeType === 'SELL' || tradeType === 'SHORT') ? 'SHORT' : tradeType;
    const typeClass = isLong ? 'type-buy' : 'type-sell';

    const pnl = trade.pnl || 0;
    const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
    const reason = trade.exit_reason || trade.reason || 'N/A';
    const reasonClass = reason === 'TAKE_PROFIT' ? 'reason-tp' :
        reason === 'STOP_LOSS' ? 'reason-sl' : 'reason-timeout';

    // Format entry and exit times
    const entryTime = formatTime(trade.entry_time);
    const exitTime = formatTime(trade.exit_time || trade.timestamp);

    row.innerHTML = `
        <td>${tradeNum}</td>
        <td>${entryTime}</td>
        <td>${exitTime}</td>
        <td class="${typeClass}">${typeDisplay}</td>
        <td>${formatCurrency(trade.entry_price)}</td>
        <td>${formatCurrency(trade.exit_price)}</td>
        <td class="${pnlClass}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</td>
        <td class="${reasonClass}">${formatReason(reason)}</td>
    `;

    // Click to navigate
    row.addEventListener('click', () => {
        const tradeWithId = { ...trade, trade_id: trade.trade_id || trade.id || `T${String(tradeNum).padStart(3, '0')}` };
        navigateToTrade(tradeWithId);
    });

    tbody.insertBefore(row, tbody.firstChild);

    // Update count
    const count = tbody.querySelectorAll('tr:not(.empty-row)').length;
    document.getElementById('trade-count').textContent = `${count} trades`;
}

// ============================================
// CSV DOWNLOAD
// ============================================
function initializeDownloadButton() {
    const btn = document.getElementById('download-csv-btn');
    if (btn) {
        btn.addEventListener('click', downloadTradesCSV);
    }
}

function downloadTradesCSV() {
    if (allTrades.length === 0) {
        alert('No trades to download');
        return;
    }

    const headers = ['#', 'Timestamp', 'Type', 'Entry Price', 'Exit Price', 'P&L', 'Reason'];
    const rows = allTrades.map((trade, idx) => [
        idx + 1,
        trade.timestamp || '',
        trade.type || trade.direction || '',
        trade.entry_price || 0,
        trade.exit_price || 0,
        trade.pnl || 0,
        trade.reason || ''
    ]);

    let csv = headers.join(',') + '\n';
    rows.forEach(row => {
        csv += row.map(v => `"${v}"`).join(',') + '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
}

// ============================================
// HELPER FUNCTIONS
// ============================================
function updateConnectionStatus(connected) {
    const dot = document.querySelector('.status-dot');
    const text = document.getElementById('status-text');

    if (connected) {
        dot?.classList.add('connected');
        dot?.classList.remove('disconnected');
        if (text) text.textContent = 'Connected';
    } else {
        dot?.classList.add('disconnected');
        dot?.classList.remove('connected');
        if (text) text.textContent = 'Disconnected';
    }
}

function updateCurrentPrice(price) {
    const priceEl = document.getElementById('current-price');
    if (priceEl) {
        priceEl.textContent = formatCurrency(price);
    }

    if (lastPrice !== null) {
        const change = price - lastPrice;
        const changePercent = (change / lastPrice) * 100;
        const changeEl = document.getElementById('price-change');

        if (changeEl) {
            if (change > 0) {
                changeEl.textContent = `+ ${changePercent.toFixed(2)}% `;
                changeEl.className = 'positive';
            } else if (change < 0) {
                changeEl.textContent = `${changePercent.toFixed(2)}% `;
                changeEl.className = 'negative';
            } else {
                changeEl.textContent = '0.00%';
                changeEl.className = 'neutral';
            }
        }
    }
    lastPrice = price;
}

function formatCurrency(value) {
    if (value == null) return '--';
    return '$' + parseFloat(value).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

function formatTime(timestamp) {
    if (!timestamp) return '--';
    const date = new Date(typeof timestamp === 'number' ? timestamp * 1000 : timestamp);
    if (isNaN(date.getTime())) return 'Invalid Date';

    const y = date.getFullYear();
    const m = String(date.getMonth() + 1).padStart(2, '0');
    const d = String(date.getDate()).padStart(2, '0');
    const H = String(date.getHours()).padStart(2, '0');
    const M = String(date.getMinutes()).padStart(2, '0');

    return `${y} /${m}/${d} ${H}:${M} `;
}

function formatReason(reason) {
    if (!reason) return '--';
    switch (reason) {
        case 'TAKE_PROFIT': return 'TP';
        case 'STOP_LOSS': return 'SL';
        case 'TIMEOUT': return 'Timeout';
        default: return reason;
    }
}

// ============================================
// TRAINING CONTROLS
// ============================================
let isTraining = false;

function initializeTrainingControls() {
    const startBtn = document.getElementById('start-train-btn');
    const saveBtn = document.getElementById('save-results-btn');

    if (startBtn) {
        startBtn.addEventListener('click', startTraining);
    }

    if (saveBtn) {
        saveBtn.addEventListener('click', saveResults);
    }

    // Socket events for training
    if (socket) {
        setupTrainingSocketEvents();
    } else {
        // Wait for socket to be initialized
        const checkSocket = setInterval(() => {
            if (socket) {
                setupTrainingSocketEvents();
                clearInterval(checkSocket);
            }
        }, 100);
    }
}

function setupTrainingSocketEvents() {
    socket.on('train_log', (data) => {
        appendTrainLog(data.message, data.type || 'info');
    });

    // Handle data reset for fresh training session
    socket.on('data_reset', (data) => {
        console.log('Data reset received:', data.message);

        // Reset all local data
        ohlcData = [];
        tradeEvents = [];
        allTrades = [];
        markers = [];
        stats = {
            totalTrades: 0, longTrades: 0, shortTrades: 0,
            winningTrades: 0, losingTrades: 0, winRate: 0,
            totalPnl: 0, currentBalance: 10000, startingBalance: 10000,
            maxDrawdown: 0, sharpeRatio: 0, equityCurve: [], pnlHistory: []
        };

        // Clear charts
        if (candlestickSeries) {
            candlestickSeries.setData([]);
            candlestickSeries.setMarkers([]);
        }
        if (equityChart) {
            equityChart.data.labels = [];
            equityChart.data.datasets[0].data = [];
            equityChart.update('none');
        }
        if (pnlChart) {
            pnlChart.data.labels = [];
            pnlChart.data.datasets[0].data = [];
            pnlChart.data.datasets[0].backgroundColor = [];
            pnlChart.update('none');
        }

        // Clear trade table
        const tbody = document.getElementById('trades-tbody');
        if (tbody) {
            tbody.innerHTML = '<tr class="empty-row"><td colspan="8">No trades yet</td></tr>';
        }
        document.getElementById('trade-count').textContent = '0 trades';

        // Reset balance display
        document.getElementById('current-balance').textContent = '$10,000.00';
        document.getElementById('total-pnl').textContent = '$0.00';
        document.getElementById('pnl-display').classList.remove('positive', 'negative');

        // Update stats display
        updateStatsDisplay();

        appendTrainLog('Previous data cleared. Starting fresh training session...', 'info');
    });

    socket.on('train_complete', (data) => {
        isTraining = false;
        updateTrainStatus('completed', 'Completed');

        const startBtn = document.getElementById('start-train-btn');
        const saveBtn = document.getElementById('save-results-btn');

        if (startBtn) {
            startBtn.disabled = false;
            startBtn.textContent = 'Start Train/Test';
            startBtn.classList.remove('running');
        }

        if (saveBtn) {
            saveBtn.style.display = 'inline-block';
        }

        appendTrainLog('='.repeat(50), 'info');
        appendTrainLog('Training and backtest completed successfully!', 'success');
        if (data.csv_path) {
            appendTrainLog(`Train history saved: ${data.csv_path}`, 'result');
        }
        if (data.screenshot_path) {
            appendTrainLog(`Screenshot saved: ${data.screenshot_path}`, 'result');
        }
    });

    socket.on('train_error', (data) => {
        isTraining = false;
        updateTrainStatus('error', 'Error');

        const startBtn = document.getElementById('start-train-btn');
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.textContent = 'Start Train/Test';
            startBtn.classList.remove('running');
        }

        appendTrainLog(`Error: ${data.message}`, 'error');
    });
}

function startTraining() {
    if (isTraining) return;

    isTraining = true;

    // Clear previous logs
    const logContainer = document.getElementById('training-log');
    if (logContainer) {
        logContainer.innerHTML = '';
    }

    // Update UI
    const startBtn = document.getElementById('start-train-btn');
    const saveBtn = document.getElementById('save-results-btn');

    if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = 'Training...';
        startBtn.classList.add('running');
    }

    if (saveBtn) {
        saveBtn.style.display = 'none';
    }

    updateTrainStatus('running', 'Training...');
    appendTrainLog('Starting training process...', 'step');

    // Request training via socket
    socket.emit('start_training');
}

function appendTrainLog(message, type = 'info') {
    const logContainer = document.getElementById('training-log');
    if (!logContainer) return;

    // Remove placeholder if exists
    const placeholder = logContainer.querySelector('.log-placeholder');
    if (placeholder) {
        placeholder.remove();
    }

    const logLine = document.createElement('div');
    logLine.className = `log-line ${type}`;

    const timestamp = new Date().toLocaleTimeString();
    logLine.innerHTML = `<span class="log-timestamp">[${timestamp}]</span> ${message}`;

    logContainer.appendChild(logLine);

    // Auto-scroll to bottom
    logContainer.scrollTop = logContainer.scrollHeight;
}

function updateTrainStatus(statusClass, statusText) {
    const statusEl = document.getElementById('train-status');
    if (statusEl) {
        statusEl.className = `train-status ${statusClass}`;
        statusEl.textContent = statusText;
    }
}

function saveResults() {
    appendTrainLog('Saving results...', 'step');
    socket.emit('save_results');
}

// Screenshot functionality using html2canvas (if available)
async function captureScreenshot() {
    try {
        // Check if html2canvas is available
        if (typeof html2canvas === 'undefined') {
            console.log('html2canvas not available, requesting server-side screenshot');
            socket.emit('capture_screenshot');
            return;
        }

        const dashboard = document.querySelector('.dashboard-container');
        const canvas = await html2canvas(dashboard, {
            backgroundColor: '#0a0e17',
            scale: 1
        });

        // Convert to blob and send to server
        canvas.toBlob((blob) => {
            const formData = new FormData();
            formData.append('screenshot', blob, 'dashboard.png');

            fetch('/api/save-screenshot', {
                method: 'POST',
                body: formData
            }).then(response => response.json())
                .then(data => {
                    if (data.success) {
                        appendTrainLog(`Screenshot saved: ${data.path}`, 'result');
                    }
                });
        });
    } catch (error) {
        console.error('Screenshot error:', error);
        socket.emit('capture_screenshot');
    }
}
