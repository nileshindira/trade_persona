document.addEventListener("DOMContentLoaded", function () {
    if (!window.report || !window.report.web_data) return;
    const charts = window.report.web_data.charts || {};

    /* -------------------------
       P&L TIMELINE CHART
       ------------------------- */
    if (charts.pnl_timeline) {
        const ctx = document.getElementById("pnlChart");
        new Chart(ctx, {
            type: "line",
            data: {
                labels: charts.pnl_timeline.dates,
                datasets: [{
                    label: "Net P&L",
                    data: charts.pnl_timeline.values,
                    borderWidth: 2,
                    tension: 0.3,
                    borderColor: "#0d6efd",
                    backgroundColor: "rgba(13,110,253,0.15)",
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: false }
                }
            }
        });
    }

    /* -------------------------
       INSTRUMENT DISTRIBUTION
       ------------------------- */
    if (charts.instrument_distribution) {
        const ctx = document.getElementById("instrumentChart");
        new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: charts.instrument_distribution.labels,
                datasets: [{
                    data: charts.instrument_distribution.values,
                    backgroundColor: [
                        "#0d6efd", "#198754", "#ffc107", "#dc3545", "#6f42c1"
                    ]
                }]
            }
        });
    }

    if (charts.segment_distribution) {
        const ctx = document.getElementById("instrumentChart");
        new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: charts.instrument_distribution.labels,
                datasets: [{
                    data: charts.instrument_distribution.values,
                    backgroundColor: [
                        "#0d6efd", "#198754", "#ffc107", "#dc3545", "#6f42c1"
                    ]
                }]
            }
        });
    }
});
