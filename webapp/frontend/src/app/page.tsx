"use client";

import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  TrendingUp, TrendingDown, Activity,
  BarChart3, AlertCircle, CheckCircle,
  Target, Zap, Clock, PieChart,
  Shield, User, ChevronDown
} from "lucide-react";
import dynamic from 'next/dynamic';

const Radar = dynamic(() => import('react-chartjs-2').then(mod => mod.Radar), { ssr: false });
const Line = dynamic(() => import('react-chartjs-2').then(mod => mod.Line), { ssr: false });
const Doughnut = dynamic(() => import('react-chartjs-2').then(mod => mod.Doughnut), { ssr: false });

import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
} from 'chart.js';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement
);

export default function AnalysisPage() {
  const [tradeFile, setTradeFile] = useState<File | null>(null);
  const [pnlFile, setPnlFile] = useState<File | null>(null);
  const [traderName, setTraderName] = useState("Trader");
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [partialMetrics, setPartialMetrics] = useState<any>(null);
  const [currentStage, setCurrentStage] = useState<string>("init");
  const [progress, setProgress] = useState(0);
  const [journeySteps, setJourneySteps] = useState<any[]>([
    { id: 'loading', label: 'Data Ingestion & Pairing', status: 'pending' },
    { id: 'auditing', label: 'Clinical Performance Audit', status: 'pending' },
    { id: 'patterns', label: 'Behavioral Pattern Discovery', status: 'pending' },
    { id: 'ai_analysis', label: 'AI Persona Orchestration', status: 'pending' },
    { id: 'finalizing', label: 'Report Synthesis', status: 'pending' },
  ]);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!tradeFile) return;

    setLoading(true);
    setError(null);
    setPartialMetrics(null);
    setProgress(0);
    setJourneySteps(steps => steps.map(s => ({ ...s, status: 'pending' })));

    const formData = new FormData();
    formData.append("trade_file", tradeFile);
    if (pnlFile) formData.append("pnl_file", pnlFile);
    formData.append("trader_name", traderName);

    try {
      const response = await fetch("http://127.0.0.1:8100/analyze-stream", {
        method: 'POST',
        body: formData,
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const update = JSON.parse(line);

            if (update.type === 'status') {
              setCurrentStage(update.stage);
              setProgress(update.progress);
              setJourneySteps(prev => prev.map(s => {
                if (s.id === update.stage) return { ...s, status: 'working' };
                const stageIndex = prev.findIndex(st => st.id === update.stage);
                const stepIndex = prev.findIndex(st => st.id === s.id);
                if (stepIndex < stageIndex) return { ...s, status: 'done' };
                return s;
              }));
            } else if (update.type === 'data') {
              if (update.metrics) {
                setPartialMetrics(update.metrics);
              }
            } else if (update.type === 'complete') {
              setReport(update.report);
            } else if (update.type === 'error') {
              setError(update.message);
              setLoading(false);
              return;
            }
          } catch (e) {
            console.error("Failed to parse stream line", e);
          }
        }
      }
    } catch (err: any) {
      setError(err.message || "Something went wrong during analysis.");
    } finally {
      setLoading(false);
    }
  };

  if (report) {
    return <Dashboard report={report} onReset={() => setReport(null)} />;
  }

  if (loading) {
    return (
      <div className="container d-flex flex-column align-items-center justify-content-center" style={{ minHeight: "90vh" }}>
        <div className="card shadow-lg dark-card p-5" style={{ maxWidth: "600px", width: "100%" }}>
          <div className="text-center mb-5">
            <h2 className="font-outfit fw-bold text-white mb-2">Preparing Persona</h2>
            <p className="text-muted uppercase-tracking smallest fw-bold opacity-75">Clinical Diagnostic Journey</p>
          </div>

          {partialMetrics && (
            <div className="row g-3 mb-5 fade-in">
              <div className="col-4">
                <div className="p-3 bg-dark rounded-3 border border-secondary border-opacity-25 text-center">
                  <span className="text-muted smallest uppercase-tracking d-block mb-1">PnL</span>
                  <span className={`h5 mb-0 fw-bold ${partialMetrics.net_pnl >= 0 ? "text-success" : "text-danger"}`}>
                    ₹{new Intl.NumberFormat('en-IN').format(partialMetrics.net_pnl || partialMetrics.total_pnl || 0)}
                  </span>
                </div>
              </div>
              <div className="col-4">
                <div className="p-3 bg-dark rounded-3 border border-secondary border-opacity-25 text-center">
                  <span className="text-muted smallest uppercase-tracking d-block mb-1">Win Rate</span>
                  <span className="h5 mb-0 fw-bold text-white">{(partialMetrics.win_rate || partialMetrics.win_rate_pct || 0).toFixed(1)}%</span>
                </div>
              </div>
              <div className="col-4">
                <div className="p-3 bg-dark rounded-3 border border-secondary border-opacity-25 text-center">
                  <span className="text-muted smallest uppercase-tracking d-block mb-1">Trades</span>
                  <span className="h5 mb-0 fw-bold text-primary">{partialMetrics.total_trades || 0}</span>
                </div>
              </div>
            </div>
          )}

          <div className="journey-steps mb-4">
            {journeySteps.map((step, idx) => (
              <div key={step.id} className={`d-flex align-items-center mb-3 transition-opacity ${step.status === 'pending' ? 'opacity-25' : 'opacity-100'}`}>
                <div className={`rounded-circle d-flex align-items-center justify-content-center me-3 ${step.status === 'done' ? 'bg-success text-white' :
                  step.status === 'working' ? 'bg-primary text-white spinner-border-sm' :
                    'bg-secondary text-white'
                  }`} style={{ width: "24px", height: "24px", fontSize: "11px" }}>
                  {step.status === 'done' ? <CheckCircle size={14} /> : step.status === 'working' ? <div className="spinner-border spinner-border-sm" style={{ width: "12px", height: "12px" }}></div> : idx + 1}
                </div>
                <div className="flex-grow-1">
                  <div className={`fw-bold small ${step.status === 'working' ? 'text-primary' : 'text-white'}`}>{step.label}</div>
                  {step.status === 'working' && <div className="smallest text-muted opacity-75">Analyzing tape data...</div>}
                </div>
              </div>
            ))}
          </div>

          <div className="progress bg-dark border border-secondary border-opacity-10 mt-4" style={{ height: "4px" }}>
            <div
              className="progress-bar progress-bar-striped progress-bar-animated bg-primary"
              style={{ width: `${progress}%`, transition: 'width 0.8s ease' }}
            ></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container d-flex align-items-center justify-content-center" style={{ minHeight: "90vh" }}>
      <div className="card shadow-lg dark-card" style={{ maxWidth: "550px", width: "100%" }}>
        <div className="card-header bg-primary text-white text-center py-4 border-0 rounded-top">
          <h2 className="mb-0 font-outfit fw-bold uppercase-tracking">Stockk Persona Analyzer</h2>
          <p className="small mb-0 opacity-75 mt-1">Trading & Behavioral Auditor</p>
        </div>
        <div className="card-body p-5">
          <form onSubmit={handleUpload}>
            <div className="mb-4">
              <label className="form-label text-muted small uppercase-tracking fw-bold">Trader Name</label>
              <input
                type="text"
                className="form-control bg-dark border-secondary text-white py-3"
                value={traderName}
                onChange={(e) => setTraderName(e.target.value)}
                placeholder="e.g. Nilesh"
                required
              />
            </div>
            <div className="mb-4">
              <label className="form-label text-muted small uppercase-tracking fw-bold">Trade Journal (Required)</label>
              <input
                type="file"
                className="form-control bg-dark border-secondary text-white py-3"
                accept=".csv, .xlsx, .xls"
                onChange={(e) => setTradeFile(e.target.files?.[0] || null)}
                required
              />
            </div>
            <div className="mb-4">
              <label className="form-label text-muted small uppercase-tracking fw-bold">PnL Statement (Optional)</label>
              <input
                type="file"
                className="form-control bg-dark border-secondary text-white py-3 border-opacity-25"
                accept=".csv, .xlsx, .xls"
                onChange={(e) => setPnlFile(e.target.files?.[0] || null)}
              />
              <span className="smallest text-muted mt-1 d-block opacity-50">Providing both improves FIFO reconciliation accuracy.</span>
            </div>
            <button
              type="submit"
              className="btn btn-primary w-100 py-3 fw-bold shadow-sm mt-3 font-outfit"
              disabled={loading}
            >
              GENERATE PERSONA REPORT
            </button>
          </form>
          {error && <div className="alert alert-danger mt-4 small border-0 bg-danger bg-opacity-10 text-danger">{error}</div>}

          <div className="mt-5 text-center opacity-50 small">
            <p className="mb-0">Supported formats: Dhan, Zerodha, Fyers, Upstox</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function Dashboard({ report, onReset }: { report: any, onReset: () => void }) {
  // Use hero.topline or web_data.kpis as fallback for executive summary
  const es = report.hero?.topline || report.executive_summary || report.web_data?.kpis || {};
  const dm = report.web_data?.kpis || report.detailed_metrics || {};
  const charts = report.web_data?.charts || { pnl_timeline: { dates: [], values: [] }, instrument_distribution: [] };

  // Construct persona scores for radar chart from KPIs if missing
  const personaScores = report.web_data?.persona_scores || {
    discipline: dm.discipline_score,
    emotional_control: dm.emotional_control,
    risk_appetite: dm.risk_appetite,
    efficiency: dm.efficiency_ratio,
    consistency: (report.consistency_score || 0) / 100
  };

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="dashboard-container fade-in">
      {/* Hero Section - Mirroring the ag-hero style */}
      <div className="ag-hero p-5 rounded-4 mb-4 border border-secondary border-opacity-25" style={{ background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)', position: 'relative' }}>
        <div className="d-flex justify-content-between align-items-center mb-3">
          <span className="text-primary small uppercase-tracking fw-bold">Surgical Performance Audit &mdash; Stockk Diagnostic Engine</span>
          <div className="d-flex gap-2">
            <button className="btn btn-sm btn-outline-primary px-3 fw-bold" onClick={() => window.open(`http://127.0.0.1:8100/reports/${report.hero?.trader_name || "Trader"}/${report.hero?.trader_name || "Trader"}_report.html`, '_blank')}>
              OPEN SURGICAL REPORT
            </button>
            <button className="btn btn-sm btn-outline-secondary opacity-50" onClick={onReset}>New Audit</button>
          </div>
        </div>
        <h1 className="display-4 font-outfit fw-800 text-white mb-2">{report.hero?.trader_name || report.trader_name || "Trader"}</h1>
        <div className="d-flex align-items-center gap-2 mb-3">
          <span className="fs-3">{report.archetype?.icon || "🧠"}</span>
          <span className="h4 font-outfit text-primary mb-0">{report.hero?.persona_name || report.archetype?.name || "The Market Participant"}</span>
        </div>
        <p className="lead text-muted fst-italic mb-4">&ldquo;{report.hero?.headline || "Data-driven insights for professional development."}&rdquo;</p>

        <div className="row g-4 mt-2">
          <ToplineItem label="Net Realized P&L" value={es.net_pnl ?? es.pnl_total ?? 0} isCurrency={true} />
          <ToplineItem label="Win Rate" value={es.win_rate ?? es.win_rate_pct ?? 0} isPct={true} />
          <ToplineItem label="Profit Factor" value={es.profit_factor ?? 0} />
          <ToplineItem label="Max Drawdown" value={es.max_drawdown_pct ?? 0} isPct={true} isRed={true} />
          <ToplineItem label="Consistency" value={report.consistency_score || 0} isScore={true} />
        </div>
      </div>

      <nav className="sticky-top bg-dark bg-opacity-75 py-3 mb-4 border-bottom border-secondary border-opacity-25 backdrop-blur">
        <ul className="nav nav-pills justify-content-center gap-3">
          <li className="nav-item"><a className="nav-link text-white small uppercase-tracking active" href="#diagnosis">Diagnosis</a></li>
          <li className="nav-item"><a className="nav-link text-muted small uppercase-tracking" href="#metrics">Metrics</a></li>
          <li className="nav-item"><a className="nav-link text-muted small uppercase-tracking" href="#proposed">Strategy</a></li>
          <li className="nav-item"><a className="nav-link text-muted small uppercase-tracking" href="#roadmap">Roadmap</a></li>
          <li className="nav-item"><a className="nav-link text-muted small uppercase-tracking" href="#evidence">Evidence</a></li>
        </ul>
      </nav>

      <div className="row g-4 mb-5" id="diagnosis">
        {/* Diagnostic Verdict */}
        <div className="col-lg-7">
          <div className="card h-100 dark-card p-4 border-start border-primary border-4">
            <div className="d-flex justify-content-between align-items-center mb-4">
              <h3 className="font-outfit fw-bold text-white mb-0">Diagnostic Verdict</h3>
              <span className="badge bg-primary px-3 py-2 text-white">{report.diagnosis?.trader_type || "Standard"}</span>
            </div>
            <div className="text-muted mb-4 lead" dangerouslySetInnerHTML={{ __html: report.diagnosis?.narrative || report.analysis_text?.trader_profile }} />

            <div className="row g-3">
              <div className="col-6">
                <div className="p-3 bg-dark rounded-3 border border-secondary border-opacity-25">
                  <span className="text-muted smallest uppercase-tracking d-block mb-1">Strategy Style</span>
                  <span className="text-white fw-bold" dangerouslySetInnerHTML={{ __html: report.diagnosis?.inferred_style || "N/A" }} />
                </div>
              </div>
              <div className="col-6">
                <div className="p-3 bg-dark rounded-3 border border-secondary border-opacity-25">
                  <span className="text-muted smallest uppercase-tracking d-block mb-1">Natural Edge</span>
                  <span className="text-success fw-bold">{report.diagnosis?.natural_edge || "Win Concentration"}</span>
                </div>
              </div>
              <div className="col-6">
                <div className="p-3 bg-dark rounded-3 border border-secondary border-opacity-25">
                  <span className="text-muted smallest uppercase-tracking d-block mb-1">Risk Profile</span>
                  <span className="text-warning fw-bold" dangerouslySetInnerHTML={{ __html: report.diagnosis?.risk_profile?.appetite || "Moderate" }} />
                </div>
              </div>
              <div className="col-6">
                <div className="p-3 bg-dark rounded-3 border border-secondary border-opacity-25">
                  <span className="text-muted smallest uppercase-tracking d-block mb-1">Data Confidence</span>
                  <span className="text-primary fw-bold">High (Verified)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Behavioral Signature Radar */}
        <div className="col-lg-5">
          <div className="card h-100 dark-card p-4">
            <h4 className="font-outfit text-white mb-4 d-flex align-items-center">
              <Activity size={20} className="me-2 text-primary" /> Behavioral Signature
            </h4>
            <div style={{ height: "280px" }}>
              <Radar
                data={{
                  labels: Object.keys(personaScores).map(s => s.replace('_score', '').replace('_', ' ').toUpperCase()),
                  datasets: [{
                    label: 'Trader Attributes',
                    data: Object.values(personaScores).map(v => (typeof v === 'number' ? v : 0) * 100),
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#3b82f6',
                  }]
                }}
                options={{
                  scales: {
                    r: {
                      beginAtZero: true,
                      max: 100,
                      angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                      grid: { color: 'rgba(255, 255, 255, 0.1)' },
                      pointLabels: { color: '#94a3b8', font: { size: 10 } },
                      ticks: { display: false }
                    }
                  },
                  plugins: { legend: { display: false } },
                  maintainAspectRatio: false
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Section */}
      <div id="metrics" className="mb-5">
        <h2 className="ag-section-title h3 font-outfit text-white mb-4 border-bottom border-secondary border-opacity-25 pb-3">Analytical Deep Dive</h2>
        <div className="row g-4 mb-4">
          <div className="col-lg-8">
            <div className="card dark-card p-4 h-100">
              <h5 className="text-white mb-4">Relative Portfolio Scale (Benchmarked to Nifty)</h5>
              <div style={{ height: "300px" }}>
                <Line
                  data={{
                    labels: report.web_data?.charts?.relative_chart?.dates || charts.pnl_timeline.dates,
                    datasets: [
                      {
                        label: 'Trader P&L (Scaled 100)',
                        data: report.web_data?.charts?.relative_chart?.pnl_normalized || charts.pnl_timeline.values,
                        borderColor: '#3b82f6',
                        tension: 0.3,
                        fill: true,
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        pointRadius: 0
                      },
                      {
                        label: 'Nifty Benchmark (Scaled 100)',
                        data: report.web_data?.charts?.relative_chart?.nifty_normalized || charts.pnl_timeline.benchmark_values,
                        borderColor: 'rgba(148, 163, 184, 0.5)',
                        borderDash: [5, 5],
                        pointRadius: 0,
                        borderWidth: 1.5
                      }
                    ]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                      x: { display: false },
                      y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    },
                    plugins: { legend: { position: 'top', align: 'end', labels: { boxWidth: 10, color: '#94a3b8' } } }
                  }}
                />
              </div>
            </div>
          </div>
          <div className="col-lg-4">
            <div className="card dark-card p-4 h-100">
              <h5 className="text-white mb-4">Sector Exposure</h5>
              <div style={{ height: "300px" }}>
                <Doughnut
                  data={{
                    labels: charts.instrument_distribution.map((d: any) => d.asset_kind || d.label),
                    datasets: [{
                      data: charts.instrument_distribution.map((d: any) => d.value),
                      backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'],
                      borderWidth: 0,
                    }]
                  }}
                  options={{
                    cutout: '70%',
                    plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, color: '#94a3b8' } } },
                    maintainAspectRatio: false
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Proposed Trades Section */}
      <div id="proposed" className="mb-5">
        <h2 className="ag-section-title h3 font-outfit text-white mb-4 border-bottom border-secondary border-opacity-25 pb-3">Strategic Trade Analysis</h2>
        <div className="card dark-card p-4">
          <div className="d-flex justify-content-between align-items-center mb-4">
            <h5 className="text-white mb-0">Proposed vs Realized Analysis</h5>
            <button className="btn btn-sm btn-outline-success px-3" onClick={() => window.open(`http://127.0.0.1:8100/reports/${report.hero?.trader_name}/${report.hero?.trader_name}_consolidated_trace.csv`, '_blank')}>
              DOWNLOAD CONSOLIDATED TRACE (CSV)
            </button>
          </div>
          <div className="table-responsive" style={{ maxHeight: "400px" }}>
            <table className="table table-dark table-hover smallest align-middle">
              <thead>
                <tr>
                  <th className="text-muted uppercase-tracking">S.No</th>
                  <th className="text-muted uppercase-tracking">Symbol</th>
                  <th className="text-muted uppercase-tracking">Trade Date</th>
                  <th className="text-muted uppercase-tracking">Type</th>
                  <th className="text-muted uppercase-tracking">Price</th>
                  <th className="text-muted uppercase-tracking">PnL</th>
                  <th className="text-muted uppercase-tracking">Market Behavior</th>
                  <th className="text-muted uppercase-tracking">T-Score</th>
                </tr>
              </thead>
              <tbody>
                {(report.appendix?.proposed_trades || []).map((trade: any, idx: number) => (
                  <tr key={idx} className="border-secondary border-opacity-10">
                    <td className="text-muted">{idx + 1}</td>
                    <td className="fw-bold text-white">{trade.symbol}</td>
                    <td className="text-muted">{trade.trade_date}</td>
                    <td>
                      <span className={`badge ${trade.transaction_type === "BUY" ? "bg-success" : "bg-danger"} bg-opacity-10 ${trade.transaction_type === "BUY" ? "text-success" : "text-danger"}`}>
                        {trade.transaction_type}
                      </span>
                    </td>
                    <td className="text-white">₹{trade.price}</td>
                    <td className={`fw-bold ${trade.pnl >= 0 ? "text-success" : "text-danger"}`}>
                      ₹{(trade.pnl || 0).toLocaleString('en-IN')}
                    </td>
                    <td className="text-muted italic">{trade.market_behaviour || "N/A"}</td>
                    <td>
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress flex-grow-1" style={{ height: "4px", width: "40px", backgroundColor: "#334155" }}>
                          <div className="progress-bar bg-primary" style={{ width: `${trade.t_score * 10}%` }}></div>
                        </div>
                        <span className="text-primary fw-bold" style={{ fontSize: "10px" }}>{trade.t_score}</span>
                      </div>
                    </td>
                  </tr>
                ))}
                {(!report.appendix?.proposed_trades || report.appendix.proposed_trades.length === 0) && (
                  <tr>
                    <td colSpan={8} className="text-center py-5 text-muted fst-italic">No proposed trades found in TRADE_FILES/Proposed_Trade_File.csv</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Roadmap Section */}
      <div id="roadmap" className="mb-5">
        <div className="card dark-card p-5 border-top border-warning border-4">
          <div className="text-center mb-5">
            <h2 className="font-outfit fw-bold text-white mb-2">Improvement Roadmap</h2>
            <p className="text-muted">A prioritized sequence of actions to transform your performance</p>
          </div>
          <div className="row g-4">
            <div className="col-md-4">
              <div className="p-4 bg-dark bg-opacity-50 rounded-4 border border-secondary border-opacity-10 h-100">
                <h5 className="text-warning mb-4 d-flex align-items-center"><Zap size={18} className="me-2" /> Next 5 Sessions</h5>
                <ul className="list-unstyled">
                  {(report.improvement_plan?.next_5_sessions || ["Refine entries", "Lower size"]).map((item: string, i: number) => (
                    <li key={i} className="mb-3 d-flex gap-2 text-muted small">
                      <div className="text-warning"><CheckCircle size={14} /></div>
                      <div dangerouslySetInnerHTML={{ __html: item }} />
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="col-md-4">
              <div className="p-4 bg-dark bg-opacity-50 rounded-4 border border-secondary border-opacity-10 h-100">
                <h5 className="text-primary mb-4 d-flex align-items-center"><Clock size={18} className="me-2" /> Next 30 Days</h5>
                <ul className="list-unstyled">
                  {(report.improvement_plan?.next_30_days || ["Systematize logging", "Emotional journaling"]).map((item: string, i: number) => (
                    <li key={i} className="mb-3 d-flex gap-2 text-muted small">
                      <div className="text-primary"><CheckCircle size={14} /></div>
                      <div dangerouslySetInnerHTML={{ __html: item }} />
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="col-md-4">
              <div className="p-4 bg-primary bg-opacity-10 rounded-4 border border-primary border-opacity-20 h-100 d-flex flex-column">
                <span className="text-info smallest uppercase-tracking mb-2">The Biggest Lever</span>
                <h4 className="text-white font-outfit mb-4" dangerouslySetInnerHTML={{ __html: report.improvement_plan?.biggest_lever || "Emotional Regulation" }} />
                <div className="mt-auto p-3 bg-dark rounded-3 border border-secondary">
                  <span className="text-muted smallest uppercase-tracking d-block mb-1">Final Verdict</span>
                  <span className="text-success fw-bold small" dangerouslySetInnerHTML={{ __html: report.final_verdict || "Highly Promising" }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Evidence Section */}
      <div id="evidence" className="mb-5">
        <h2 className="ag-section-title h3 font-outfit text-white mb-4 border-bottom border-secondary border-opacity-25 pb-3">Evidence Ledger</h2>
        <div className="card dark-card p-4">
          <div className="table-responsive">
            <table className="table table-dark table-hover align-middle">
              <thead>
                <tr>
                  <th className="text-muted small uppercase-tracking" style={{ width: "40%" }}>Diagnostic Claim</th>
                  <th className="text-muted small uppercase-tracking" style={{ width: "20%" }}>Confidence</th>
                  <th className="text-muted small uppercase-tracking">Supporting Proof</th>
                </tr>
              </thead>
              <tbody>
                {(report.evidence_ledger || []).map((e: any, i: number) => (
                  <tr key={i} className="border-secondary border-opacity-10">
                    <td className="fw-bold text-white py-3">{e.claim}</td>
                    <td>
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress flex-grow-1" style={{ height: "4px" }}>
                          <div className="progress-bar bg-primary" style={{ width: `${(e.confidence || 0) * 10}%` }}></div>
                        </div>
                        <span className="text-muted small">{e.confidence}/10</span>
                      </div>
                    </td>
                    <td>
                      <div className="d-flex flex-wrap gap-1">
                        {(e.supporting_metrics || []).map((m: string, mi: number) => (
                          <span key={mi} className="badge bg-dark border border-secondary border-opacity-25 text-muted smaller">{m}</span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
                {(!report.evidence_ledger || report.evidence_ledger.length === 0) && (
                  <tr><td colSpan={3} className="text-center py-5 text-muted">Evidence ledger will be populated from AI analysis.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <footer className="py-5 text-center text-muted small opacity-50">
        <p className="mb-1">&copy; 2026 Stockk Diagnostic Engine &mdash; All analyses are AI-generated.</p>
        <p className="mb-0">Confidential Trading Analytics. Financial decisions remain the responsibility of the trader.</p>
      </footer>
    </div>
  );
}

function ToplineItem({ label, value, isCurrency, isPct, isScore, isRed }: any) {
  let displayValue = value;
  if (isCurrency) displayValue = "₹" + new Intl.NumberFormat('en-IN').format(value);
  else if (isPct) displayValue = `${Math.abs(value).toFixed(1)}%`;
  else if (isScore) displayValue = `${value}/100`;

  let textColor = "text-white";
  if (isCurrency) textColor = value >= 0 ? "text-success" : "text-danger";
  if (isRed) textColor = "text-danger";

  return (
    <div className="col-6 col-md-2">
      <div className="border-start border-secondary ps-3">
        <span className="text-muted smallest uppercase-tracking d-block mb-1">{label}</span>
        <div className={`h4 font-outfit fw-bold mb-0 ${textColor}`}>{displayValue}</div>
      </div>
    </div>
  );
}
