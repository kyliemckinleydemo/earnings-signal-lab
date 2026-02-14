import { useState, useCallback } from "react";

const FM = {
  mgmt_hedging: { name: "Management Hedging Language", cat: "Management Behavior", color: "#e74c3c", bear: true, desc: "Hedging words vs definitive language" },
  mgmt_deflection: { name: "Q&A Deflection Rate", cat: "Management Behavior", color: "#c0392b", bear: true, desc: "Non-answers and redirects in Q&A" },
  mgmt_specificity: { name: "Guidance Specificity", cat: "Management Behavior", color: "#e67e22", bear: false, desc: "Specific numbers vs vague qualitative" },
  mgmt_confidence_shift: { name: "Confidence Shift (Prepared→Q&A)", cat: "Management Behavior", color: "#d35400", bear: true, desc: "Confidence drop from script to live" },
  analyst_skepticism: { name: "Analyst Skepticism", cat: "Analyst Behavior", color: "#2980b9", bear: true, desc: "Challenging follow-ups and doubt" },
  analyst_surprise: { name: "Analyst Surprise Indicators", cat: "Analyst Behavior", color: "#3498db", bear: false, desc: "Unexpected information reactions" },
  analyst_focus_cluster: { name: "Analyst Question Clustering", cat: "Analyst Behavior", color: "#1abc9c", bear: false, desc: "Multiple analysts on same topic" },
  guidance_revision_dir: { name: "Guidance Revision Direction", cat: "Forward Guidance", color: "#9b59b6", bear: false, desc: "Raised vs lowered vs maintained" },
  guidance_qualifiers: { name: "Qualifier Density", cat: "Forward Guidance", color: "#8e44ad", bear: true, desc: "Conditional language density" },
  new_risk_mention: { name: "New Risk Factor Mentions", cat: "Risk Signals", color: "#e74c3c", bear: true, desc: "First-time risk factors" },
  macro_blame: { name: "External Blame Attribution", cat: "Risk Signals", color: "#c0392b", bear: true, desc: "Blaming macro vs taking ownership" },
  capex_language: { name: "CapEx/Investment Tone", cat: "Strategic Signals", color: "#27ae60", bear: false, desc: "Aggressive expansion vs cautious" },
  hiring_language: { name: "Hiring & Headcount Signals", cat: "Strategic Signals", color: "#2ecc71", bear: false, desc: "Growth hiring vs restructuring" },
  competitive_mentions: { name: "Competitive Positioning", cat: "Strategic Signals", color: "#16a085", bear: true, desc: "Concerned vs dismissive" },
  customer_language: { name: "Customer/Demand Descriptors", cat: "Demand Signals", color: "#f39c12", bear: false, desc: "Robust/record vs softening" },
  pricing_power: { name: "Pricing Power Indicators", cat: "Demand Signals", color: "#f1c40f", bear: false, desc: "Raising prices vs discounting" },
};

const CATS = [...new Set(Object.values(FM).map(f => f.cat))];
const PDS = ["1D", "5D", "10D", "21D"];

function icColor(ic) { const a = Math.abs(ic); return a > 0.15 ? "#4ade80" : a > 0.08 ? "#fbbf24" : "#6b7280"; }

function FeatureCard({ fid, data, sel, onClick }) {
  const m = FM[fid]; if (!m) return null;
  let best = null;
  if (data) {
    Object.entries(data).forEach(([p, s]) => {
      if (!best || Math.abs(s.ic || 0) > Math.abs(best.ic || 0)) best = { ...s, period: p };
    });
  }
  return (
    <div onClick={onClick} style={{ background: sel ? "#1a1a2e" : "#0d0d1a", border: `1px solid ${sel ? m.color : "#1a1a2e"}`, borderRadius: "8px", padding: "12px", cursor: "pointer", transition: "all 0.15s" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: "12px", fontWeight: 600, color: m.color }}>{m.name}</div>
          <div style={{ fontSize: "10px", color: "#6b7280", marginTop: "1px" }}>{m.desc}</div>
          {m.bear && <span style={{ fontSize: "9px", color: "#f87171", background: "rgba(248,113,113,0.1)", padding: "1px 5px", borderRadius: "3px", marginTop: "3px", display: "inline-block" }}>bearish when high</span>}
        </div>
        {best && (
          <div style={{ textAlign: "right", marginLeft: "10px", flexShrink: 0 }}>
            <div style={{ fontSize: "16px", fontWeight: 700, color: icColor(best.ic || 0) }}>
              {(best.ic || 0) > 0 ? "+" : ""}{(best.ic || 0).toFixed(3)}
            </div>
            <div style={{ fontSize: "9px", color: "#6b7280" }}>IC @ {best.period}</div>
          </div>
        )}
      </div>
      {best && (
        <div style={{ display: "flex", gap: "10px", marginTop: "6px", fontSize: "10px", color: "#9ca3af" }}>
          <span>Acc: {((best.accuracy || 0) * 100).toFixed(0)}%</span>
          <span>Win: {((best.win_rate || 0) * 100).toFixed(0)}%</span>
          <span>p={best.ic_pvalue?.toFixed(3)}</span>
          <span>n={best.n_observations}</span>
        </div>
      )}
    </div>
  );
}

function DetailView({ fid, data, samples }) {
  const m = FM[fid]; if (!m || !data) return null;
  return (
    <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "10px", padding: "20px" }}>
      <div style={{ fontSize: "15px", fontWeight: 700, color: m.color, marginBottom: "2px" }}>{m.name}</div>
      <div style={{ fontSize: "11px", color: "#9ca3af", marginBottom: "16px" }}>{m.desc}</div>

      <div style={{ fontSize: "12px", fontWeight: 600, color: "#e5e7eb", marginBottom: "8px" }}>Signal by Holding Period</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px", marginBottom: "20px" }}>
        {PDS.map(p => {
          const s = data[p];
          if (!s) return <div key={p} style={{ background: "#111127", borderRadius: "6px", padding: "10px", textAlign: "center", opacity: 0.4 }}><div style={{ fontSize: "10px", color: "#6b7280" }}>{p}</div></div>;
          return (
            <div key={p} style={{ background: "#111127", borderRadius: "6px", padding: "10px", textAlign: "center" }}>
              <div style={{ fontSize: "10px", color: "#6b7280", marginBottom: "4px" }}>{p}</div>
              <div style={{ fontSize: "20px", fontWeight: 700, color: icColor(s.ic || 0) }}>{(s.ic || 0) > 0 ? "+" : ""}{(s.ic || 0).toFixed(3)}</div>
              <div style={{ fontSize: "9px", color: "#6b7280" }}>Information Coefficient</div>
              <div style={{ marginTop: "6px", fontSize: "10px", color: "#9ca3af", lineHeight: 1.5 }}>
                <div>Accuracy: {((s.accuracy || 0) * 100).toFixed(1)}%</div>
                <div>Win Rate: {((s.win_rate || 0) * 100).toFixed(1)}%</div>
                <div>Avg Return: {(s.avg_return_pct || 0) > 0 ? "+" : ""}{(s.avg_return_pct || 0).toFixed(2)}%</div>
                <div>Sharpe: {(s.sharpe || 0).toFixed(2)}</div>
                <div>p-value: {(s.ic_pvalue || 0).toFixed(4)}</div>
                <div>n = {s.n_observations} ({s.n_signal_triggered} triggered)</div>
              </div>
            </div>
          );
        })}
      </div>

      {samples && samples.length > 0 && (
        <div>
          <div style={{ fontSize: "12px", fontWeight: 600, color: "#e5e7eb", marginBottom: "8px" }}>Sample Extractions (Claude Analysis)</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {samples.map((s, i) => {
              const scoreColor = s.score > 0.6 ? (m.bear ? "#f87171" : "#4ade80") : s.score < 0.4 ? (m.bear ? "#4ade80" : "#f87171") : "#fbbf24";
              return (
                <div key={i} style={{ background: "#111127", borderRadius: "6px", padding: "10px", borderLeft: `3px solid ${scoreColor}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px", flexWrap: "wrap", gap: "4px" }}>
                    <span style={{ fontSize: "12px", fontWeight: 600, color: "#e5e7eb" }}>
                      {s.symbol} <span style={{ color: "#6b7280", fontWeight: 400 }}>{s.quarter}</span>
                    </span>
                    <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                      <span style={{ fontSize: "10px", padding: "1px 6px", borderRadius: "3px", background: "rgba(255,255,255,0.05)", color: "#9ca3af" }}>
                        Score: {s.score}
                      </span>
                      {s.return_5D != null && (
                        <span style={{ fontSize: "11px", fontWeight: 600, color: s.return_5D > 0 ? "#4ade80" : "#f87171" }}>
                          5D: {s.return_5D > 0 ? "+" : ""}{Number(s.return_5D).toFixed(1)}%
                        </span>
                      )}
                      {s.return_10D != null && (
                        <span style={{ fontSize: "11px", fontWeight: 600, color: s.return_10D > 0 ? "#4ade80" : "#f87171" }}>
                          10D: {s.return_10D > 0 ? "+" : ""}{Number(s.return_10D).toFixed(1)}%
                        </span>
                      )}
                    </div>
                  </div>
                  {s.evidence && <div style={{ fontSize: "11px", color: "#9ca3af", lineHeight: 1.3 }}>{s.evidence}</div>}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function CombosView({ combos }) {
  if (!combos || combos.length === 0) return <div style={{ padding: "40px", textAlign: "center", color: "#6b7280" }}>No combination data available.</div>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
      <div style={{ fontSize: "13px", fontWeight: 600, color: "#e5e7eb" }}>Multi-Feature Combinations</div>
      <div style={{ fontSize: "11px", color: "#6b7280", marginBottom: "6px" }}>Combining uncorrelated features for stronger composite signals</div>
      {[...combos].sort((a, b) => Math.abs(b.sharpe) - Math.abs(a.sharpe)).map((c, i) => (
        <div key={i} style={{ background: "#0d0d1a", borderRadius: "8px", padding: "14px", borderLeft: `3px solid ${Math.abs(c.sharpe) > 1 ? "#4ade80" : Math.abs(c.sharpe) > 0.5 ? "#fbbf24" : "#60a5fa"}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div>
              <div style={{ fontSize: "13px", fontWeight: 600, color: "#e5e7eb" }}>{c.name}</div>
              <div style={{ display: "flex", gap: "4px", marginTop: "6px", flexWrap: "wrap" }}>
                {(c.features || []).map(f => {
                  const mt = FM[f];
                  return <span key={f} style={{ fontSize: "9px", padding: "1px 6px", borderRadius: "3px", background: `${mt?.color || "#666"}22`, color: mt?.color || "#999" }}>{mt?.name?.split(" ").slice(0, 2).join(" ") || f}</span>;
                })}
              </div>
            </div>
            <div style={{ textAlign: "right", flexShrink: 0, marginLeft: "12px" }}>
              <div style={{ fontSize: "20px", fontWeight: 700, color: Math.abs(c.sharpe) > 1 ? "#4ade80" : "#fbbf24" }}>{(c.sharpe || 0).toFixed(2)}</div>
              <div style={{ fontSize: "9px", color: "#6b7280" }}>Sharpe @ {c.period}</div>
            </div>
          </div>
          <div style={{ display: "flex", gap: "12px", marginTop: "8px", fontSize: "10px", color: "#9ca3af" }}>
            <span>Avg Return: {(c.avg_return || 0) > 0 ? "+" : ""}{(c.avg_return || 0).toFixed(2)}%</span>
            <span>Win Rate: {((c.win_rate || 0) * 100).toFixed(0)}%</span>
            <span>Triggered: {c.n_triggered}/{c.n_total}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function CorrView({ matrix }) {
  if (!matrix || Object.keys(matrix).length === 0) return <div style={{ padding: "40px", textAlign: "center", color: "#6b7280" }}>No correlation data available.</div>;
  const fs = Object.keys(matrix).filter(f => FM[f]).slice(0, 10);
  return (
    <div>
      <div style={{ fontSize: "13px", fontWeight: 600, color: "#e5e7eb", marginBottom: "4px" }}>Feature Correlation Matrix</div>
      <div style={{ fontSize: "11px", color: "#6b7280", marginBottom: "12px" }}>High correlation = redundant signal. Combine uncorrelated features for alpha.</div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={{ padding: "4px" }}></th>
              {fs.map(f => <th key={f} style={{ padding: "3px", fontSize: "8px", color: FM[f]?.color, writingMode: "vertical-lr", transform: "rotate(180deg)", height: "70px", textAlign: "left", fontWeight: 500 }}>{FM[f]?.name?.split(" ").slice(0, 2).join(" ")}</th>)}
            </tr>
          </thead>
          <tbody>
            {fs.map(f1 => (
              <tr key={f1}>
                <td style={{ padding: "3px 6px 3px 0", fontSize: "9px", color: FM[f1]?.color, whiteSpace: "nowrap", fontWeight: 500 }}>{FM[f1]?.name?.split(" ").slice(0, 2).join(" ")}</td>
                {fs.map(f2 => {
                  const v = matrix[f1]?.[f2] ?? 0;
                  const a = Math.abs(v);
                  const bg = f1 === f2 ? "#2d2d4a" : a > 0.5 ? `rgba(239,68,68,${a * 0.5})` : a > 0.3 ? `rgba(251,191,36,${a * 0.4})` : a > 0.1 ? `rgba(74,222,128,${a * 0.3})` : "rgba(255,255,255,0.02)";
                  return <td key={f2} style={{ padding: "2px" }}><div style={{ background: bg, borderRadius: "2px", padding: "4px 2px", fontSize: "9px", color: f1 === f2 ? "#6b7280" : "#e5e7eb", textAlign: "center", fontWeight: a > 0.3 ? 600 : 400 }}>{f1 === f2 ? "—" : v.toFixed(2)}</div></td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function makeDemo() {
  const rng = (s) => { let x = Math.sin(s) * 10000; return x - Math.floor(x); };
  const features = {};
  const samples = {};
  Object.keys(FM).forEach((fid, fi) => {
    const pd = {};
    PDS.forEach((p, pi) => {
      const s = fi * 100 + pi * 10 + 42;
      const baseIC = (rng(s) - 0.45) * 0.35;
      const ic = FM[fid].bear ? -Math.abs(baseIC) - 0.02 : Math.abs(baseIC) + 0.02;
      pd[p] = {
        ic: Math.round(ic * 1000) / 1000,
        ic_pvalue: Math.round(Math.max(0.001, 0.15 - Math.abs(ic) * 0.8 + rng(s + 1) * 0.1) * 1000) / 1000,
        accuracy: Math.round((0.5 + Math.abs(ic) * 0.8 + (rng(s + 2) - 0.5) * 0.06) * 1000) / 1000,
        sharpe: Math.round((ic * 6 + (rng(s + 3) - 0.5) * 0.4) * 100) / 100,
        avg_return_pct: Math.round((ic * 3 + (rng(s + 4) - 0.5) * 0.5) * 100) / 100,
        win_rate: Math.round((0.5 + ic * 0.6 + (rng(s + 5) - 0.5) * 0.06) * 1000) / 1000,
        n_observations: Math.floor(80 + rng(s + 6) * 120),
        n_signal_triggered: Math.floor(20 + rng(s + 7) * 40),
      };
    });
    features[fid] = pd;
    samples[fid] = [
      { symbol: "NVDA", quarter: "Q3 2024", score: 0.85, evidence: "Strong signal in prepared remarks and Q&A", return_5D: Math.round((rng(fi * 10 + 1) - 0.4) * 800) / 100, return_10D: Math.round((rng(fi * 10 + 2) - 0.4) * 1200) / 100 },
      { symbol: "AAPL", quarter: "Q4 2024", score: 0.35, evidence: "Moderate signal with mixed evidence across sections", return_5D: Math.round((rng(fi * 10 + 3) - 0.5) * 600) / 100, return_10D: Math.round((rng(fi * 10 + 4) - 0.5) * 1000) / 100 },
      { symbol: "JPM", quarter: "Q2 2024", score: 0.62, evidence: "Notable language patterns in Q&A section", return_5D: Math.round((rng(fi * 10 + 5) - 0.5) * 500) / 100, return_10D: Math.round((rng(fi * 10 + 6) - 0.45) * 800) / 100 },
    ];
  });
  const corr = {};
  Object.keys(FM).forEach(f1 => { corr[f1] = {}; Object.keys(FM).forEach(f2 => { if (f1 === f2) corr[f1][f2] = 1; else { const s = f1.length * f2.length + f1.charCodeAt(0) * f2.charCodeAt(0); corr[f1][f2] = Math.round((rng(s) * 0.6 - 0.1) * 100) / 100; } }); });
  return {
    metadata: { total_events: 187, companies: ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "GS", "JNJ", "WMT", "XOM"], date_range: ["2023-01-15", "2025-01-20"], generated_at: "DEMO DATA" },
    features, sample_extractions: samples, correlation_matrix: corr,
    combinations: [
      { name: "Confidence Gap + Analyst Skepticism", features: ["mgmt_confidence_shift", "analyst_skepticism"], period: "5D", sharpe: 1.18, avg_return: 2.4, win_rate: 0.68, n_triggered: 24, n_total: 187 },
      { name: "Guidance Raise + Low Hedging + Strong Demand", features: ["guidance_revision_dir", "mgmt_hedging", "customer_language"], period: "10D", sharpe: 1.35, avg_return: 3.1, win_rate: 0.71, n_triggered: 18, n_total: 187 },
      { name: "Deflection + Question Clustering", features: ["mgmt_deflection", "analyst_focus_cluster"], period: "5D", sharpe: 0.94, avg_return: 1.8, win_rate: 0.63, n_triggered: 31, n_total: 187 },
      { name: "New Risks + Blame + Qualifiers", features: ["new_risk_mention", "macro_blame", "guidance_qualifiers"], period: "10D", sharpe: 1.02, avg_return: 2.2, win_rate: 0.65, n_triggered: 22, n_total: 187 },
      { name: "Pricing Power + CapEx", features: ["pricing_power", "capex_language"], period: "21D", sharpe: 0.78, avg_return: 1.5, win_rate: 0.61, n_triggered: 28, n_total: 187 },
    ],
  };
}

export default function EarningsSignalLab() {
  const [results, setResults] = useState(null);
  const [sel, setSel] = useState(null);
  const [tab, setTab] = useState("features");
  const [catFilter, setCatFilter] = useState("All");
  const [err, setErr] = useState(null);

  const onUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErr(null);
    const reader = new FileReader();
    reader.onload = (evt) => {
      try { setResults(JSON.parse(evt.target.result)); }
      catch { setErr("Invalid JSON file. Upload backtest_results.json from the pipeline."); }
    };
    reader.readAsText(file);
  }, []);

  const fids = Object.keys(FM);
  const filtered = catFilter === "All" ? fids : fids.filter(f => FM[f].cat === catFilter);
  const sorted = results ? [...filtered].sort((a, b) => {
    const aD = results.features?.[a]; const bD = results.features?.[b];
    if (!aD && !bD) return 0; if (!aD) return 1; if (!bD) return -1;
    const aMax = Math.max(...Object.values(aD).map(p => Math.abs(p.ic || 0)));
    const bMax = Math.max(...Object.values(bD).map(p => Math.abs(p.ic || 0)));
    return bMax - aMax;
  }) : filtered;

  const isDemo = results?.metadata?.generated_at === "DEMO DATA";

  return (
    <div style={{ background: "#070714", minHeight: "100vh", color: "#e5e7eb", fontFamily: "'JetBrains Mono','SF Mono','Fira Code',monospace" }}>
      {/* Header */}
      <div style={{ background: "linear-gradient(135deg,#0d0d1a,#111127)", borderBottom: "1px solid #1a1a2e", padding: "16px 20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
          <div>
            <div style={{ fontSize: "18px", fontWeight: 700, letterSpacing: "-0.5px" }}>
              <span style={{ color: "#4ade80" }}>EARNINGS</span>
              <span style={{ color: "#6b7280" }}> SIGNAL </span>
              <span style={{ color: "#60a5fa" }}>LAB</span>
              {isDemo && <span style={{ fontSize: "10px", color: "#f87171", marginLeft: "8px", fontWeight: 400 }}>DEMO</span>}
            </div>
            {results?.metadata && (
              <div style={{ fontSize: "10px", color: "#6b7280", marginTop: "2px" }}>
                {results.metadata.total_events} earnings events · {results.metadata.companies?.length} companies · {results.metadata.date_range?.[0]} to {results.metadata.date_range?.[1]}
              </div>
            )}
          </div>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <button onClick={() => setResults(makeDemo())} style={{ background: "transparent", border: "1px solid #1a1a2e", color: "#6b7280", borderRadius: "6px", padding: "8px 14px", fontSize: "11px", cursor: "pointer", fontFamily: "inherit" }}>
              DEMO DATA
            </button>
            <label style={{ background: "linear-gradient(135deg,#4ade80,#22c55e)", color: "#070714", borderRadius: "6px", padding: "8px 14px", fontSize: "11px", fontWeight: 700, cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.5px" }}>
              LOAD RESULTS
              <input type="file" accept=".json" onChange={onUpload} style={{ display: "none" }} />
            </label>
          </div>
        </div>
        {err && <div style={{ color: "#f87171", fontSize: "11px", marginTop: "6px" }}>{err}</div>}
      </div>

      {/* Empty state */}
      {!results && (
        <div style={{ padding: "60px 20px", textAlign: "center" }}>
          <div style={{ fontSize: "40px", marginBottom: "12px" }}>📊</div>
          <div style={{ fontSize: "16px", fontWeight: 600, color: "#e5e7eb", marginBottom: "8px" }}>Earnings Transcript Signal Tester</div>
          <div style={{ fontSize: "12px", color: "#6b7280", maxWidth: "520px", margin: "0 auto", lineHeight: 1.7 }}>
            Run <code style={{ background: "#1a1a2e", padding: "2px 6px", borderRadius: "3px" }}>python earnings_signal_pipeline.py</code> to pull real transcripts, extract 16 features with Claude, and backtest against real price data.
            Then upload <code style={{ background: "#1a1a2e", padding: "2px 6px", borderRadius: "3px" }}>backtest_results.json</code> here.
          </div>
          <div style={{ marginTop: "24px", background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "10px", padding: "16px", maxWidth: "520px", margin: "24px auto 0", textAlign: "left" }}>
            <div style={{ fontSize: "11px", fontWeight: 600, color: "#4ade80", marginBottom: "8px" }}>Pipeline Steps</div>
            <div style={{ fontSize: "11px", color: "#9ca3af", lineHeight: 1.8 }}>
              1. FMP API → Pull earnings transcripts (free tier, includes Q&A)<br />
              2. Claude API → Extract 16 granular NLP features per transcript<br />
              3. Yahoo Finance → Get post-earnings price data (1D/5D/10D/21D)<br />
              4. Backtest → IC, accuracy, Sharpe, p-values, correlations
            </div>
          </div>
        </div>
      )}

      {/* Tabs + Content */}
      {results && (
        <>
          <div style={{ display: "flex", borderBottom: "1px solid #1a1a2e", padding: "0 20px" }}>
            {[["features", "Individual Features"], ["combinations", "Feature Combos"], ["correlations", "Correlations"]].map(([key, label]) => (
              <button key={key} onClick={() => setTab(key)} style={{
                background: "none", border: "none",
                borderBottom: tab === key ? "2px solid #4ade80" : "2px solid transparent",
                color: tab === key ? "#e5e7eb" : "#6b7280",
                padding: "10px 16px", fontSize: "11px", fontWeight: 600, cursor: "pointer",
                fontFamily: "inherit", textTransform: "uppercase", letterSpacing: "0.5px",
              }}>{label}</button>
            ))}
          </div>

          <div style={{ padding: "16px 20px" }}>
            {tab === "features" && (
              <>
                <div style={{ display: "flex", gap: "5px", marginBottom: "12px", flexWrap: "wrap" }}>
                  {["All", ...CATS].map(cat => (
                    <button key={cat} onClick={() => setCatFilter(cat)} style={{
                      background: catFilter === cat ? "#1a1a2e" : "transparent",
                      border: `1px solid ${catFilter === cat ? "#4ade80" : "#1a1a2e"}`,
                      color: catFilter === cat ? "#4ade80" : "#6b7280",
                      borderRadius: "5px", padding: "4px 10px", fontSize: "10px", cursor: "pointer", fontFamily: "inherit",
                    }}>{cat}</button>
                  ))}
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: "14px" }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px", maxHeight: "80vh", overflowY: "auto" }}>
                    {sorted.map(fid => (
                      <FeatureCard key={fid} fid={fid} data={results.features?.[fid]} sel={sel === fid} onClick={() => setSel(fid)} />
                    ))}
                  </div>
                  <div>
                    {sel && results.features?.[sel] ? (
                      <DetailView fid={sel} data={results.features[sel]} samples={results.sample_extractions?.[sel]} />
                    ) : (
                      <div style={{ background: "#0d0d1a", border: "1px solid #1a1a2e", borderRadius: "10px", padding: "40px", textAlign: "center", color: "#6b7280" }}>
                        <div style={{ fontSize: "28px", marginBottom: "6px" }}>←</div>
                        <div style={{ fontSize: "12px" }}>Select a feature to view detailed backtest results</div>
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
            {tab === "combinations" && <CombosView combos={results.combinations} />}
            {tab === "correlations" && <CorrView matrix={results.correlation_matrix} />}
          </div>
        </>
      )}
    </div>
  );
}
