"use client";

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useState, useEffect, useRef } from "react";

// ─── Data ────────────────────────────────────────────────────────────────────

const METRICS = [
  { label: "Predictions Today", value: 12847, suffix: "", icon: "⚡", prefix: "" },
  { label: "Accuracy Rate", value: 87.4, suffix: "%", icon: "🎯", prefix: "" },
  { label: "Customers Retained", value: 3210, suffix: "+", icon: "🔒", prefix: "" },
  { label: "Revenue Saved", value: 2.4, suffix: "M", icon: "💰", prefix: "$" },
];

const FEATURES = [
  {
    title: "Real-Time Scoring",
    desc: "Get instant churn probability scores for individual customers the moment you need them.",
    icon: "▶",
    color: "#00FFB2",
    badge: "",
  },
  {
    title: "Batch Analysis",
    desc: "Upload entire customer segments and receive ranked risk reports within seconds.",
    icon: "◈",
    color: "#FF5C5C",
    badge: "Coming Soon",
  },
  {
    title: "Feature Insights",
    desc: "Understand which factors — tenure, usage, billing — drive each churn prediction.",
    icon: "◉",
    color: "#FFCC00",
    badge: "",
  },
  {
    title: "API Access",
    desc: "Plug directly into your CRM or data pipeline with our clean FastAPI endpoint.",
    icon: "⬡",
    color: "#5C8FFF",
    badge: "",
  },
];

const RECENT = [
  { id: "CX-4821", score: 87, risk: "HIGH", delta: "+12%" },
  { id: "CX-1093", score: 23, risk: "LOW", delta: "-5%" },
  { id: "CX-7742", score: 61, risk: "MED", delta: "+3%" },
  { id: "CX-2255", score: 91, risk: "HIGH", delta: "+18%" },
  { id: "CX-5580", score: 34, risk: "LOW", delta: "-2%" },
];

// ─── Sub-components ──────────────────────────────────────────────────────────

function AnimatedCounter({
  end,
  suffix = "",
  prefix = "",
}: {
  end: number;
  suffix?: string;
  prefix?: string;
}) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const started = useRef(false);

  useEffect(() => {
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting || started.current) return;
        started.current = true;
        let elapsed = 0;
        const duration = 1800;
        const timer = setInterval(() => {
          elapsed += 16;
          const progress = Math.min(elapsed / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          const val = end * eased;
          setCount(end % 1 !== 0 ? parseFloat(val.toFixed(1)) : Math.floor(val));
          if (progress >= 1) { setCount(end); clearInterval(timer); }
        }, 16);
      },
      { threshold: 0.3 }
    );
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, [end]);

  return (
    <span ref={ref}>
      {prefix}
      {end % 1 !== 0 ? count.toFixed(1) : Math.floor(count)}
      {suffix}
    </span>
  );
}

function RiskBadge({ risk }: { risk: string }) {
  const styles: Record<string, string> = {
    HIGH: "bg-red-500/10 text-red-400 border-red-500/30",
    MED: "bg-yellow-400/10 text-yellow-300 border-yellow-400/30",
    LOW: "bg-emerald-400/10 text-emerald-400 border-emerald-400/30",
    MEDIUM: "bg-yellow-400/10 text-yellow-300 border-yellow-400/30",
  };
  return (
    <span className={`inline-block px-2.5 py-0.5 rounded text-[11px] font-bold tracking-widest border font-mono ${styles[risk] ?? ""}`}>
      {risk}
    </span>
  );
}

function ScoreBar({ score }: { score: number }) {
  const color = score > 70 ? "#FF5C5C" : score > 45 ? "#FFCC00" : "#00FFB2";
  return (
    <div className="flex items-center gap-2.5">
      <div className="w-20 h-1.5 rounded-full overflow-hidden bg-[#1a1a2e]">
        <div
          className="h-full rounded-full transition-all duration-1000"
          style={{ width: `${score}%`, background: color, boxShadow: `0 0 8px ${color}88` }}
        />
      </div>
      <span className="font-mono text-[13px] font-bold" style={{ color }}>{score}%</span>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function ChurnHome() {
  const router = useRouter();

  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [predInput, setPredInput] = useState("");
  const [predResult, setPredResult] = useState<{ score: number; risk: string } | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const handle = (e: MouseEvent) => setMousePos({ x: e.clientX, y: e.clientY });
    window.addEventListener("mousemove", handle);
    return () => window.removeEventListener("mousemove", handle);
  }, []);

  const handlePredict = () => {
    if (!predInput.trim()) return;
    setLoading(true);
    setPredResult(null);
    setTimeout(() => {
      const score = Math.floor(Math.random() * 100);
      setPredResult({ score, risk: score > 70 ? "HIGH" : score > 45 ? "MED" : "LOW" });
      setLoading(false);
    }, 1400);
  };

  const goToPredict = () => router.push("/predict");

  return (
    <div className="min-h-screen bg-[#070714] text-[#E8E8F0] overflow-x-hidden relative">

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;0,9..40,900&display=swap');
        * { font-family: 'DM Sans', sans-serif; }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInLeft {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes glowPulse {
          0%,100% { box-shadow: 0 0 24px rgba(0,255,178,0.3); }
          50%      { box-shadow: 0 0 52px rgba(0,255,178,0.6); }
        }
        @keyframes scanline {
          from { top: -10%; } to { top: 110%; }
        }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes spin   { to { transform: rotate(360deg); } }

        .anim-hero   { animation: fadeInLeft .7s ease both; }
        .anim-widget { animation: fadeInLeft .7s .15s ease both; }
        .anim-result { animation: fadeIn .4s ease; }
        .glow-btn    { animation: glowPulse 2.4s ease infinite; }
        .blink       { animation: blink 1.8s ease infinite; }
        .spinner     { animation: spin .7s linear infinite; }
        .scan        {
          position: absolute; left: 0; right: 0; height: 50px;
          background: linear-gradient(transparent, rgba(0,255,178,0.025), transparent);
          animation: scanline 4s linear infinite; pointer-events: none;
        }

        ::-webkit-scrollbar       { width: 5px; }
        ::-webkit-scrollbar-track { background: #070714; }
        ::-webkit-scrollbar-thumb { background: #1a1a2e; border-radius: 3px; }
        ::selection               { background: rgba(0,255,178,0.25); }
      `}</style>

      {/* Ambient cursor glow */}
      <div
        className="fixed rounded-full pointer-events-none z-0"
        style={{
          left: mousePos.x - 300,
          top: mousePos.y - 300,
          width: 600,
          height: 600,
          background: "radial-gradient(circle, rgba(0,255,178,0.05) 0%, transparent 70%)",
          transition: "left .25s ease, top .25s ease",
        }}
      />

      {/* Grid bg */}
      <div
        className="fixed inset-0 z-0 pointer-events-none opacity-[0.045]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(0,255,178,0.6) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,178,0.6) 1px, transparent 1px)",
          backgroundSize: "56px 56px",
        }}
      />

      {/* ══════ NAV ══════ */}
      {/* px-6 on mobile, px-12 on desktop */}
      <nav className="fixed top-0 left-0 right-0 z-50 h-16 px-6 md:px-12 flex items-center justify-between bg-[rgba(7,7,20,0.82)] backdrop-blur-xl border-b border-white/[0.05]">
        <div className="flex items-center gap-2.5">
          <div className="relative w-8 h-8 flex-shrink-0">
            <Image
              src="/StayFlow.png"
              alt="StayFlow logo"
              fill
              className="object-contain rounded-lg"
              priority
            />
          </div>
          <span className="font-bold text-[17px] tracking-tight text-white">
            StayFlow<span className="text-[#00FFB2]">.</span>
          </span>
        </div>

        <button
          onClick={goToPredict}
          className="glow-btn bg-[#00FFB2] text-[#070714] font-bold text-[13px] px-5 py-2 rounded-lg tracking-wide hover:scale-105 active:scale-95 transition-transform"
        >
          Launch App →
        </button>
      </nav>

      {/* ══════ CONTENT ══════ */}
      <div className="relative z-10">

        {/* ── HERO ──
            mobile:  single column, tighter padding
            desktop: two columns, original padding                          */}
        <section className="max-w-[1100px] mx-auto px-6 md:px-12 pt-32 md:pt-40 pb-16 md:pb-24 grid grid-cols-1 md:grid-cols-2 gap-10 md:gap-20 items-center">

          {/* Left */}
          <div className="anim-hero">
            <div className="inline-flex items-center gap-2 bg-[rgba(0,255,178,0.08)] border border-[rgba(0,255,178,0.2)] rounded-full px-4 py-1.5 mb-7">
              <span className="blink w-1.5 h-1.5 rounded-full bg-[#00FFB2] shadow-[0_0_8px_#00FFB2]" />
              <span className="text-[11px] text-[#00FFB2] font-bold tracking-widest">ML MODEL LIVE · 94.3% ACCURACY</span>
            </div>

            <h1 className="text-[clamp(40px,5vw,64px)] font-black leading-[1.04] tracking-[-2.5px] text-white mb-6">
              Predict Churn<br />
              <span className="bg-gradient-to-r from-[#00FFB2] to-[#00BFFF] bg-clip-text text-transparent">
                Before It Happens.
              </span>
            </h1>

            <p className="text-[18px] text-[#666] leading-[1.75] mb-10 max-w-[440px]">
              Harness machine learning to identify customers at risk of churning —
              so your team can act first, retain more, and grow smarter.
            </p>

            <div className="flex gap-3 flex-wrap">
              <button
                onClick={goToPredict}
                className="glow-btn bg-[#00FFB2] text-[#070714] font-bold text-[15px] px-7 py-3.5 rounded-xl hover:scale-105 active:scale-95 transition-transform"
              >
                Run a Prediction →
              </button>
            </div>
          </div>

          {/* Right — quick predict widget */}
          <div className="anim-widget relative bg-gradient-to-br from-white/[0.04] to-white/[0.01] border border-white/[0.08] rounded-2xl p-8 backdrop-blur-md overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-[#00FFB2] to-transparent opacity-60" />
            <div className="scan" />

            <div className="text-[10px] text-[#3a3a3a] font-bold tracking-[1.8px] mb-5 font-mono uppercase">
              Quick Predict // Demo
            </div>

            <div className="mb-4">
              <label className="block text-[13px] text-[#555] mb-2">Customer ID or Email</label>
              <input
                value={predInput}
                onChange={e => setPredInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handlePredict()}
                placeholder="e.g. CX-4821 or user@email.com"
                className="w-full bg-white/[0.04] border border-white/10 rounded-xl px-4 py-3 text-[14px] text-[#E8E8F0] outline-none focus:border-[rgba(0,255,178,0.4)] transition-colors placeholder:text-[#2d2d2d]"
              />
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full py-3.5 rounded-xl font-bold text-[14px] text-[#070714] transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 hover:opacity-90"
              style={{ background: loading ? "rgba(0,255,178,0.35)" : "#00FFB2" }}
            >
              {loading && <span className="spinner w-4 h-4 border-2 border-[#070714] border-t-transparent rounded-full" />}
              {loading ? "Analyzing…" : "Predict Churn Risk"}
            </button>

            {predResult && (
              <div
                className="anim-result mt-5 p-5 rounded-xl border"
                style={{
                  background: predResult.score > 70 ? "rgba(255,92,92,0.08)" : "rgba(0,255,178,0.07)",
                  borderColor: predResult.score > 70 ? "rgba(255,92,92,0.25)" : "rgba(0,255,178,0.2)",
                }}
              >
                <div className="flex justify-between items-center mb-3">
                  <span className="text-[11px] text-[#555] font-mono tracking-widest">CHURN PROBABILITY</span>
                  <RiskBadge risk={predResult.risk} />
                </div>
                <ScoreBar score={predResult.score} />
                <p className="text-[12px] text-[#555] mt-3 leading-relaxed">
                  {predResult.score > 70
                    ? "⚠️ High risk — immediate intervention recommended"
                    : predResult.score > 45
                      ? "⚡ Moderate risk — monitor and engage proactively"
                      : "✅ Low risk — customer appears stable"}
                </p>
                <button
                  onClick={goToPredict}
                  className="mt-3 w-full py-2 rounded-lg text-[12px] font-bold text-[#070714] bg-[#00FFB2] hover:opacity-90 transition-opacity"
                >
                  Run full prediction with all 19 fields →
                </button>
              </div>
            )}
          </div>
        </section>

        {/* ── METRICS ──
            mobile:  2 columns (2×2 grid feels natural)
            desktop: 4 columns                                              */}
        <section className="border-t border-b border-white/[0.05] py-16 px-6 md:px-12">
          <div className="max-w-[1100px] mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
            {METRICS.map((m, i) => (
              <div key={i} className="text-center group">
                <div className="text-3xl mb-2 group-hover:scale-110 transition-transform duration-300">{m.icon}</div>
                <div className="text-[clamp(28px,3vw,42px)] font-black tracking-[-1.5px] text-white tabular-nums">
                  <AnimatedCounter end={m.value} suffix={m.suffix} prefix={m.prefix} />
                </div>
                <div className="text-[13px] text-[#555] mt-1">{m.label}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── FEATURES ──
            mobile:  single column
            desktop: two columns                                            */}
        <section className="max-w-[1100px] mx-auto px-6 md:px-12 py-24">
          <div className="text-center mb-16">
            <div className="text-[11px] text-[#00FFB2] font-bold tracking-[3px] mb-4 font-mono">CAPABILITIES</div>
            <h2 className="text-[clamp(28px,4vw,48px)] font-black tracking-[-1.5px] text-white">
              Built for real decisions
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {FEATURES.map((f, i) => (
              <div
                key={i}
                className="relative bg-white/[0.02] border border-white/[0.06] rounded-2xl p-8 transition-all duration-300 hover:-translate-y-1 cursor-default"
                onMouseEnter={e => { (e.currentTarget as HTMLDivElement).style.borderColor = f.color + "44"; }}
                onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.borderColor = "rgba(255,255,255,0.06)"; }}
              >
                {f.badge && (
                  <span className="absolute top-4 right-4 text-[10px] font-bold tracking-widest px-2.5 py-1 rounded-full bg-yellow-400/10 text-yellow-300 border border-yellow-400/20 font-mono">
                    {f.badge}
                  </span>
                )}
                <div
                  className="w-11 h-11 rounded-xl flex items-center justify-center text-xl mb-5 border"
                  style={{ background: f.color + "18", color: f.color, borderColor: f.color + "30" }}
                >
                  {f.icon}
                </div>
                <h3 className="text-[17px] font-bold text-white mb-2.5 tracking-tight">{f.title}</h3>
                <p className="text-[14px] text-[#666] leading-[1.72]">{f.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── RECENT PREDICTIONS TABLE ──
            Table gets a horizontal scroll wrapper on mobile so it never
            squishes — the inner div has min-w-[560px] to preserve columns  */}
        <section className="max-w-[1100px] mx-auto px-6 md:px-12 pb-24">
          <div className="flex justify-between items-center mb-8">
            <div>
              <h2 className="text-[28px] font-black text-white tracking-tight">Recent Predictions</h2>
              <p className="text-[14px] text-[#555] mt-1">Sample feed from the model</p>
            </div>
            <button className="text-[#00FFB2] border border-[rgba(0,255,178,0.3)] px-4 py-2 rounded-lg text-[12px] font-bold font-mono tracking-wide hover:bg-[rgba(0,255,178,0.06)] transition-colors">
              VIEW ALL →
            </button>
          </div>

          <div className="overflow-x-auto rounded-2xl">
            <div className="bg-white/[0.02] border border-white/[0.06] rounded-2xl overflow-hidden min-w-[560px]">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b border-white/[0.06]">
                    {["Customer", "Risk Score", "Level", "Change", "Action"].map(h => (
                      <th key={h} className="px-5 py-3.5 text-left text-[10px] text-[#3a3a3a] font-bold tracking-[1.5px] font-mono">
                        {h.toUpperCase()}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {RECENT.map((r) => (
                    <tr
                      key={r.id}
                      className="border-b border-white/[0.03] last:border-0 hover:bg-white/[0.025] transition-colors"
                    >
                      <td className="px-5 py-4 font-mono text-[13px] text-[#aaa]">{r.id}</td>
                      <td className="px-5 py-4"><ScoreBar score={r.score} /></td>
                      <td className="px-5 py-4"><RiskBadge risk={r.risk} /></td>
                      <td className="px-5 py-4 font-mono text-[13px] font-bold" style={{ color: r.delta.startsWith("+") ? "#FF5C5C" : "#00FFB2" }}>
                        {r.delta}
                      </td>
                      <td className="px-5 py-4">
                        <button className="bg-white/[0.04] border border-white/[0.08] text-[#666] px-3 py-1.5 rounded-md text-[11px] hover:text-white hover:border-white/20 transition-all">
                          Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* ── CTA ──
            tighter padding on mobile                                       */}
        <section className="max-w-[1100px] mx-auto px-6 md:px-12 pb-32">
          <div className="relative bg-gradient-to-br from-[rgba(0,255,178,0.08)] to-[rgba(0,191,255,0.05)] border border-[rgba(0,255,178,0.15)] rounded-3xl px-8 md:px-16 py-16 md:py-24 text-center overflow-hidden">
            <div className="absolute -top-20 -right-20 w-80 h-80 rounded-full pointer-events-none"
              style={{ background: "radial-gradient(circle, rgba(0,255,178,0.12), transparent 70%)" }} />
            <div className="absolute -bottom-24 -left-16 w-96 h-96 rounded-full pointer-events-none"
              style={{ background: "radial-gradient(circle, rgba(0,191,255,0.08), transparent 70%)" }} />

            <div className="relative">
              <h2 className="text-[clamp(28px,4vw,52px)] font-black text-white tracking-[-2px] mb-4">
                Start retaining customers<br />today.
              </h2>
              <p className="text-[17px] text-[#666] mb-10">Your ML model is live. Put it to work.</p>
              <div className="flex gap-3 justify-center flex-wrap">
                <button
                  onClick={goToPredict}
                  className="glow-btn bg-[#00FFB2] text-[#070714] font-black text-[16px] px-9 py-4 rounded-xl hover:scale-105 active:scale-95 transition-transform"
                >
                  Open Full Predictor →
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* ── FOOTER ──
            mobile:  stacked column, centered
            desktop: row, space-between                                     */}
        <footer className="border-t border-white/[0.05] px-6 md:px-12 py-8 max-w-[1100px] mx-auto flex flex-col md:flex-row items-center gap-4 md:gap-0 md:justify-between">
          <div className="flex items-center gap-2.5">
            <div className="relative w-8 h-8 flex-shrink-0">
              <Image
                src="/StayFlow.png"
                alt="StayFlow logo"
                fill
                className="object-contain rounded-lg"
                priority
              />
            </div>
            <span className="font-bold text-[17px] tracking-tight text-white">
              StayFlow<span className="text-[#00FFB2]">.</span>
            </span>
          </div>
          <span className="text-[13px] text-[#444]">Built with ❤️ by Maaz · Powered by ML</span>
        </footer>

      </div>
    </div>
  );
}