import React, { useEffect, useState } from 'react';
import {
  Database,
  PieChart as PieChartIcon,
  ShieldCheck,
  Code2,
  Layers,
  FileSpreadsheet,
  Cpu,
  CheckCircle2,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { api } from '../services/api';

const CLASS_DISTRIBUTION = [
  { class: 'Glioma', train: 2414, val: 604, test: 755, total: 3773, pct: '31.3%' },
  { class: 'Meningioma', train: 1747, val: 436, test: 546, total: 2729, pct: '22.6%' },
  { class: 'No Tumor', train: 1556, val: 389, test: 487, total: 2432, pct: '20.2%' },
  { class: 'Pituitary', train: 2003, val: 501, test: 626, total: 3130, pct: '25.9%' },
];

export default function DatasetInfo() {
  const [datasetStats, setDatasetStats] = useState(null);

  useEffect(() => {
    let isMounted = true;
    async function loadStats() {
      try {
        const data = await api.getDatasetStats();
        if (isMounted) setDatasetStats(data);
      } catch (err) {
        console.error('Failed to load dataset stats:', err);
      }
    }
    loadStats();
    return () => {
      isMounted = false;
    };
  }, []);

  const chartData = CLASS_DISTRIBUTION.map((c) => ({
    name: c.class,
    Training: c.train,
    Validation: c.val,
    Testing: c.test,
  }));

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-semibold mb-2">
          <Database className="w-3.5 h-3.5" />
          Clinical Corpus Architecture
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Epic & CSCR Hospital Dataset Repository
        </h1>
        <p className="text-slate-400 text-sm mt-1 max-w-3xl">
          Comprehensive radiological manifest detailing 12,064 high-resolution magnetic resonance imaging scans stratified across four diagnostic cohorts.
        </p>
      </div>

      {/* High Level Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl glass-panel">
          <div className="text-xs text-slate-400 font-medium uppercase tracking-wider">Total Scans</div>
          <div className="text-3xl font-black text-white mt-1">12,064</div>
          <div className="text-xs text-sky-400 mt-1 font-mono">100% Verified Corpus</div>
        </div>

        <div className="p-5 rounded-xl glass-panel">
          <div className="text-xs text-slate-400 font-medium uppercase tracking-wider">Training Partition</div>
          <div className="text-3xl font-black text-sky-400 mt-1">7,720</div>
          <div className="text-xs text-slate-400 mt-1 font-mono">64.0% Split Ratio</div>
        </div>

        <div className="p-5 rounded-xl glass-panel">
          <div className="text-xs text-slate-400 font-medium uppercase tracking-wider">Validation Partition</div>
          <div className="text-3xl font-black text-cyan-400 mt-1">1,930</div>
          <div className="text-xs text-slate-400 mt-1 font-mono">16.0% Split Ratio</div>
        </div>

        <div className="p-5 rounded-xl glass-panel">
          <div className="text-xs text-slate-400 font-medium uppercase tracking-wider">Independent Test Set</div>
          <div className="text-3xl font-black text-indigo-400 mt-1">2,414</div>
          <div className="text-xs text-slate-400 mt-1 font-mono">20.0% Holdout Split</div>
        </div>
      </div>

      {/* Dataset Distribution Table */}
      <div className="p-6 rounded-2xl glass-panel space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-base font-bold text-white">Pathological Class Partition Manifest</h3>
            <p className="text-xs text-slate-400">Strictly stratified across Training, Validation, and Holdout Test subsets</p>
          </div>
          <span className="text-xs font-mono text-sky-400 bg-sky-500/10 px-2.5 py-1 rounded border border-sky-500/20">
            Epic & CSCR Hospital
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 uppercase font-mono text-[10px] tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Diagnostic Category</th>
                <th className="py-3 px-4">Training (64%)</th>
                <th className="py-3 px-4">Validation (16%)</th>
                <th className="py-3 px-4">Test Set (20%)</th>
                <th className="py-3 px-4">Total Scans</th>
                <th className="py-3 px-4 text-right">Corpus Share</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-medium">
              {CLASS_DISTRIBUTION.map((c) => (
                <tr key={c.class} className="hover:bg-slate-900/40 transition-colors">
                  <td className="py-3.5 px-4 font-bold text-white flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-sky-400" />
                    {c.class}
                  </td>
                  <td className="py-3.5 px-4 font-mono text-sky-300">{c.train.toLocaleString()}</td>
                  <td className="py-3.5 px-4 font-mono text-cyan-300">{c.val.toLocaleString()}</td>
                  <td className="py-3.5 px-4 font-mono text-indigo-300">{c.test.toLocaleString()}</td>
                  <td className="py-3.5 px-4 font-mono font-bold text-white">{c.total.toLocaleString()}</td>
                  <td className="py-3.5 px-4 font-mono text-right text-slate-400">{c.pct}</td>
                </tr>
              ))}
              <tr className="bg-slate-900/60 font-bold border-t-2 border-slate-700">
                <td className="py-3.5 px-4 text-sky-400">Total Aggregation</td>
                <td className="py-3.5 px-4 font-mono text-sky-400">7,720</td>
                <td className="py-3.5 px-4 font-mono text-cyan-400">1,930</td>
                <td className="py-3.5 px-4 font-mono text-indigo-400">2,414</td>
                <td className="py-3.5 px-4 font-mono text-white">12,064</td>
                <td className="py-3.5 px-4 font-mono text-right text-sky-400">100.0%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Distribution Chart */}
      <div className="p-6 rounded-2xl glass-panel space-y-4">
        <div>
          <h3 className="text-base font-bold text-white">Class Partition Visualization</h3>
          <p className="text-xs text-slate-400">Stacked scan frequencies across all four clinical categories</p>
        </div>

        <div className="h-72 w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
              <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 12, fontWeight: 500 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-700 shadow-xl text-xs space-y-1">
                        <div className="font-bold text-white text-sm">{label} Pathology</div>
                        <div className="text-sky-400 font-mono">Training: {payload[0].value} scans</div>
                        <div className="text-cyan-400 font-mono">Validation: {payload[1].value} scans</div>
                        <div className="text-indigo-400 font-mono">Testing: {payload[2].value} scans</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend wrapperStyle={{ paddingTop: 10, fontSize: 12 }} />
              <Bar dataKey="Training" stackId="a" fill="#0ea5e9" />
              <Bar dataKey="Validation" stackId="a" fill="#06b6d4" />
              <Bar dataKey="Testing" stackId="a" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Preprocessing & Normalization Protocol */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Custom CNN Preprocessing */}
        <div className="p-6 rounded-2xl glass-panel space-y-4 border-amber-500/30">
          <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
            <Code2 className="w-4 h-4" />
            Custom CNN Normalization Pipeline
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Raw MRI scans are scaled linearly into standard floating-point unit interval representations suitable for backpropagation with ReLU activations:
          </p>
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-amber-300 space-y-1">
            <div className="text-slate-400 text-[10px]"># Formula: Standard 8-bit to float range</div>
            <div>X_norm = image / 255.0</div>
            <div className="text-slate-400 text-[10px]">Range: [0.0, 1.0] | Spatial: 224 × 224 × 3 RGB</div>
          </div>
        </div>

        {/* MobileNetV2 Preprocessing */}
        <div className="p-6 rounded-2xl glass-panel space-y-4 border-sky-500/30">
          <div className="flex items-center gap-2 text-sky-400 font-bold text-sm">
            <Cpu className="w-4 h-4" />
            MobileNetV2 ImageNet Normalization Pipeline
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Pretrained feature extractors require zero-centered symmetric bipolar scaling to match ImageNet transfer learning weights:
          </p>
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-sky-300 space-y-1">
            <div className="text-slate-400 text-[10px]"># Formula: Pretrained MobileNetV2 format</div>
            <div>X_norm = (image / 127.5) - 1.0</div>
            <div className="text-slate-400 text-[10px]">Range: [-1.0, 1.0] | Spatial: 224 × 224 × 3 RGB</div>
          </div>
        </div>
      </div>

      {/* Strict Anti-Leakage Protocol Card */}
      <div className="p-6 rounded-2xl glass-panel space-y-3 border-emerald-500/30">
        <div className="flex items-center gap-2 text-emerald-400 font-bold text-base">
          <ShieldCheck className="w-5 h-5" />
          Strict Data Isolation & Anti-Leakage Protocol
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          The test split (2,414 scans) is never exposed to model training, feature selection, or hyperparameter optimization. During 3-fold StratifiedKFold cross-validation, the <code className="bg-slate-800 px-1 py-0.5 rounded text-sky-300">StandardScaler</code> is fitted exclusively on each training fold inside a scikit-learn Pipeline and only applied as a transform to the validation fold, preventing information leakage.
        </p>
      </div>
    </div>
  );
}

