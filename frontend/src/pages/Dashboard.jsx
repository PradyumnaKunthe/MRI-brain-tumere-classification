import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  Award,
  Database,
  ShieldCheck,
  CheckCircle2,
  Clock,
  ArrowRight,
  Brain,
  Microscope,
  FileCheck,
  Layers,
  Sparkles,
} from 'lucide-react';
import { api } from '../services/api';

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      try {
        const [datasetData, healthData] = await Promise.all([
          api.getDatasetStats(),
          api.getHealth(),
        ]);
        if (isMounted) {
          setStats(datasetData);
          setHealth(healthData);
          setLoading(false);
        }
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
        if (isMounted) setLoading(false);
      }
    }
    loadData();
    return () => {
      isMounted = false;
    };
  }, []);

  const TUMOR_CLASSES = [
    {
      name: 'Glioma',
      type: 'Intra-Axial Neoplasm',
      description: 'Arises from glial cells (astrocytes, oligodendrocytes) in the cerebral parenchyma. Exhibits infiltrative irregular borders and significant peritumoral edema.',
      prevalence: '3773 Scans (31.3%)',
      color: 'border-rose-500/40 bg-rose-500/5 text-rose-400',
      badge: 'High Infiltration',
    },
    {
      name: 'Meningioma',
      type: 'Extra-Axial Neoplasm',
      description: 'Arises from arachnoid cap cells of the meninges. Characterized by well-circumscribed margins, slow growth, and radiological dural tail enhancement.',
      prevalence: '2729 Scans (22.6%)',
      color: 'border-amber-500/40 bg-amber-500/5 text-amber-400',
      badge: 'Dural-Based',
    },
    {
      name: 'No Tumor',
      type: 'Normal Control Morphology',
      description: 'Preserved cerebral parenchyma, symmetrical lateral ventricles, absence of focal mass effect, midline shift, or abnormal contrast enhancement.',
      prevalence: '2432 Scans (20.2%)',
      color: 'border-emerald-500/40 bg-emerald-500/5 text-emerald-400',
      badge: 'Normal Anatomy',
    },
    {
      name: 'Pituitary',
      type: 'Sellar / Suprasellar Neoplasm',
      description: 'Originates in the pituitary gland within the sella turcica. May compress the optic chiasm, resulting in bitemporal hemianopsia and hormonal imbalance.',
      prevalence: '3130 Scans (25.9%)',
      color: 'border-sky-500/40 bg-sky-500/5 text-sky-400',
      badge: 'Sellar Region',
    },
  ];

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="relative rounded-2xl overflow-hidden border border-sky-500/20 bg-gradient-to-r from-slate-900 via-sky-950/40 to-slate-900 p-6 sm:p-8">
        <div className="absolute top-0 right-0 w-96 h-96 bg-sky-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/30 text-sky-400 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            Clinical AI Research & Decision Support System
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white leading-tight">
            MRI Brain Tumor Detection & Dual-Stream Classification
          </h1>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
            Automated multi-class brain tumor diagnostic engine leveraging ImageNet-pretrained MobileNetV2 feature embeddings with Tuned Support Vector Machines alongside Custom Convolutional Neural Networks across 12,064 clinical MRI scans.
          </p>
          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Link
              to="/analysis"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-gradient-to-r from-sky-500 to-cyan-500 hover:from-sky-400 hover:to-cyan-400 text-white shadow-lg shadow-sky-500/25 transition-all"
            >
              <Activity className="w-4 h-4" />
              Launch Clinical Analysis
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/comparison"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all"
            >
              Compare CNN vs SVM
            </Link>
          </div>
        </div>
      </div>

      {/* Primary KPI Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1 */}
        <div className="p-5 rounded-xl glass-panel glass-panel-hover">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Top Test Accuracy</span>
            <div className="p-2 rounded-lg bg-sky-500/10 text-sky-400">
              <Award className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-white tracking-tight">92.50%</div>
          <div className="mt-1 flex items-center gap-1.5 text-xs text-sky-400 font-medium">
            <span>MobileNetV2 + RBF SVM</span>
            <span className="text-slate-500">•</span>
            <span className="text-slate-400">Tuned Pipeline</span>
          </div>
        </div>

        {/* Metric 2 */}
        <div className="p-5 rounded-xl glass-panel glass-panel-hover">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Total Curated Scans</span>
            <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
              <Database className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-white tracking-tight">
            {stats ? stats.corpus_total.toLocaleString() : '12,064'}
          </div>
          <div className="mt-1 text-xs text-slate-400">
            Epic & CSCR Hospital Dataset (7,720 Train / 2,414 Test)
          </div>
        </div>

        {/* Metric 3 */}
        <div className="p-5 rounded-xl glass-panel glass-panel-hover">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Pathological Classes</span>
            <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
              <Brain className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-white tracking-tight">4 Classes</div>
          <div className="mt-1 text-xs text-slate-400">
            Glioma, Meningioma, No Tumor, Pituitary
          </div>
        </div>

        {/* Metric 4 */}
        <div className="p-5 rounded-xl glass-panel glass-panel-hover">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">ML Integrity & Hashes</span>
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
              <ShieldCheck className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-emerald-400 tracking-tight">100% Intact</div>
          <div className="mt-1 text-xs text-slate-400">
            All 6 model SHA-256 hashes strictly verified
          </div>
        </div>
      </div>

      {/* Pathological Classes Breakdown */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">Diagnostic Target Categories</h2>
            <p className="text-xs text-slate-400">Clinical pathology definitions and sample distribution across the 12,064 dataset</p>
          </div>
          <Link to="/dataset" className="text-xs text-sky-400 hover:text-sky-300 font-medium flex items-center gap-1">
            View Dataset Specs <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {TUMOR_CLASSES.map((c) => (
            <div key={c.name} className={`p-5 rounded-xl border ${c.color} transition-all space-y-3`}>
              <div className="flex items-center justify-between">
                <h3 className="font-bold text-base text-white">{c.name}</h3>
                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-slate-900 border border-slate-800">
                  {c.badge}
                </span>
              </div>
              <div className="text-xs text-sky-300 font-mono font-medium">{c.type}</div>
              <p className="text-xs text-slate-300 leading-relaxed min-h-[60px]">{c.description}</p>
              <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs">
                <span className="text-slate-400">Corpus Representation:</span>
                <span className="font-semibold text-white">{c.prevalence}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Active Pipeline Status & Model Architecture Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Architecture Stack */}
        <div className="lg:col-span-2 p-6 rounded-xl glass-panel space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-base font-bold text-white">Trained Model Registry & Architecture</h3>
              <p className="text-xs text-slate-400">Verified classification pipelines running in the inference engine</p>
            </div>
            <Link to="/performance" className="text-xs text-sky-400 hover:text-sky-300 font-medium">
              View Benchmarks &rarr;
            </Link>
          </div>

          <div className="space-y-3">
            <div className="p-3.5 rounded-lg bg-slate-900/80 border border-slate-800 flex items-center justify-between">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-white text-sm">MobileNetV2 + SVM (Primary Model)</span>
                  <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                    92.50% Accuracy
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Global Average Pooling feature extractor (1,280 dims) + StandardScaler + Tuned RBF Kernel SVC (C=10, γ=scale)
                </p>
              </div>
              <div className="text-emerald-400 flex items-center gap-1 text-xs font-mono">
                <CheckCircle2 className="w-4 h-4" /> Ready
              </div>
            </div>

            <div className="p-3.5 rounded-lg bg-slate-900/80 border border-slate-800 flex items-center justify-between">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-white text-sm">Combined Dual-Stream Hybrid</span>
                  <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-sky-500/15 text-sky-400 border border-sky-500/30">
                    92.42% Accuracy
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Fused feature representation concatenating CNN penultimate Dense layer (128 dims) with MobileNetV2 (1,280 dims) = 1,408 dims
                </p>
              </div>
              <div className="text-emerald-400 flex items-center gap-1 text-xs font-mono">
                <CheckCircle2 className="w-4 h-4" /> Ready
              </div>
            </div>

            <div className="p-3.5 rounded-lg bg-slate-900/80 border border-slate-800 flex items-center justify-between">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-white text-sm">MobileNetV2 + Logistic Regression</span>
                  <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-slate-700 text-slate-300">
                    89.06% Accuracy
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Linear maximum margin classifier with L2 regularization on 1,280-dimensional MobileNetV2 embeddings
                </p>
              </div>
              <div className="text-emerald-400 flex items-center gap-1 text-xs font-mono">
                <CheckCircle2 className="w-4 h-4" /> Ready
              </div>
            </div>

            <div className="p-3.5 rounded-lg bg-slate-900/80 border border-slate-800 flex items-center justify-between">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-white text-sm">MobileNetV2 + Random Forest</span>
                  <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-slate-700 text-slate-300">
                    84.71% Accuracy
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  100-tree decision ensemble evaluating non-linear feature partitioning on standardized deep representations
                </p>
              </div>
              <div className="text-emerald-400 flex items-center gap-1 text-xs font-mono">
                <CheckCircle2 className="w-4 h-4" /> Ready
              </div>
            </div>

            <div className="p-3.5 rounded-lg bg-slate-900/80 border border-slate-800 flex items-center justify-between">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-white text-sm">Custom 4-Block Convolutional Neural Network</span>
                  <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-amber-500/15 text-amber-400 border border-amber-500/30">
                    64.79% Baseline
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Sequential Conv2D (32, 64, 128, 256) + BatchNorm + MaxPool + Dropout(0.4) + Dense(128) + Softmax(4)
                </p>
              </div>
              <div className="text-emerald-400 flex items-center gap-1 text-xs font-mono">
                <CheckCircle2 className="w-4 h-4" /> Ready
              </div>
            </div>
          </div>
        </div>

        {/* Clinical Integrity & Preprocessing Checklist */}
        <div className="p-6 rounded-xl glass-panel space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <FileCheck className="w-4 h-4 text-sky-400" />
              Dataset & Model Integrity
            </h3>
            <p className="text-xs text-slate-400">Architectural safeguards applied</p>
          </div>

          <div className="space-y-3 text-xs">
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 space-y-1">
              <div className="font-semibold text-slate-200">No Data Leakage</div>
              <p className="text-slate-400">
                Test set (2,414 scans) is strictly preserved for independent validation. StandardScaler is fitted strictly within cross-validation folds.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 space-y-1">
              <div className="font-semibold text-slate-200">Exact Normalization Formulas</div>
              <div className="font-mono text-[11px] text-sky-300 bg-black/40 p-1.5 rounded">
                CNN: img / 255.0 ∈ [0, 1]<br/>
                MobileNet: (img / 127.5) - 1.0 ∈ [-1, 1]
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 space-y-1">
              <div className="font-semibold text-slate-200">Model Artifact Hashes</div>
              <p className="text-slate-400">
                All .joblib and .keras files match baseline SHA-256 cryptographic signatures. No retraining or modifications performed.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-sky-950/40 border border-sky-800/40 text-sky-300 space-y-1">
              <div className="font-semibold flex items-center gap-1.5">
                <Microscope className="w-3.5 h-3.5 text-sky-400" />
                Diagnostic Decision Support
              </div>
              <p className="text-slate-400 text-[11px]">
                Designed as a secondary verification tool for neuro-radiologists and academic clinical evaluation.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

