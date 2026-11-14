import React, { useState, useEffect } from 'react';
import {
  GitCompare,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Zap,
  Layers,
  ArrowRight,
  ShieldCheck,
  Brain,
  Upload,
  RefreshCw,
} from 'lucide-react';
import { api } from '../services/api';

const CLASS_COLORS = {
  Glioma: '#f43f5e',
  Meningioma: '#f59e0b',
  'No Tumor': '#10b981',
  Pituitary: '#0ea5e9',
};

export default function Comparison() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [samples, setSamples] = useState({});
  const [loadingSamples, setLoadingSamples] = useState(true);
  const [comparing, setComparing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Load sample gallery
  useEffect(() => {
    let isMounted = true;
    async function loadSamples() {
      try {
        const data = await api.getSamples();
        if (isMounted) {
          setSamples(data.samples || {});
          setLoadingSamples(false);
        }
      } catch (err) {
        console.error('Failed to load sample scans:', err);
        if (isMounted) setLoadingSamples(false);
      }
    }
    loadSamples();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleSampleSelect = async (sample) => {
    try {
      setComparing(true);
      setError(null);
      setResult(null);
      const res = await fetch(sample.url);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: 'image/jpeg' });
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(blob));

      // Automatically execute comparison for convenience
      const compareRes = await api.predictCompare(file);
      setResult(compareRes);
      setComparing(false);
    } catch (err) {
      setError('Failed to load or compare sample image.');
      setComparing(false);
    }
  };

  const runComparison = async () => {
    if (!selectedFile) {
      setError('Please select or upload an MRI scan first.');
      return;
    }

    try {
      setComparing(true);
      setError(null);
      const res = await api.predictCompare(selectedFile);
      setResult(res);
      setComparing(false);
    } catch (err) {
      console.error('Comparison error:', err);
      setError(err.response?.data?.detail || 'Comparison inference failed.');
      setComparing(false);
    }
  };

  const consensus = result?.consensus;
  const cnn = result?.models?.cnn;
  const svm = result?.models?.mobilenetv2_svm;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-semibold mb-2">
          <GitCompare className="w-3.5 h-3.5" />
          Head-to-Head Comparative Study
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Custom CNN vs. Pretrained MobileNetV2 + SVM
        </h1>
        <p className="text-slate-400 text-sm mt-1 max-w-3xl">
          Evaluate how end-to-end convolutional feature learning compares against transfer learning embeddings classified by a tuned Support Vector Machine on identical clinical scan slices.
        </p>
      </div>

      {/* Upload and Sample Selection Strip */}
      <div className="p-6 rounded-2xl glass-panel space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <h3 className="text-sm font-bold text-white">Select MRI Scan for Comparative Analysis</h3>
            <p className="text-xs text-slate-400">
              Both models will process this exact image with their required normalization pipelines.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <label className="cursor-pointer px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center gap-2 transition-all">
              <Upload className="w-3.5 h-3.5 text-sky-400" />
              Upload Custom Scan
              <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            </label>

            <button
              onClick={runComparison}
              disabled={!selectedFile || comparing}
              className="px-5 py-2 rounded-xl text-xs font-bold bg-gradient-to-r from-sky-500 to-cyan-500 hover:from-sky-400 hover:to-cyan-400 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-600 text-white shadow-md shadow-sky-500/25 flex items-center gap-2 transition-all"
            >
              {comparing ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-sky-200" />
                  Comparing Models...
                </>
              ) : (
                <>
                  <Zap className="w-3.5 h-3.5" />
                  Execute Comparison
                </>
              )}
            </button>
          </div>
        </div>

        {/* Quick Sample Selector */}
        {!loadingSamples && (
          <div className="pt-3 border-t border-slate-800 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400 font-medium">Quick Test Samples:</span>
            {Object.entries(samples).map(([classKey, list]) => {
              const sample = list[0];
              if (!sample) return null;
              return (
                <button
                  key={classKey}
                  onClick={() => handleSampleSelect(sample)}
                  className="px-3 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-sky-500/40 text-xs text-slate-300 font-mono capitalize transition-all"
                >
                  {classKey} Sample
                </button>
              );
            })}
          </div>
        )}

        {error && (
          <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Comparison Results */}
      {result && (
        <div className="space-y-6">
          {/* Agreement Status Banner */}
          <div
            className={`p-5 rounded-2xl border flex items-center justify-between gap-4 transition-all ${
              consensus.agreement
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
            }`}
          >
            <div className="flex items-center gap-3">
              {consensus.agreement ? (
                <div className="w-10 h-10 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0">
                  <CheckCircle2 className="w-6 h-6" />
                </div>
              ) : (
                <div className="w-10 h-10 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center shrink-0">
                  <AlertTriangle className="w-6 h-6" />
                </div>
              )}
              <div>
                <h3 className="font-bold text-base text-white">
                  {consensus.agreement
                    ? 'Cross-Architecture Diagnostic Consensus Reached'
                    : 'Diagnostic Model Discrepancy Detected'}
                </h3>
                <p className="text-xs mt-0.5 opacity-90">
                  {consensus.note}
                </p>
              </div>
            </div>

            <div className="text-right hidden sm:block">
              <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 block">
                Primary Recommendation
              </span>
              <span className="text-sm font-extrabold text-white">
                Rely on MobileNetV2 + SVM (92.50%)
              </span>
            </div>
          </div>

          {/* Side-by-Side Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Left Card: Custom CNN */}
            <div className="p-6 rounded-2xl glass-panel space-y-5 border-amber-500/30">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div>
                  <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-amber-500/15 text-amber-400 border border-amber-500/30">
                    Baseline Architecture
                  </span>
                  <h3 className="text-lg font-bold text-white mt-1">Custom CNN</h3>
                  <p className="text-xs text-slate-400">Trained from scratch on 7,720 training images</p>
                </div>
                <div className="text-right font-mono">
                  <div className="text-xs text-slate-400">Benchmark Acc</div>
                  <div className="text-lg font-extrabold text-amber-400">64.79%</div>
                </div>
              </div>

              {/* Diagnosis Badge */}
              <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400">Predicted Class</div>
                  <div className="text-xl font-black text-white mt-0.5">
                    {cnn.predicted_class}
                  </div>
                </div>
                <div className="text-right font-mono">
                  <div className="text-xs text-slate-400">Confidence</div>
                  <div className="text-2xl font-black text-amber-400">
                    {cnn.confidence_pct}
                  </div>
                </div>
              </div>

              {/* Specifications */}
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-800">
                  <span className="text-slate-400 block">Feature Extraction</span>
                  <span className="font-semibold text-slate-200">128-d Penultimate Dense</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-800">
                  <span className="text-slate-400 block">Inference Latency</span>
                  <span className="font-semibold text-sky-400 font-mono flex items-center gap-1">
                    <Clock className="w-3 h-3" /> {cnn.inference_time_ms} ms
                  </span>
                </div>
              </div>

              {/* Probability Bars */}
              <div className="space-y-2 pt-2 border-t border-slate-800">
                <span className="text-xs font-semibold text-slate-300">Softmax Class Probabilities:</span>
                <div className="space-y-1.5">
                  {Object.entries(cnn.probabilities).map(([cKey, prob]) => {
                    const pct = (prob * 100).toFixed(1);
                    return (
                      <div key={cKey} className="space-y-1">
                        <div className="flex justify-between text-xs font-mono">
                          <span className="capitalize text-slate-300">{cKey}</span>
                          <span className="text-slate-400">{pct}%</span>
                        </div>
                        <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                          <div
                            className="bg-amber-400 h-full rounded-full transition-all"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Right Card: MobileNetV2 + SVM (Champion) */}
            <div className="p-6 rounded-2xl glass-panel space-y-5 border-sky-500/40 relative">
              <div className="absolute -top-3 right-6 px-3 py-0.5 rounded-full bg-gradient-to-r from-sky-500 to-cyan-400 text-white text-[11px] font-bold shadow-md shadow-sky-500/30">
                ★ Best Performing Pipeline
              </div>

              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div>
                  <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-sky-500/15 text-sky-400 border border-sky-500/30">
                    Deep Feature Transfer Learning
                  </span>
                  <h3 className="text-lg font-bold text-white mt-1">MobileNetV2 + RBF SVM</h3>
                  <p className="text-xs text-slate-400">Pretrained ImageNet features + Tuned RBF SVC</p>
                </div>
                <div className="text-right font-mono">
                  <div className="text-xs text-slate-400">Benchmark Acc</div>
                  <div className="text-lg font-extrabold text-sky-400">92.50%</div>
                </div>
              </div>

              {/* Diagnosis Badge */}
              <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400">Predicted Class</div>
                  <div className="text-xl font-black text-white mt-0.5">
                    {svm.predicted_class}
                  </div>
                </div>
                <div className="text-right font-mono">
                  <div className="text-xs text-slate-400">Confidence</div>
                  <div className="text-2xl font-black text-sky-400">
                    {svm.confidence_pct}
                  </div>
                </div>
              </div>

              {/* Specifications */}
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-800">
                  <span className="text-slate-400 block">Feature Extraction</span>
                  <span className="font-semibold text-slate-200">1,280-d Global Avg Pooling</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-800">
                  <span className="text-slate-400 block">Inference Latency</span>
                  <span className="font-semibold text-sky-400 font-mono flex items-center gap-1">
                    <Clock className="w-3 h-3" /> {svm.inference_time_ms} ms
                  </span>
                </div>
              </div>

              {/* Probability Bars */}
              <div className="space-y-2 pt-2 border-t border-slate-800">
                <span className="text-xs font-semibold text-slate-300">Calibrated Class Probabilities:</span>
                <div className="space-y-1.5">
                  {Object.entries(svm.probabilities).map(([cKey, prob]) => {
                    const pct = (prob * 100).toFixed(1);
                    return (
                      <div key={cKey} className="space-y-1">
                        <div className="flex justify-between text-xs font-mono">
                          <span className="capitalize text-slate-300">{cKey}</span>
                          <span className="text-slate-400">{pct}%</span>
                        </div>
                        <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                          <div
                            className="bg-sky-400 h-full rounded-full transition-all"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Deep Architectural Analysis Card */}
      <div className="p-6 rounded-2xl glass-panel space-y-4">
        <div className="border-b border-slate-800 pb-3">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Brain className="w-4 h-4 text-sky-400" />
            Why MobileNetV2 + SVM Outperforms Custom CNN by +27.71%
          </h3>
          <p className="text-xs text-slate-400">
            Theoretical machine learning rationale behind the experimental benchmark difference
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
            <div className="font-bold text-sky-300 text-sm">1. Rich Pretrained Feature Hierarchy</div>
            <p className="text-slate-300 leading-relaxed">
              MobileNetV2 carries inverted residual bottleneck weights pretrained on over 1.4 million images from ImageNet. It has already learned generalized low-level edge, texture, and structural filters that transfer smoothly to radiological brain scans.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
            <div className="font-bold text-sky-300 text-sm">2. Maximum Margin Generalization</div>
            <p className="text-slate-300 leading-relaxed">
              While Custom CNN uses cross-entropy loss that can be sensitive to class imbalances and overfitting, the SVM constructs optimal hyperplanes that maximize margin distance between classes in high-dimensional Hilbert space via the RBF kernel.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
            <div className="font-bold text-sky-300 text-sm">3. Prevention of Gradient Vanishing</div>
            <p className="text-slate-300 leading-relaxed">
              Decoupling deep feature extraction from classification prevents the gradient degradation common when training deeper CNN architectures from scratch on modest medical datasets, ensuring rock-solid numerical stability.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

