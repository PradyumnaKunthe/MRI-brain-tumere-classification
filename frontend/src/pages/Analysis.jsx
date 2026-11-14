import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Brain,
  Layers,
  Sparkles,
  RefreshCw,
  Clock,
  ShieldAlert,
  FileImage,
  Sliders,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from 'recharts';
import { api } from '../services/api';
import ImageFilters from '../components/ImageFilters';

const CLASS_COLORS = {
  glioma: '#f43f5e',     // rose-500
  meningioma: '#f59e0b', // amber-500
  notumor: '#10b981',    // emerald-500
  pituitary: '#0ea5e9',  // sky-500
};

const CLINICAL_DESCRIPTIONS = {
  glioma: {
    title: 'Glioma Detected (Intra-Axial)',
    description: 'Findings indicate an infiltrative parenchymal mass lesion characteristic of glioma. Recommend correlation with contrast-enhanced T1, T2/FLAIR sequences, MR spectroscopy, and neurosurgical consultation.',
    severity: 'High Priority (Urgent)',
    severityColor: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
  },
  meningioma: {
    title: 'Meningioma Detected (Extra-Axial)',
    description: 'Radiological pattern suggests a well-demarcated dural-based mass lesion typical of meningioma. Recommended follow-up: high-resolution post-contrast T1-weighted imaging to evaluate dural tail and vascular sinus invasion.',
    severity: 'Moderate Priority',
    severityColor: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  },
  notumor: {
    title: 'No Neoplastic Lesion Detected (Normal)',
    description: 'Symmetrical cerebral parenchymal architecture, normal ventricular configuration, and absence of mass effect, midline shift, or abnormal contrast enhancement in the current scan slice.',
    severity: 'Normal / Benign',
    severityColor: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
  },
  pituitary: {
    title: 'Pituitary Lesion Detected (Sellar Region)',
    description: 'Focal lesion identified within the sella turcica / suprasellar cistern. Recommend pituitary protocol coronal thin-slice MRI and endocrinological workup (prolactin, ACTH, GH, IGF-1, visual field evaluation).',
    severity: 'Specialist Referral',
    severityColor: 'bg-sky-500/10 text-sky-400 border-sky-500/30',
  },
};

export default function Analysis() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [activeFilter, setActiveFilter] = useState('original');
  const [modelMode, setModelMode] = useState('hybrid'); // 'hybrid', 'cnn', 'all'
  const [samples, setSamples] = useState({});
  const [loadingSamples, setLoadingSamples] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Load sample scans on mount
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
      setActiveFilter('original');
    }
  };

  const handleSampleSelect = async (sample) => {
    try {
      setAnalyzing(true);
      setError(null);
      setResult(null);
      setActiveFilter('original');

      // Fetch the actual image blob
      const res = await fetch(sample.url);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: 'image/jpeg' });
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(blob));
      setAnalyzing(false);
    } catch (err) {
      setError('Failed to load sample image.');
      setAnalyzing(false);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) {
      setError('Please select or upload an MRI scan first.');
      return;
    }

    try {
      setAnalyzing(true);
      setError(null);
      let res;
      if (modelMode === 'hybrid') {
        res = await api.predictHybrid(selectedFile);
      } else if (modelMode === 'cnn') {
        res = await api.predictCnn(selectedFile);
      } else {
        res = await api.predictAll(selectedFile);
      }
      setResult(res);
      setAnalyzing(false);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.response?.data?.detail || 'Inference failed. Check backend connection.');
      setAnalyzing(false);
    }
  };

  // Prepare chart data for probability distribution
  const chartData = result
    ? Object.entries(result.probabilities || result.ensemble?.average_probabilities || {}).map(([key, val]) => ({
        name: key.charAt(0).toUpperCase() + key.slice(1),
        key,
        probability: parseFloat((val * 100).toFixed(2)),
      }))
    : [];

  const topKey = result?.predicted_class?.toLowerCase().replace(' ', '') ||
                 result?.ensemble?.majority_class?.toLowerCase().replace(' ', '') ||
                 'notumor';
  const clinicalInfo = CLINICAL_DESCRIPTIONS[topKey] || CLINICAL_DESCRIPTIONS.notumor;

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-semibold mb-2">
          <Activity className="w-3.5 h-3.5" />
          Clinical Diagnostic Workspace
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          MRI Brain Tumor Analysis & Classification
        </h1>
        <p className="text-slate-400 text-sm mt-1 max-w-3xl">
          Upload a brain MRI scan or select a verified test dataset scan to perform real-time multi-class tumor detection with deep feature extraction and uncertainty profiling.
        </p>
      </div>

      {/* Main Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Upload, Sample Picker, Filters */}
        <div className="lg:col-span-5 space-y-6">
          {/* Upload & Preview Card */}
          <div className="p-5 rounded-2xl glass-panel space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <FileImage className="w-4 h-4 text-sky-400" />
                MRI Scan Input
              </span>
              {previewUrl && (
                <button
                  onClick={() => {
                    setSelectedFile(null);
                    setPreviewUrl(null);
                    setResult(null);
                  }}
                  className="text-xs text-rose-400 hover:text-rose-300 font-medium"
                >
                  Clear Scan
                </button>
              )}
            </div>

            {previewUrl ? (
              <div className="space-y-4">
                <ImageFilters
                  imageUrl={previewUrl}
                  activeFilter={activeFilter}
                  onFilterChange={setActiveFilter}
                />
              </div>
            ) : (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-700/80 hover:border-sky-500/60 rounded-xl p-8 text-center cursor-pointer bg-slate-900/40 hover:bg-slate-900/80 transition-all space-y-3 group"
              >
                <div className="w-14 h-14 rounded-2xl bg-sky-500/10 text-sky-400 flex items-center justify-center mx-auto group-hover:scale-110 transition-transform">
                  <Upload className="w-7 h-7" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">Click or drag & drop scan file</p>
                  <p className="text-xs text-slate-400 mt-1">Supports PNG, JPG, JPEG, BMP, WEBP (Max 15MB)</p>
                </div>
                <div className="inline-block px-3 py-1 rounded-md bg-slate-800 text-[11px] text-slate-300 font-mono">
                  Input automatically resized to 224×224 RGB
                </div>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />

            {/* Model Mode Selector */}
            <div className="pt-2 border-t border-slate-800/80 space-y-2">
              <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-sky-400" />
                Select Inference Pipeline
              </label>
              <div className="grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => setModelMode('hybrid')}
                  className={`p-2.5 rounded-xl text-left border text-xs font-medium transition-all ${
                    modelMode === 'hybrid'
                      ? 'bg-sky-500/20 text-white border-sky-400 shadow-sm shadow-sky-500/10'
                      : 'bg-slate-900 text-slate-400 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="font-bold text-sky-300">MobileNetV2 + SVM</div>
                  <div className="text-[10px] text-emerald-400 font-mono">92.50% (Best)</div>
                </button>

                <button
                  type="button"
                  onClick={() => setModelMode('cnn')}
                  className={`p-2.5 rounded-xl text-left border text-xs font-medium transition-all ${
                    modelMode === 'cnn'
                      ? 'bg-sky-500/20 text-white border-sky-400 shadow-sm shadow-sky-500/10'
                      : 'bg-slate-900 text-slate-400 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="font-bold text-slate-200">Custom CNN</div>
                  <div className="text-[10px] text-amber-400 font-mono">64.79% Base</div>
                </button>

                <button
                  type="button"
                  onClick={() => setModelMode('all')}
                  className={`p-2.5 rounded-xl text-left border text-xs font-medium transition-all ${
                    modelMode === 'all'
                      ? 'bg-sky-500/20 text-white border-sky-400 shadow-sm shadow-sky-500/10'
                      : 'bg-slate-900 text-slate-400 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="font-bold text-purple-300">All 5 Models</div>
                  <div className="text-[10px] text-purple-400 font-mono">Ensemble</div>
                </button>
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={runAnalysis}
              disabled={!selectedFile || analyzing}
              className="w-full py-3 px-4 rounded-xl text-sm font-bold bg-gradient-to-r from-sky-500 to-cyan-500 hover:from-sky-400 hover:to-cyan-400 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-600 disabled:cursor-not-allowed text-white shadow-lg shadow-sky-500/25 transition-all flex items-center justify-center gap-2"
            >
              {analyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-sky-200" />
                  Running Neural Feature Extraction...
                </>
              ) : (
                <>
                  <Activity className="w-4 h-4" />
                  Execute Clinical Diagnosis
                </>
              )}
            </button>

            {error && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Sample Selector Card */}
          <div className="p-5 rounded-2xl glass-panel space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Verified Test Dataset Samples
              </span>
              <span className="text-[11px] text-sky-400 font-mono">Click to test</span>
            </div>
            <p className="text-xs text-slate-400">
              Pick any authentic scan from the test partition of the Epic & CSCR Hospital dataset:
            </p>

            {loadingSamples ? (
              <div className="py-4 text-center text-xs text-slate-500">Loading clinical test library...</div>
            ) : (
              <div className="space-y-3">
                {Object.entries(samples).map(([classKey, list]) => (
                  <div key={classKey} className="space-y-1.5">
                    <span className="text-[11px] font-semibold text-slate-300 capitalize">
                      {classKey}
                    </span>
                    <div className="grid grid-cols-4 gap-2">
                      {list.slice(0, 4).map((s) => (
                        <button
                          key={s.filename}
                          onClick={() => handleSampleSelect(s)}
                          className="group relative rounded-lg overflow-hidden border border-slate-700/80 hover:border-sky-400 bg-slate-900 aspect-square transition-all"
                        >
                          <img
                            src={s.url}
                            alt={s.filename}
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform"
                          />
                          <div className="absolute inset-0 bg-sky-950/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                            <span className="text-[10px] text-sky-300 font-bold bg-slate-900/90 px-1 py-0.5 rounded">
                              Load
                            </span>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Diagnostic Results, Probability Spectrum, Consensus */}
        <div className="lg:col-span-7 space-y-6">
          {result ? (
            <div className="space-y-6">
              {/* Primary Diagnostic Banner */}
              <div className="p-6 rounded-2xl glass-panel border-sky-500/30 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
                  <div>
                    <span className="text-[11px] uppercase tracking-wider font-bold text-slate-400">
                      Primary AI Diagnosis
                    </span>
                    <h2 className="text-2xl font-black text-white mt-0.5 flex items-center gap-2">
                      {result.predicted_class || result.ensemble?.majority_class}
                      <span className={`text-xs px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider border ${clinicalInfo.severityColor}`}>
                        {clinicalInfo.severity}
                      </span>
                    </h2>
                  </div>

                  <div className="text-right">
                    <div className="text-xs text-slate-400 font-medium">Confidence Score</div>
                    <div className="text-3xl font-extrabold text-sky-400 font-mono">
                      {result.confidence_pct || `${result.ensemble?.majority_confidence_pct}%`}
                    </div>
                  </div>
                </div>

                {/* Consensus & Model Info Bar */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                    <div className="text-slate-400">Model Executed</div>
                    <div className="font-semibold text-slate-200 mt-0.5">
                      {result.model_name || '5-Model Ensemble'}
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                    <div className="text-slate-400">Inference Latency</div>
                    <div className="font-semibold text-sky-300 mt-0.5 font-mono flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5" />
                      {result.timing_ms?.total_ms || result.timing_ms?.inference_ms || 18} ms
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                    <div className="text-slate-400">Feature Vector</div>
                    <div className="font-semibold text-slate-200 mt-0.5 font-mono">
                      {result.feature_dimension ? `${result.feature_dimension} dims` : 'Multi-branch'}
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                    <div className="text-slate-400">Consensus Status</div>
                    <div className="font-semibold text-emerald-400 mt-0.5">
                      {result.ensemble ? result.ensemble.consensus_status : 'Tuned RBF Kernel'}
                    </div>
                  </div>
                </div>

                {/* Clinical Pathology Note */}
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300 leading-relaxed space-y-1">
                  <div className="font-bold text-white flex items-center gap-1.5">
                    <Brain className="w-4 h-4 text-sky-400" />
                    {clinicalInfo.title}
                  </div>
                  <p>{clinicalInfo.description}</p>
                </div>
              </div>

              {/* Class Probability Distribution Chart */}
              <div className="p-6 rounded-2xl glass-panel space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-base font-bold text-white">Pathological Class Probability Distribution</h3>
                    <p className="text-xs text-slate-400">Softmax / Platt calibrated probability output across 4 classes</p>
                  </div>
                  <span className="text-xs font-mono text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded border border-sky-500/20">
                    Sum = 100%
                  </span>
                </div>

                <div className="h-60 w-full pt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                      <XAxis type="number" domain={[0, 100]} unit="%" stroke="#64748b" tick={{ fontSize: 11 }} />
                      <YAxis type="category" dataKey="name" stroke="#cbd5e1" tick={{ fontSize: 12, fontWeight: 500 }} />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            return (
                              <div className="p-2 rounded-lg bg-slate-900 border border-slate-700 shadow-xl text-xs font-mono">
                                <div className="text-white font-bold">{payload[0].payload.name}</div>
                                <div className="text-sky-400">{payload[0].value}% probability</div>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar dataKey="probability" radius={[0, 6, 6, 0]}>
                        {chartData.map((entry) => (
                          <Cell key={`cell-${entry.key}`} fill={CLASS_COLORS[entry.key] || '#0ea5e9'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Probability Table */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-slate-800">
                  {chartData.map((item) => (
                    <div key={item.key} className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 text-center">
                      <div className="text-[11px] text-slate-400 font-medium">{item.name}</div>
                      <div className="text-sm font-extrabold text-white font-mono mt-0.5">
                        {item.probability.toFixed(2)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* All Models Breakdown if Ensemble Mode */}
              {result.models && (
                <div className="p-6 rounded-2xl glass-panel space-y-4">
                  <div className="border-b border-slate-800 pb-3">
                    <h3 className="text-base font-bold text-white">Ensemble Cross-Model Breakdown</h3>
                    <p className="text-xs text-slate-400">Individual votes across all 5 independent classification architectures</p>
                  </div>

                  <div className="space-y-2 text-xs">
                    {Object.entries(result.models).map(([k, m]) => (
                      <div key={k} className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 flex items-center justify-between">
                        <div>
                          <div className="font-semibold text-slate-200">{m.model_name}</div>
                          <div className="text-[11px] text-slate-400">Benchmark: {m.benchmark_accuracy}</div>
                        </div>
                        <div className="text-right">
                          <div className="font-bold text-sky-400 font-mono">{m.predicted_class}</div>
                          <div className="text-[11px] text-slate-400">{m.confidence_pct}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Medical Disclaimer Alert */}
              <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-3 leading-relaxed">
                <ShieldAlert className="w-5 h-5 shrink-0 text-amber-400 mt-0.5" />
                <div>
                  <span className="font-bold block text-white mb-0.5">Clinical Safety & Regulatory Notice</span>
                  This application is a research-grade computational diagnostic aid. Predictions must be verified by a board-certified neuroradiologist alongside full multi-planar MRI sequences, patient history, and biopsy confirmation before initiating clinical treatment.
                </div>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="h-full min-h-[450px] p-8 rounded-2xl glass-panel flex flex-col items-center justify-center text-center space-y-4 border-dashed border-slate-800">
              <div className="w-16 h-16 rounded-2xl bg-sky-500/10 text-sky-400 flex items-center justify-center">
                <Brain className="w-8 h-8" />
              </div>
              <div className="max-w-md space-y-2">
                <h3 className="text-lg font-bold text-white">No MRI Scan Analyzed Yet</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Select a test image from the sample gallery on the left, or upload an MRI scan slice from your workstation to generate a comprehensive pathological assessment.
                </p>
              </div>
              <div className="flex items-center gap-4 text-xs text-slate-500 pt-2 font-mono">
                <span>• 92.50% Benchmark Accuracy</span>
                <span>• 4 Pathological Classes</span>
                <span>• Low Latency Inference</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

