import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  Award,
  Layers,
  FileText,
  TrendingUp,
  Image as ImageIcon,
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

const MODEL_CARDS = [
  {
    id: 'svm',
    name: 'MobileNetV2 + SVM',
    shortName: 'SVM (Best)',
    testAcc: 92.50,
    valAcc: 94.04,
    f1: 0.9250,
    matrixImg: '/api/outputs/mobilenetv2_plus_svm_confusion_matrix.png',
    reportKey: 'svm',
    isPrimary: true,
  },
  {
    id: 'combined',
    name: 'Combined Dual-Stream (CNN+MobileNet) + SVM',
    shortName: 'Dual-Stream',
    testAcc: 92.42,
    valAcc: 94.15,
    f1: 0.9242,
    matrixImg: '/api/outputs/combined_hybrid_confusion_matrix.png',
    reportKey: 'combined',
  },
  {
    id: 'lr',
    name: 'MobileNetV2 + Logistic Regression',
    shortName: 'LogReg',
    testAcc: 89.06,
    valAcc: 92.38,
    f1: 0.8904,
    matrixImg: '/api/outputs/mobilenetv2_plus_logistic_regression_confusion_matrix.png',
    reportKey: 'lr',
  },
  {
    id: 'rf',
    name: 'MobileNetV2 + Random Forest',
    shortName: 'Random Forest',
    testAcc: 84.71,
    valAcc: 88.08,
    f1: 0.8468,
    matrixImg: '/api/outputs/mobilenetv2_plus_random_forest_confusion_matrix.png',
    reportKey: 'rf',
  },
  {
    id: 'cnn',
    name: 'Custom 4-Block CNN',
    shortName: 'Custom CNN',
    testAcc: 64.79,
    valAcc: 68.24,
    f1: 0.6470,
    matrixImg: '/api/outputs/cnn_confusion_matrix.png',
    reportKey: 'cnn',
  },
];

export default function Performance() {
  const [selectedModel, setSelectedModel] = useState(MODEL_CARDS[0]);
  const [activeTab, setActiveTab] = useState('matrix'); // 'matrix', 'curves', 'report'
  const [reportText, setReportText] = useState('');
  const [loadingReport, setLoadingReport] = useState(false);

  useEffect(() => {
    let isMounted = true;
    async function loadReport() {
      setLoadingReport(true);
      try {
        const data = await api.getReport(selectedModel.reportKey);
        if (isMounted) {
          setReportText(data.report || 'Report not available.');
          setLoadingReport(false);
        }
      } catch (err) {
        if (isMounted) {
          setReportText('Could not load classification report file.');
          setLoadingReport(false);
        }
      }
    }
    loadReport();
    return () => {
      isMounted = false;
    };
  }, [selectedModel]);

  const chartData = MODEL_CARDS.map((m) => ({
    name: m.shortName,
    fullName: m.name,
    'Test Accuracy': m.testAcc,
    'Validation Accuracy': m.valAcc,
  }));

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-semibold mb-2">
          <BarChart3 className="w-3.5 h-3.5" />
          Empirical Evaluation & Benchmarking
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Model Evaluation & Performance Diagnostics
        </h1>
        <p className="text-slate-400 text-sm mt-1 max-w-3xl">
          Verified research metrics, interactive confusion matrices, and detailed precision-recall classification reports evaluated on 2,414 independent test MRI scans.
        </p>
      </div>

      {/* Model Benchmark Accuracy Table */}
      <div className="p-6 rounded-2xl glass-panel space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-base font-bold text-white">Comprehensive Model Leaderboard</h3>
            <p className="text-xs text-slate-400">Evaluated on independent 2,414-image test set (Epic & CSCR Hospital)</p>
          </div>
          <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/20">
            All 5 Pipelines Evaluated
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 uppercase font-mono text-[10px] tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Model Pipeline</th>
                <th className="py-3 px-4">Test Accuracy</th>
                <th className="py-3 px-4">Validation Accuracy</th>
                <th className="py-3 px-4">Weighted F1-Score</th>
                <th className="py-3 px-4">Classification Engine</th>
                <th className="py-3 px-4 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-medium">
              {MODEL_CARDS.map((m) => (
                <tr
                  key={m.id}
                  onClick={() => setSelectedModel(m)}
                  className={`cursor-pointer transition-colors ${
                    selectedModel.id === m.id
                      ? 'bg-sky-500/10 border-l-2 border-l-sky-400'
                      : 'hover:bg-slate-900/50'
                  }`}
                >
                  <td className="py-3.5 px-4">
                    <div className="flex items-center gap-2">
                      {m.isPrimary && <Award className="w-4 h-4 text-sky-400 shrink-0" />}
                      <span className={`font-bold ${m.isPrimary ? 'text-sky-300' : 'text-slate-200'}`}>
                        {m.name}
                      </span>
                    </div>
                  </td>
                  <td className="py-3.5 px-4 font-mono font-bold text-white text-sm">
                    {m.testAcc.toFixed(2)}%
                  </td>
                  <td className="py-3.5 px-4 font-mono text-slate-300">
                    {m.valAcc.toFixed(2)}%
                  </td>
                  <td className="py-3.5 px-4 font-mono text-slate-300">
                    {m.f1.toFixed(4)}
                  </td>
                  <td className="py-3.5 px-4 text-slate-400">
                    {m.id === 'svm' ? 'Tuned RBF SVC (C=10, γ=scale)' :
                     m.id === 'combined' ? '1408-d Dual-Stream Concatenation' :
                     m.id === 'lr' ? 'L2 Regularized Logistic Regression' :
                     m.id === 'rf' ? '100 Random Estimators' :
                     '4-Block Convolutional Softmax'}
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    <button
                      onClick={() => setSelectedModel(m)}
                      className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all ${
                        selectedModel.id === m.id
                          ? 'bg-sky-500 text-white shadow-sm'
                          : 'bg-slate-800 text-slate-400 hover:text-white'
                      }`}
                    >
                      {selectedModel.id === m.id ? 'Inspecting' : 'Inspect'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Comparative Performance Chart */}
      <div className="p-6 rounded-2xl glass-panel space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white">Comparative Accuracy Spectrum</h3>
            <p className="text-xs text-slate-400">Test Accuracy vs. Validation Accuracy Across All Architectures</p>
          </div>
        </div>

        <div className="h-72 w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
              <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 12, fontWeight: 500 }} />
              <YAxis domain={[50, 100]} unit="%" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-700 shadow-xl text-xs space-y-1">
                        <div className="font-bold text-white text-sm">{payload[0].payload.fullName}</div>
                        <div className="text-sky-400 font-mono">Test Accuracy: {payload[0].value}%</div>
                        <div className="text-cyan-400 font-mono">Validation Accuracy: {payload[1].value}%</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend wrapperStyle={{ paddingTop: 10, fontSize: 12 }} />
              <Bar dataKey="Test Accuracy" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Validation Accuracy" fill="#06b6d4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Deep Model Diagnostic Inspector */}
      <div className="p-6 rounded-2xl glass-panel space-y-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div>
            <span className="text-[11px] font-mono text-sky-400 uppercase tracking-wider">
              Diagnostic Inspector
            </span>
            <h3 className="text-lg font-bold text-white mt-0.5">
              Inspecting: {selectedModel.name}
            </h3>
          </div>

          {/* Tab Selector */}
          <div className="flex items-center gap-2 bg-slate-900 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab('matrix')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                activeTab === 'matrix'
                  ? 'bg-sky-500 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <ImageIcon className="w-3.5 h-3.5" />
              Confusion Matrix
            </button>

            {selectedModel.id === 'cnn' && (
              <button
                onClick={() => setActiveTab('curves')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                  activeTab === 'curves'
                    ? 'bg-sky-500 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <TrendingUp className="w-3.5 h-3.5" />
                Training History
              </button>
            )}

            <button
              onClick={() => setActiveTab('report')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                activeTab === 'report'
                  ? 'bg-sky-500 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              Classification Report
            </button>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'matrix' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Actual vs. Predicted class counts on 2,414 test MRI samples</span>
              <span className="font-mono text-sky-400">{selectedModel.name}</span>
            </div>
            <div className="flex justify-center bg-slate-950/80 rounded-xl p-4 border border-slate-800">
              <img
                src={selectedModel.matrixImg}
                alt={`${selectedModel.name} Confusion Matrix`}
                className="max-h-[460px] w-auto object-contain rounded-lg shadow-xl"
              />
            </div>
          </div>
        )}

        {activeTab === 'curves' && selectedModel.id === 'cnn' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Epoch Accuracy and Loss Curves across 10 Training Epochs</span>
              <span className="font-mono text-sky-400">Custom CNN Architecture</span>
            </div>
            <div className="flex justify-center bg-slate-950/80 rounded-xl p-4 border border-slate-800">
              <img
                src="/api/outputs/cnn_training_history.png"
                alt="CNN Training History"
                className="max-h-[460px] w-auto object-contain rounded-lg shadow-xl"
              />
            </div>
          </div>
        )}

        {activeTab === 'report' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Detailed Per-Class Precision, Recall, F1-Score & Support</span>
              <span className="font-mono text-sky-400">{selectedModel.name}</span>
            </div>
            {loadingReport ? (
              <div className="py-8 text-center text-xs text-slate-500">Loading classification report...</div>
            ) : (
              <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-sky-300 font-mono text-xs overflow-x-auto leading-relaxed">
                {reportText}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

