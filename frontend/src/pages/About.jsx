import React from 'react';
import {
  Info,
  BrainCircuit,
  Layers,
  Cpu,
  ShieldAlert,
  Code,
  GraduationCap,
  Sparkles,
  ArrowRight,
  Server,
  Monitor,
} from 'lucide-react';

export default function About() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-semibold mb-2">
          <Info className="w-3.5 h-3.5" />
          Academic Research & Technical Architecture
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          System Architecture & Research Methodology
        </h1>
        <p className="text-slate-400 text-sm mt-1 max-w-3xl">
          Deep transfer learning and dual-stream feature fusion for magnetic resonance neuro-oncology imaging, developed as a clinical decision support framework.
        </p>
      </div>

      {/* Visual System Architecture Diagram */}
      <div className="p-6 rounded-2xl glass-panel space-y-6">
        <div>
          <h3 className="text-base font-bold text-white">Dual-Branch End-to-End System Architecture</h3>
          <p className="text-xs text-slate-400">Flow diagram from raw DICOM/MRI input slice to multi-classifier consensus</p>
        </div>

        {/* Diagram Flow */}
        <div className="p-6 rounded-xl bg-slate-950/80 border border-slate-800 space-y-6">
          {/* Step 1: Input */}
          <div className="flex flex-col items-center">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-700 text-center max-w-sm w-full shadow-lg">
              <span className="text-[10px] font-mono uppercase tracking-wider text-sky-400">Stage 1: Input Acquisition</span>
              <div className="font-bold text-white text-sm mt-0.5">Brain MRI Axial Scan Slice</div>
              <div className="text-xs text-slate-400 mt-1 font-mono">224 × 224 × 3 RGB | Lanczos Resampling</div>
            </div>
            <div className="h-6 w-0.5 bg-sky-500/50 my-1"></div>
          </div>

          {/* Step 2: Dual Branch Preprocessing & Feature Extraction */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Branch A: Custom CNN */}
            <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/30 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono uppercase tracking-wider text-amber-400">Branch A: Scratch CNN</span>
                <span className="text-[10px] bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded font-mono">64.79% Acc</span>
              </div>
              <div className="font-bold text-white text-sm">Custom 4-Block Conv2D Pipeline</div>
              <div className="text-xs text-slate-300 font-mono space-y-1">
                <div>• Normalization: image / 255.0 ∈ [0, 1]</div>
                <div>• Conv2D(32) → Conv2D(64) → Conv2D(128) → Conv2D(256)</div>
                <div>• BatchNorm + MaxPool(2x2) + Dropout(0.4)</div>
                <div>• Output: 128-d Penultimate Dense Embedding</div>
              </div>
            </div>

            {/* Branch B: MobileNetV2 Transfer Learning */}
            <div className="p-4 rounded-xl bg-sky-500/5 border border-sky-500/30 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono uppercase tracking-wider text-sky-400">Branch B: Transfer Learning</span>
                <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded font-mono">92.50% Acc</span>
              </div>
              <div className="font-bold text-white text-sm">ImageNet Pretrained MobileNetV2</div>
              <div className="text-xs text-slate-300 font-mono space-y-1">
                <div>• Normalization: (image / 127.5) - 1.0 ∈ [-1, 1]</div>
                <div>• Inverted Residual Bottleneck Layers (Frozen)</div>
                <div>• Global Average Pooling 2D</div>
                <div>• Output: 1,280-d Feature Representation</div>
              </div>
            </div>
          </div>

          {/* Step 3: Classification & Feature Fusion */}
          <div className="flex flex-col items-center">
            <div className="h-6 w-0.5 bg-sky-500/50 my-1"></div>
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-700 text-center max-w-xl w-full shadow-lg space-y-2">
              <span className="text-[10px] font-mono uppercase tracking-wider text-indigo-400">Stage 3: Classifier & Ensemble Array</span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <div className="p-2 rounded bg-sky-500/10 border border-sky-500/30 font-semibold text-sky-300">
                  RBF SVM (C=10)<br/><span className="text-[10px] text-emerald-400">92.50%</span>
                </div>
                <div className="p-2 rounded bg-indigo-500/10 border border-indigo-500/30 font-semibold text-indigo-300">
                  Dual Fusion (1408d)<br/><span className="text-[10px] text-sky-400">92.42%</span>
                </div>
                <div className="p-2 rounded bg-slate-800 border border-slate-700 font-semibold text-slate-300">
                  Logistic Reg<br/><span className="text-[10px] text-slate-400">89.06%</span>
                </div>
                <div className="p-2 rounded bg-slate-800 border border-slate-700 font-semibold text-slate-300">
                  Random Forest<br/><span className="text-[10px] text-slate-400">84.71%</span>
                </div>
              </div>
            </div>
            <div className="h-6 w-0.5 bg-sky-500/50 my-1"></div>
          </div>

          {/* Step 4: Final Output */}
          <div className="flex flex-col items-center">
            <div className="p-4 rounded-xl bg-gradient-to-r from-sky-950 to-indigo-950 border border-sky-500/40 text-center max-w-md w-full shadow-xl">
              <span className="text-[10px] font-mono uppercase tracking-wider text-emerald-400">Stage 4: Clinical Diagnostic Output</span>
              <div className="font-bold text-white text-base mt-0.5">Ensemble Consensus & Probabilities</div>
              <div className="text-xs text-sky-300 mt-1 font-mono">
                Glioma | Meningioma | No Tumor | Pituitary
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Technology Stack Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Machine Learning Core */}
        <div className="p-6 rounded-2xl glass-panel space-y-3">
          <div className="flex items-center gap-2 text-sky-400 font-bold text-sm">
            <Cpu className="w-4 h-4" />
            Machine Learning Core
          </div>
          <ul className="text-xs text-slate-300 space-y-2 leading-relaxed">
            <li><strong className="text-white">Python 3.11.9:</strong> Runtime execution environment.</li>
            <li><strong className="text-white">TensorFlow 2.16 & Keras 3:</strong> Deep neural network architectures and MobileNetV2 feature extraction.</li>
            <li><strong className="text-white">Scikit-Learn 1.4+:</strong> GridSearchCV, StratifiedKFold, Pipeline, StandardScaler, and Support Vector Classification.</li>
          </ul>
        </div>

        {/* Integration API Layer */}
        <div className="p-6 rounded-2xl glass-panel space-y-3">
          <div className="flex items-center gap-2 text-indigo-400 font-bold text-sm">
            <Server className="w-4 h-4" />
            Integration API Layer
          </div>
          <ul className="text-xs text-slate-300 space-y-2 leading-relaxed">
            <li><strong className="text-white">FastAPI:</strong> High-performance asynchronous REST endpoints.</li>
            <li><strong className="text-white">ExistingModelAdapter:</strong> Singleton bridge reusing exact model checkpoints without ML code modifications.</li>
            <li><strong className="text-white">Uvicorn:</strong> ASGI web server with sub-20ms inference latency.</li>
          </ul>
        </div>

        {/* Frontend Experience */}
        <div className="p-6 rounded-2xl glass-panel space-y-3">
          <div className="flex items-center gap-2 text-cyan-400 font-bold text-sm">
            <Monitor className="w-4 h-4" />
            Clinical Web Interface
          </div>
          <ul className="text-xs text-slate-300 space-y-2 leading-relaxed">
            <li><strong className="text-white">React 18 & Vite:</strong> Ultra-fast reactive single-page clinical workstation.</li>
            <li><strong className="text-white">Tailwind CSS:</strong> Professional medical grade visual layout and glassmorphism.</li>
            <li><strong className="text-white">Recharts & Lucide:</strong> Dynamic probability distributions and responsive diagnostic iconography.</li>
          </ul>
        </div>
      </div>

      {/* Academic Defense & Project Evaluation Notes */}
      <div className="p-6 rounded-2xl glass-panel space-y-4 border-indigo-500/30">
        <div className="flex items-center gap-2 text-indigo-400 font-bold text-base">
          <GraduationCap className="w-5 h-5" />
          Academic Project Viva & Faculty Evaluation Guide
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          Key discussion points for presentation to college faculty and project evaluators:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="font-bold text-white block">1. Why did MobileNetV2 + SVM beat Custom CNN?</span>
            <p className="text-slate-400 leading-relaxed">
              Medical imaging datasets are often insufficient in volume to train deep CNN kernels from scratch without overfitting. ImageNet pretraining provides robust feature primitives, while SVM's structural risk minimization maximizes the margin between pathological classes.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="font-bold text-white block">2. How was data leakage prevented?</span>
            <p className="text-slate-400 leading-relaxed">
              StandardScaler was bundled inside an sklearn Pipeline with 3-fold StratifiedKFold. Scaling parameters (mean and variance) were computed strictly on the training fold and applied only as a transform to the validation fold. The 2,414 test scans were never seen during tuning.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="font-bold text-white block">3. Why build an isolated FastAPI integration layer?</span>
            <p className="text-slate-400 leading-relaxed">
              In enterprise healthcare software, existing validated and audited ML model artifacts must never be destabilized. The isolated adapter layer wraps the existing model artifacts with zero code alteration, ensuring 100% cryptographic hash identity.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="font-bold text-white block">4. What is the role of Streamlit?</span>
            <p className="text-slate-400 leading-relaxed">
              The original Streamlit application (app.py) is kept completely untouched on port 8501, functioning as a high-availability fallback if web microservice components encounter infrastructure downtime.
            </p>
          </div>
        </div>
      </div>

      {/* Clinical Disclaimer */}
      <div className="p-5 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-3 leading-relaxed">
        <ShieldAlert className="w-5 h-5 shrink-0 text-amber-400 mt-0.5" />
        <div>
          <span className="font-bold block text-white mb-0.5">Ethical AI & Clinical Deployment Notice</span>
          This software is engineered exclusively for academic and research evaluation. In clinical settings, computer-aided detection systems must operate under the supervision of qualified radiologists, neuro-oncologists, and clinical pathologists in compliance with HIPAA, FDA, and CE medical software regulations.
        </div>
      </div>
    </div>
  );
}

