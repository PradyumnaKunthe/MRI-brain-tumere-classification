import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import Comparison from './pages/Comparison';
import Performance from './pages/Performance';
import DatasetInfo from './pages/DatasetInfo';
import About from './pages/About';
import Settings from './pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
        <Navbar />

        <div className="flex-1 flex max-w-7xl w-full mx-auto">
          <Sidebar />

          <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto max-w-full">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/comparison" element={<Comparison />} />
              <Route path="/performance" element={<Performance />} />
              <Route path="/dataset" element={<DatasetInfo />} />
              <Route path="/about" element={<About />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>

        {/* Clinical Footer */}
        <footer className="border-t border-slate-800/80 bg-slate-950 py-4 text-center text-xs text-slate-500 font-mono">
          NeuroScan AI • MRI Brain Tumor Detection Platform • 12,064 Curated Hospital Scans • Dual-Stream Research Framework
        </footer>
      </div>
    </BrowserRouter>
  );
}

