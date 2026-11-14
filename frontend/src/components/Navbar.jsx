import React, { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, ShieldCheck, Cpu, AlertTriangle, Layers, BrainCircuit } from 'lucide-react';
import { api } from '../services/api';

export default function Navbar() {
  const [health, setHealth] = useState({ status: 'checking', ready: false });
  const location = useLocation();

  useEffect(() => {
    let isMounted = true;
    const checkStatus = async () => {
      try {
        const data = await api.getHealth();
        if (isMounted) setHealth(data);
      } catch (err) {
        if (isMounted) setHealth({ status: 'offline', ready: false });
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-cyan-400 p-0.5 shadow-lg shadow-sky-500/20 group-hover:scale-105 transition-transform">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <BrainCircuit className="w-5 h-5 text-sky-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-sky-100 to-sky-400 bg-clip-text text-transparent">
                NeuroScan AI
              </span>
              <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">
                v2.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400 -mt-0.5 font-medium hidden sm:block">
              Clinical MRI Brain Tumor Classification Platform
            </p>
          </div>
        </Link>

        {/* Live System Status & Action */}
        <div className="flex items-center gap-4">
          {/* Health Pill */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono">
            {health.ready ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span className="text-emerald-400 font-medium">5 Models Loaded</span>
              </>
            ) : health.status === 'checking' ? (
              <>
                <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse"></span>
                <span className="text-slate-400">Connecting...</span>
              </>
            ) : (
              <>
                <span className="h-2 w-2 rounded-full bg-rose-500"></span>
                <span className="text-rose-400">Backend Offline</span>
              </>
            )}
          </div>

          <Link
            to="/analysis"
            className="hidden sm:inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-gradient-to-r from-sky-500 to-cyan-500 hover:from-sky-400 hover:to-cyan-400 text-white shadow-md shadow-sky-500/25 transition-all active:scale-95"
          >
            <Activity className="w-3.5 h-3.5" />
            Analyze Scan
          </Link>
        </div>
      </div>
    </header>
  );
}

