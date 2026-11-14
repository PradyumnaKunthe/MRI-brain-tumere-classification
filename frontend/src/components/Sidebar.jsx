import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  ScanLine,
  GitCompare,
  BarChart3,
  Database,
  Info,
  Settings as SettingsIcon,
  ExternalLink,
  Sparkles,
} from 'lucide-react';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard, exact: true },
  { path: '/analysis', label: 'MRI Analysis', icon: ScanLine, badge: 'Live AI' },
  { path: '/comparison', label: 'Model Comparison', icon: GitCompare, badge: 'Head-to-Head' },
  { path: '/performance', label: 'Model Performance', icon: BarChart3, badge: '92.50%' },
  { path: '/dataset', label: 'Dataset Information', icon: Database, badge: '12,064 Scans' },
  { path: '/about', label: 'About & Architecture', icon: Info },
  { path: '/settings', label: 'System Health', icon: SettingsIcon },
];

export default function Sidebar() {
  return (
    <aside className="w-64 shrink-0 hidden md:block border-r border-slate-800/80 bg-slate-950/50 p-4 space-y-6">
      {/* Navigation Group */}
      <div>
        <p className="px-3 text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-2">
          Clinical Navigation
        </p>
        <nav className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.exact}
                className={({ isActive }) =>
                  `flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium transition-all group ${
                    isActive
                      ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30 shadow-sm shadow-sky-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent'
                  }`
                }
              >
                <div className="flex items-center gap-3">
                  <Icon className="w-4 h-4 text-sky-400 group-hover:scale-110 transition-transform" />
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-sky-300 border border-slate-700/60">
                    {item.badge}
                  </span>
                )}
              </NavLink>
            );
          })}
        </nav>
      </div>

      {/* Streamlit Fallback Card */}
      <div className="p-3.5 rounded-xl border border-sky-900/50 bg-gradient-to-b from-sky-950/40 to-slate-950/60 text-xs space-y-2">
        <div className="flex items-center justify-between text-sky-400 font-semibold">
          <span className="flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5" />
            Streamlit Fallback
          </span>
          <span className="text-[10px] bg-sky-500/20 text-sky-300 px-1.5 py-0.5 rounded">Active</span>
        </div>
        <p className="text-slate-400 text-[11px] leading-relaxed">
          The original Streamlit application remains fully operational at port 8501 as an emergency fallback.
        </p>
        <a
          href="http://localhost:8501"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-[11px] font-medium text-sky-400 hover:text-sky-300 transition-colors"
        >
          Open Streamlit Interface <ExternalLink className="w-3 h-3" />
        </a>
      </div>

      {/* Hospital Dataset Badge */}
      <div className="p-3 rounded-lg border border-slate-800/80 bg-slate-900/40 text-[11px] text-slate-400 space-y-1">
        <div className="font-semibold text-slate-300">Epic & CSCR Hospital</div>
        <div>12,064 Curated MRI Images across Glioma, Meningioma, Pituitary & Healthy controls.</div>
      </div>
    </aside>
  );
}

