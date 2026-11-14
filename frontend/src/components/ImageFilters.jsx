import React, { useEffect, useRef } from 'react';
import { Eye, Sliders, Contrast, Sparkles, Activity } from 'lucide-react';

export const FILTER_OPTIONS = [
  { id: 'original', name: 'Original Scan', icon: Eye },
  { id: 'contrast', name: 'CLAHE Enhanced', icon: Contrast },
  { id: 'grayscale', name: 'Grayscale Spec', icon: Sliders },
  { id: 'invert', name: 'Inverted Density', icon: Activity },
  { id: 'edge', name: 'Edge Boundary', icon: Sparkles },
];

export default function ImageFilters({ imageUrl, activeFilter, onFilterChange }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageUrl) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      canvas.width = img.naturalWidth || 300;
      canvas.height = img.naturalHeight || 300;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      if (activeFilter === 'original') return;

      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;
      const width = canvas.width;
      const height = canvas.height;

      if (activeFilter === 'grayscale') {
        for (let i = 0; i < data.length; i += 4) {
          const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          data[i] = gray;
          data[i + 1] = gray;
          data[i + 2] = gray;
        }
        ctx.putImageData(imgData, 0, 0);
      } else if (activeFilter === 'invert') {
        for (let i = 0; i < data.length; i += 4) {
          data[i] = 255 - data[i];
          data[i + 1] = 255 - data[i + 1];
          data[i + 2] = 255 - data[i + 2];
        }
        ctx.putImageData(imgData, 0, 0);
      } else if (activeFilter === 'contrast') {
        // High contrast / CLAHE approximation: stretch histogram
        let min = 255, max = 0;
        for (let i = 0; i < data.length; i += 4) {
          const v = (data[i] + data[i + 1] + data[i + 2]) / 3;
          if (v < min) min = v;
          if (v > max) max = v;
        }
        const range = max - min || 1;
        for (let i = 0; i < data.length; i += 4) {
          data[i] = Math.min(255, Math.max(0, ((data[i] - min) / range) * 255 * 1.15));
          data[i + 1] = Math.min(255, Math.max(0, ((data[i + 1] - min) / range) * 255 * 1.15));
          data[i + 2] = Math.min(255, Math.max(0, ((data[i + 2] - min) / range) * 255 * 1.15));
        }
        ctx.putImageData(imgData, 0, 0);
      } else if (activeFilter === 'edge') {
        // Sobel filter approximation
        const grayscale = new Float32Array(width * height);
        for (let i = 0, j = 0; i < data.length; i += 4, j++) {
          grayscale[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        }

        const outputData = ctx.createImageData(width, height);
        const out = outputData.data;

        for (let y = 1; y < height - 1; y++) {
          for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            const gx =
              -1 * grayscale[idx - width - 1] + 1 * grayscale[idx - width + 1] +
              -2 * grayscale[idx - 1]         + 2 * grayscale[idx + 1] +
              -1 * grayscale[idx + width - 1] + 1 * grayscale[idx + width + 1];

            const gy =
              -1 * grayscale[idx - width - 1] - 2 * grayscale[idx - width] - 1 * grayscale[idx - width + 1] +
               1 * grayscale[idx + width - 1] + 2 * grayscale[idx + width] + 1 * grayscale[idx + width + 1];

            const mag = Math.min(255, Math.sqrt(gx * gx + gy * gy) * 1.5);
            const oIdx = (y * width + x) * 4;
            out[oIdx] = 0;        // R
            out[oIdx + 1] = mag;  // G (cyan-green tint)
            out[oIdx + 2] = mag;  // B
            out[oIdx + 3] = 255;  // A
          }
        }
        ctx.putImageData(outputData, 0, 0);
      }
    };

    img.src = imageUrl;
  }, [imageUrl, activeFilter]);

  return (
    <div className="space-y-4">
      {/* Canvas for Rendering Filtered Scan */}
      <div className="relative rounded-xl overflow-hidden border border-slate-700/60 bg-black/60 flex items-center justify-center min-h-[300px] shadow-inner">
        <canvas ref={canvasRef} className="max-h-[380px] w-auto object-contain rounded-lg shadow-lg" />
        <div className="absolute top-3 right-3 px-2.5 py-1 rounded-md text-xs font-mono bg-slate-900/80 border border-slate-700 text-sky-400 backdrop-blur-sm">
          {FILTER_OPTIONS.find(f => f.id === activeFilter)?.name || 'Filter Active'}
        </div>
      </div>

      {/* Filter Control Pills */}
      <div className="flex flex-wrap gap-2">
        {FILTER_OPTIONS.map((f) => {
          const Icon = f.icon;
          const isActive = activeFilter === f.id;
          return (
            <button
              key={f.id}
              onClick={() => onFilterChange(f.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                isActive
                  ? 'bg-sky-500/20 text-sky-400 border border-sky-400/50 shadow-sm shadow-sky-500/10'
                  : 'bg-slate-800/80 text-slate-400 hover:text-slate-200 border border-slate-700/60 hover:border-slate-600'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {f.name}
            </button>
          );
        })}
      </div>
    </div>
  );
}

