import axios from 'axios';

// Use relative URL so Vite proxy forwards to FastAPI backend on port 8000
const API_BASE = '/api';

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export const api = {
  // System Health & Model Status
  getHealth: async () => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  // Available Models
  getModels: async () => {
    const res = await apiClient.get('/models');
    return res.data;
  },

  // Empirical Metrics & Comparison Data
  getMetrics: async () => {
    const res = await apiClient.get('/metrics');
    return res.data;
  },

  // Classification Reports
  getReport: async (modelKey) => {
    const res = await apiClient.get(`/metrics/report/${modelKey}`);
    return res.data;
  },

  // Dataset Statistics
  getDatasetStats: async () => {
    const res = await apiClient.get('/dataset/stats');
    return res.data;
  },

  // Pre-loaded Test Samples
  getSamples: async () => {
    const res = await apiClient.get('/samples');
    return res.data;
  },

  // Prediction: Custom CNN
  predictCnn: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post('/predict/cnn', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // Prediction: MobileNetV2 + SVM (Best Model)
  predictHybrid: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post('/predict/hybrid', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // Prediction: Side-by-side Head-to-Head Comparison
  predictCompare: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post('/predict/compare', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // Prediction: All 5 Models with Consensus
  predictAll: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post('/predict/all', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
};

export default api;

