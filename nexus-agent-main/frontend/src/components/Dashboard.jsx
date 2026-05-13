import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { fetchDashboardData, fetchChartData } from '../utils/api';
import MetricsCards from './MetricsCards';
import StockChart from './StockChart';
import AIAnalysis from './AIAnalysis';
import RiskMeter from './RiskMeter';
import RecommendationCard from './RecommendationCard';
import SentimentCard from './SentimentCard';
import { RefreshCw, AlertTriangle, Sparkles } from 'lucide-react';

const Dashboard = ({ ticker }) => {
  const [data, setData] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeframe, setTimeframe] = useState('6mo');

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [dashRes, chartRes] = await Promise.all([
          fetchDashboardData(ticker),
          fetchChartData(ticker, timeframe)
        ]);
        setData(dashRes);
        setChartData(chartRes);
      } catch (err) {
        console.error(err);
        setError(err.message || 'Failed to load data. The backend server might be unreachable.');
      } finally {
        setLoading(false);
      }
    };
    
    if (ticker) {
      loadData();
    }
  }, [ticker, timeframe]);

  if (!ticker) {
    return (
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center"
      >
        <div className="w-24 h-24 bg-gradient-to-br from-indigo-500/20 to-cyan-400/20 rounded-full flex items-center justify-center mb-6 shadow-lg shadow-indigo-500/10 border border-indigo-500/20">
          <Sparkles className="w-10 h-10 text-indigo-400" />
        </div>
        <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400 mb-4 tracking-tight">
          Welcome to Nexus AI
        </h2>
        <p className="text-slate-400 max-w-lg text-lg leading-relaxed">
          Search for a company ticker above (e.g., <strong>AAPL</strong>, <strong>MSFT</strong>, <strong>NVDA</strong>) to launch your AI-powered financial analysis dashboard.
        </p>
      </motion.div>
    );
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96">
        <div className="relative">
          <div className="w-16 h-16 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <RefreshCw className="w-6 h-6 text-indigo-400 animate-pulse" />
          </div>
        </div>
        <p className="mt-6 text-slate-400 font-medium animate-pulse">Running AI analysis on {ticker}...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-center">
        <AlertTriangle className="w-16 h-16 text-red-500 mb-4 opacity-80" />
        <h3 className="text-xl font-semibold text-slate-200 mb-2">Analysis Failed</h3>
        <p className="text-slate-400 max-w-md">{error}</p>
        <button 
          onClick={() => window.location.reload()} 
          className="mt-6 btn-primary"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  if (!data) return null;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6 pb-12"
    >
      {/* Header section */}
      <div className="flex items-end justify-between mb-8">
        <div>
          <motion.h2 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-4xl font-bold text-white tracking-tight"
          >
            {data.ticker} <span className="text-xl text-slate-400 font-normal ml-2">{data.company_name}</span>
          </motion.h2>
        </div>
      </div>

      {/* Top row: Metrics */}
      <MetricsCards metrics={data.price_summary} indicators={data.indicators} />

      {/* Middle row: Chart and Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-panel p-5 relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/5 rounded-full blur-3xl -mr-20 -mt-20 transition-all duration-500 group-hover:bg-indigo-500/10"></div>
          <div className="flex items-center justify-between mb-6 relative z-10">
            <h3 className="text-lg font-semibold text-slate-200">Price Action & Moving Averages</h3>
            <div className="flex bg-black/40 p-1 rounded-lg border border-white/5">
              {['1mo', '3mo', '6mo', '1y'].map((tf) => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    timeframe === tf 
                      ? 'bg-indigo-500/20 text-indigo-300' 
                      : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                  }`}
                >
                  {tf.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <div className="h-[350px] relative z-10">
            <StockChart data={chartData} />
          </div>
        </div>
        
        <div className="lg:col-span-1 space-y-6">
          <RecommendationCard recommendation={data.recommendation} />
          <AIAnalysis explanation={data.explanation} />
        </div>
      </div>

      {/* Bottom row: Risk, Sentiment, News */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <RiskMeter prediction={data.ml_prediction} />
        <SentimentCard sentiment={data.sentiment} />
        
        {/* Simple News List */}
        <div className="glass-panel p-5 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500/50 to-cyan-500/50"></div>
          <h3 className="text-lg font-semibold text-slate-200 mb-4">Latest Headlines</h3>
          <div className="space-y-4">
            {data.news && data.news.length > 0 ? (
              data.news.slice(0, 4).map((item, idx) => (
                <div key={idx} className="border-b border-white/5 last:border-0 pb-3 last:pb-0">
                  <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-sm text-slate-300 hover:text-indigo-400 transition-colors line-clamp-2">
                    {item.title}
                  </a>
                  <p className="text-xs text-slate-500 mt-1">{item.source}</p>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500 italic">No recent news found.</p>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default Dashboard;
