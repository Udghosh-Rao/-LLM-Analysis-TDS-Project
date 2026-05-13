import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, DollarSign, BarChart2, Activity, PieChart } from 'lucide-react';

const MetricCard = ({ title, value, subtitle, icon: Icon, isPositive, delay }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className="glass-card p-5 flex items-start justify-between group"
    >
      <div>
        <p className="text-slate-400 text-sm font-medium mb-1">{title}</p>
        <h4 className="text-2xl font-bold text-white mb-2">{value}</h4>
        {subtitle && (
          <div className="flex items-center gap-1.5">
            {isPositive !== undefined && (
              isPositive ? <TrendingUp className="w-3.5 h-3.5 text-emerald-400" /> : <TrendingDown className="w-3.5 h-3.5 text-rose-400" />
            )}
            <span className={`text-xs font-medium ${isPositive === undefined ? 'text-slate-500' : (isPositive ? 'text-emerald-400' : 'text-rose-400')}`}>
              {subtitle}
            </span>
          </div>
        )}
      </div>
      <div className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center border border-white/5 group-hover:scale-110 transition-transform duration-300">
        <Icon className="w-5 h-5 text-slate-300" />
      </div>
    </motion.div>
  );
};

const MetricsCards = ({ metrics, indicators }) => {
  if (!metrics) return null;

  const currentPrice = metrics.current_price ? `$${metrics.current_price.toFixed(2)}` : 'N/A';
  const changePct = metrics.daily_change_pct ? metrics.daily_change_pct.toFixed(2) : 0;
  const isPositive = changePct >= 0;
  
  const formatNumber = (num) => {
    if (!num) return 'N/A';
    if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    return num.toLocaleString();
  };

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
      <MetricCard 
        title="Current Price" 
        value={currentPrice} 
        subtitle={`${isPositive ? '+' : ''}${changePct}% Today`}
        isPositive={isPositive}
        icon={DollarSign}
        delay={0.1}
      />
      <MetricCard 
        title="Market Cap" 
        value={formatNumber(metrics.market_cap)} 
        icon={PieChart}
        delay={0.2}
      />
      <MetricCard 
        title="Volume" 
        value={formatNumber(metrics.volume).replace('$', '')} 
        icon={BarChart2}
        delay={0.3}
      />
      <MetricCard 
        title="RSI (14)" 
        value={indicators?.rsi_14 ? indicators.rsi_14.toFixed(1) : 'N/A'} 
        subtitle={indicators?.rsi_14 > 70 ? 'Overbought' : indicators?.rsi_14 < 30 ? 'Oversold' : 'Neutral'}
        isPositive={indicators?.rsi_14 <= 70 && indicators?.rsi_14 >= 30 ? undefined : (indicators?.rsi_14 < 30)}
        icon={Activity}
        delay={0.4}
      />
      <MetricCard 
        title="Trend Signal" 
        value={(indicators?.trend_signal || 'N/A').toUpperCase()} 
        isPositive={indicators?.trend_signal?.toLowerCase() === 'bullish' ? true : (indicators?.trend_signal?.toLowerCase() === 'bearish' ? false : undefined)}
        icon={indicators?.trend_signal?.toLowerCase() === 'bullish' ? TrendingUp : TrendingDown}
        delay={0.5}
      />
    </div>
  );
};

export default MetricsCards;
