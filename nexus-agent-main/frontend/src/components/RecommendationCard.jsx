import React from 'react';
import { motion } from 'framer-motion';
import { Target } from 'lucide-react';

const RecommendationCard = ({ recommendation }) => {
  if (!recommendation) return null;

  const { label = 'HOLD', confidence = 0, reasons = [] } = recommendation;
  
  let colorClass = 'from-slate-400 to-slate-500';
  let glowClass = 'shadow-slate-500/20';
  let badgeClass = 'bg-slate-500/20 text-slate-300 border-slate-500/30';
  
  if (label.toUpperCase() === 'BUY') {
    colorClass = 'from-emerald-400 to-teal-500';
    glowClass = 'shadow-emerald-500/20';
    badgeClass = 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';
  } else if (label.toUpperCase() === 'SELL') {
    colorClass = 'from-rose-400 to-red-500';
    glowClass = 'shadow-rose-500/20';
    badgeClass = 'bg-rose-500/20 text-rose-300 border-rose-500/30';
  } else if (label.toUpperCase() === 'HOLD' || label.toUpperCase() === 'WATCH') {
    colorClass = 'from-amber-400 to-orange-500';
    glowClass = 'shadow-amber-500/20';
    badgeClass = 'bg-amber-500/20 text-amber-300 border-amber-500/30';
  }

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="glass-panel p-6 relative overflow-hidden"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-200">Recommendation</h3>
        <Target className="w-5 h-5 text-indigo-400" />
      </div>

      <div className="flex items-center gap-6 mb-6">
        <div className={`relative px-6 py-3 rounded-xl border ${badgeClass} shadow-lg ${glowClass} flex items-center justify-center`}>
          <span className="text-3xl font-black tracking-wider bg-clip-text text-transparent bg-gradient-to-br from-white to-white/70">
            {label.toUpperCase()}
          </span>
        </div>
        
        <div>
          <p className="text-sm text-slate-400 mb-1">Confidence Score</p>
          <div className="text-2xl font-bold text-white">{confidence}%</div>
        </div>
      </div>

      {reasons && reasons.length > 0 && (
        <div>
          <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold mb-2">Key Drivers</p>
          <ul className="space-y-2">
            {reasons.map((reason, idx) => (
              <li key={idx} className="text-sm text-slate-300 flex items-start gap-2">
                <span className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 bg-gradient-to-r ${colorClass}`} />
                {reason}
              </li>
            ))}
          </ul>
        </div>
      )}
    </motion.div>
  );
};

export default RecommendationCard;
