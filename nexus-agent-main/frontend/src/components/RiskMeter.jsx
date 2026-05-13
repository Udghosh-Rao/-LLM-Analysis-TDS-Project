import React from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, ShieldCheck, Shield } from 'lucide-react';

const RiskMeter = ({ prediction }) => {
  if (!prediction) return null;

  const riskScore = prediction.risk_score || 0;
  const isAnomaly = prediction.anomaly;
  
  // 0.0 to 1.0 -> 0% to 100%
  const riskPercentage = Math.round(riskScore * 100);
  
  let riskLevel = 'Low';
  let riskColor = 'from-emerald-400 to-green-500';
  let Icon = ShieldCheck;
  
  if (riskPercentage > 66) {
    riskLevel = 'High';
    riskColor = 'from-rose-400 to-red-500';
    Icon = ShieldAlert;
  } else if (riskPercentage > 33) {
    riskLevel = 'Medium';
    riskColor = 'from-amber-400 to-orange-500';
    Icon = Shield;
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="glass-panel p-6 flex flex-col justify-between"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-slate-200">Risk Assessment</h3>
        <Icon className={`w-6 h-6 ${riskLevel === 'High' ? 'text-rose-400' : riskLevel === 'Medium' ? 'text-amber-400' : 'text-emerald-400'}`} />
      </div>

      <div className="flex-1 flex flex-col items-center justify-center relative py-4">
        {/* Simple gauge UI */}
        <div className="relative w-full h-4 bg-white/10 rounded-full overflow-hidden mb-4">
          <motion.div 
            initial={{ width: 0 }}
            animate={{ width: `${riskPercentage}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
            className={`absolute top-0 left-0 h-full bg-gradient-to-r ${riskColor}`}
          />
        </div>
        
        <div className="flex w-full justify-between text-xs text-slate-500 mb-6 font-medium">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>

        <div className="text-center">
          <div className={`text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r ${riskColor}`}>
            {riskLevel} Risk
          </div>
          <p className="text-slate-400 text-sm mt-1">
            Score: {riskPercentage}% {isAnomaly ? '• Anomaly Detected' : ''}
          </p>
        </div>
      </div>
    </motion.div>
  );
};

export default RiskMeter;
