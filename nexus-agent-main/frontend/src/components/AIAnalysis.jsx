import React from 'react';
import { motion } from 'framer-motion';
import { Sparkles, BrainCircuit } from 'lucide-react';

const AIAnalysis = ({ explanation }) => {
  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="glass-panel p-6 relative overflow-hidden group"
    >
      <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
        <BrainCircuit className="w-24 h-24 text-indigo-400" />
      </div>
      
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg">
          <Sparkles className="w-4 h-4 text-white" />
        </div>
        <h3 className="text-lg font-semibold bg-clip-text text-transparent bg-gradient-to-r from-white to-indigo-200">
          AI Insights
        </h3>
      </div>
      
      <div className="relative z-10 prose prose-invert prose-p:text-slate-300 prose-p:leading-relaxed text-sm">
        {explanation ? (
          <p>{explanation}</p>
        ) : (
          <div className="flex flex-col gap-2">
            <div className="h-4 bg-white/10 rounded w-full animate-pulse"></div>
            <div className="h-4 bg-white/10 rounded w-5/6 animate-pulse"></div>
            <div className="h-4 bg-white/10 rounded w-4/6 animate-pulse"></div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default AIAnalysis;
