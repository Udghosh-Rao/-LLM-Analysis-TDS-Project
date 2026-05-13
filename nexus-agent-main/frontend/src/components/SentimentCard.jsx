import React from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, ThumbsUp, ThumbsDown, Minus } from 'lucide-react';

const SentimentCard = ({ sentiment }) => {
  if (!sentiment) return null;

  const { positive_pct = 0, negative_pct = 0, neutral_pct = 0, verdict = 'neutral' } = sentiment;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.4 }}
      className="glass-panel p-6 flex flex-col justify-between"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-slate-200">News Sentiment</h3>
        <MessageSquare className="w-5 h-5 text-indigo-400" />
      </div>

      <div className="flex-1 flex flex-col justify-center">
        {/* Sentiment Bars */}
        <div className="space-y-4 w-full">
          <div className="flex items-center gap-3">
            <ThumbsUp className="w-4 h-4 text-emerald-400 shrink-0" />
            <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${positive_pct}%` }}
                transition={{ duration: 1 }}
                className="h-full bg-emerald-400 rounded-full"
              />
            </div>
            <span className="text-xs text-slate-400 w-8 text-right">{positive_pct}%</span>
          </div>
          
          <div className="flex items-center gap-3">
            <Minus className="w-4 h-4 text-slate-400 shrink-0" />
            <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${neutral_pct}%` }}
                transition={{ duration: 1 }}
                className="h-full bg-slate-400 rounded-full"
              />
            </div>
            <span className="text-xs text-slate-400 w-8 text-right">{neutral_pct}%</span>
          </div>

          <div className="flex items-center gap-3">
            <ThumbsDown className="w-4 h-4 text-rose-400 shrink-0" />
            <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${negative_pct}%` }}
                transition={{ duration: 1 }}
                className="h-full bg-rose-400 rounded-full"
              />
            </div>
            <span className="text-xs text-slate-400 w-8 text-right">{negative_pct}%</span>
          </div>
        </div>

        <div className="mt-6 text-center">
          <span className="text-sm text-slate-400">Overall Verdict: </span>
          <span className={`font-semibold capitalize ${
            verdict === 'positive' ? 'text-emerald-400' : 
            verdict === 'negative' ? 'text-rose-400' : 'text-slate-300'
          }`}>
            {verdict}
          </span>
        </div>
      </div>
    </motion.div>
  );
};

export default SentimentCard;
