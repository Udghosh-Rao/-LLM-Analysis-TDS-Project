import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-card/95 border border-white/10 p-3 rounded-lg shadow-xl backdrop-blur-md">
        <p className="text-slate-400 text-xs mb-2">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="text-sm font-semibold flex items-center gap-2" style={{ color: entry.color }}>
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }}></span>
            {entry.name}: <span className="text-slate-200">${entry.value.toFixed(2)}</span>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

const StockChart = ({ data }) => {
  if (!data || !data.dates || data.dates.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-white/5 rounded-xl border border-white/5">
        <p className="text-slate-500">No chart data available</p>
      </div>
    );
  }

  // Format data for Recharts
  const chartData = data.dates.map((date, index) => ({
    date: date,
    close: data.close[index],
    ma20: data.ma_20[index],
    ma50: data.ma_50[index],
  })).filter(item => item.close !== null);

  const minClose = Math.min(...chartData.map(d => d.close)) * 0.95;
  const maxClose = Math.max(...chartData.map(d => d.close)) * 1.05;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={chartData} margin={{ top: 10, right: 0, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#4f46e5" stopOpacity={0.4}/>
            <stop offset="95%" stopColor="#4f46e5" stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
        <XAxis 
          dataKey="date" 
          stroke="#ffffff40" 
          fontSize={10} 
          tickFormatter={(tick) => {
            const date = new Date(tick);
            return `${date.getMonth() + 1}/${date.getDate()}`;
          }}
          minTickGap={30}
        />
        <YAxis 
          domain={[minClose, maxClose]} 
          stroke="#ffffff40" 
          fontSize={10} 
          tickFormatter={(tick) => `$${tick.toFixed(0)}`}
          orientation="right"
          axisLine={false}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        
        <Area 
          type="monotone" 
          dataKey="close" 
          name="Price"
          stroke="#4f46e5" 
          strokeWidth={2}
          fillOpacity={1} 
          fill="url(#colorClose)" 
        />
        
        {chartData.some(d => d.ma20 !== null) && (
          <Area 
            type="monotone" 
            dataKey="ma20" 
            name="20 DMA"
            stroke="#0ea5e9" 
            strokeWidth={1}
            fill="none" 
            strokeDasharray="5 5"
          />
        )}
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default StockChart;
