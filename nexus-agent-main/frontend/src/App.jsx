import React, { useState } from 'react';
import { Search, Sparkles } from 'lucide-react';
import Dashboard from './components/Dashboard';
import ChatAssistant from './components/ChatAssistant';

function App() {
  const [tickerInput, setTickerInput] = useState('');
  const [activeTicker, setActiveTicker] = useState('');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const handleSearch = (e) => {
    e.preventDefault();
    if (tickerInput.trim()) {
      setActiveTicker(tickerInput.trim().toUpperCase());
      setTickerInput('');
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-slate-200 flex overflow-hidden">
      {/* Main Content Area */}
      <main className={`flex-1 flex flex-col h-screen transition-all duration-300 ${isSidebarOpen ? 'mr-96' : ''}`}>
        {/* Header */}
        <header className="h-16 border-b border-white/5 bg-black/20 backdrop-blur-md flex items-center justify-between px-6 z-10 sticky top-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
              Nexus AI
            </h1>
          </div>
          
          <form onSubmit={handleSearch} className="relative max-w-md w-full ml-8">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-4 w-4 text-slate-400" />
            </div>
            <input
              type="text"
              className="block w-full pl-10 pr-3 py-2 border border-white/10 rounded-full leading-5 bg-white/5 text-slate-200 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm transition-all duration-300"
              placeholder="Search stocks (e.g. TSLA, AAPL, NVDA)..."
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value)}
            />
          </form>
          
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="btn-primary rounded-full px-5 py-2 text-sm shadow-[0_0_15px_rgba(79,70,229,0.3)] hover:shadow-[0_0_25px_rgba(79,70,229,0.5)] transition-all"
            >
              <Sparkles className="w-4 h-4" />
              Ask AI
            </button>
          </div>
        </header>

        {/* Dashboard Area */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden p-6 scroll-smooth">
          <div className="max-w-7xl mx-auto">
            <Dashboard ticker={activeTicker} />
          </div>
        </div>
      </main>

      {/* AI Assistant Sidebar */}
      <div 
        className={`fixed inset-y-0 right-0 w-96 bg-[#0f0f16] border-l border-white/5 transform transition-transform duration-300 ease-in-out z-20 shadow-2xl ${
          isSidebarOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <ChatAssistant 
          onClose={() => setIsSidebarOpen(false)} 
          contextTicker={activeTicker}
        />
      </div>
    </div>
  );
}

export default App;
