import React, { useState, useRef, useEffect } from 'react';
import { X, Send, Bot, User, Sparkles } from 'lucide-react';
import { sendChatMessage } from '../utils/api';

const ChatAssistant = ({ onClose, contextTicker }) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: contextTicker 
        ? `Hello! I'm your AI financial assistant. I'm currently analyzing ${contextTicker}. What would you like to know?`
        : `Hello! I'm your AI financial assistant. Please search for a stock ticker to begin our analysis.`
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Add a system message when context changes
    setMessages(prev => [
      ...prev,
      {
        role: 'system',
        content: `Context switched to ${contextTicker}`
      }
    ]);
  }, [contextTicker]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await sendChatMessage(userMessage, { ticker: contextTicker });
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: response.response || response.reply || "I couldn't process that request." }
      ]);
    } catch (error) {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Sorry, I encountered an error connecting to the AI brain.' }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col relative bg-[#0f0f16]">
      {/* Background glow */}
      <div className="absolute top-0 right-0 w-full h-32 bg-indigo-500/10 blur-3xl rounded-full"></div>
      
      {/* Header */}
      <div className="p-4 border-b border-white/5 flex justify-between items-center bg-black/20 backdrop-blur-md relative z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Bot className="w-4 h-4 text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-200">AI Assistant</h3>
            <p className="text-xs text-indigo-400 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse"></span>
              Online
            </p>
          </div>
        </div>
        <button 
          onClick={onClose}
          className="p-2 hover:bg-white/5 rounded-full transition-colors text-slate-400 hover:text-white"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth">
        {messages.map((msg, idx) => {
          if (msg.role === 'system') {
            return (
              <div key={idx} className="flex justify-center">
                <span className="text-xs text-slate-500 bg-white/5 px-3 py-1 rounded-full border border-white/5">
                  {msg.content}
                </span>
              </div>
            );
          }
          
          const isUser = msg.role === 'user';
          
          return (
            <div key={idx} className={`flex ${isUser ? 'justify-end' : 'justify-start'} gap-2`}>
              {!isUser && (
                <div className="w-6 h-6 rounded-full bg-indigo-500/20 flex items-center justify-center shrink-0 mt-1">
                  <Bot className="w-3 h-3 text-indigo-400" />
                </div>
              )}
              
              <div 
                className={`max-w-[85%] rounded-2xl p-3 text-sm ${
                  isUser 
                    ? 'bg-indigo-600 text-white rounded-tr-sm' 
                    : 'bg-white/5 text-slate-200 border border-white/5 rounded-tl-sm'
                }`}
              >
                <p className="leading-relaxed whitespace-pre-wrap">{msg.content}</p>
              </div>
              
              {isUser && (
                <div className="w-6 h-6 rounded-full bg-slate-800 flex items-center justify-center shrink-0 mt-1">
                  <User className="w-3 h-3 text-slate-400" />
                </div>
              )}
            </div>
          );
        })}
        
        {isLoading && (
          <div className="flex justify-start gap-2">
            <div className="w-6 h-6 rounded-full bg-indigo-500/20 flex items-center justify-center shrink-0 mt-1">
              <Bot className="w-3 h-3 text-indigo-400" />
            </div>
            <div className="bg-white/5 border border-white/5 rounded-2xl rounded-tl-sm p-4 flex gap-1 items-center">
              <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce"></span>
              <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
              <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 bg-black/20 backdrop-blur-md border-t border-white/5 relative z-10">
        <form onSubmit={handleSubmit} className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about financials, news, trends..."
            className="w-full bg-white/5 border border-white/10 rounded-full pl-4 pr-12 py-3 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
          />
          <button 
            type="submit"
            disabled={!input.trim() || isLoading}
            className="absolute right-1.5 top-1.5 bottom-1.5 aspect-square bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-600/50 disabled:text-white/50 text-white rounded-full flex items-center justify-center transition-colors"
          >
            <Send className="w-4 h-4 ml-0.5" />
          </button>
        </form>
        <div className="mt-2 flex items-center justify-center gap-1 text-[10px] text-slate-500">
          <Sparkles className="w-3 h-3 text-indigo-400" />
          <span>AI can make mistakes. Verify important information.</span>
        </div>
      </div>
    </div>
  );
};

export default ChatAssistant;
