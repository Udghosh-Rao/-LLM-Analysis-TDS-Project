import axios from 'axios';

const API_BASE_URL = 'http://localhost:7860';

export const fetchDashboardData = async (ticker) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/dashboard/${ticker}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching dashboard data:', error);
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw error;
  }
};

export const fetchChartData = async (ticker, period = '1y') => {
  try {
    const response = await axios.get(`${API_BASE_URL}/chart/${ticker}?period=${period}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching chart data:', error);
    throw error;
  }
};

export const sendChatMessage = async (message, context = {}) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/chat`, {
      message,
      context,
      session_id: 'default'
    });
    return response.data;
  } catch (error) {
    console.error('Error sending chat message:', error);
    throw error;
  }
};
