#!/bin/bash

# Start the Backend Server
echo "Starting Backend API on port 7860..."
cd /Users/udghoshrao/Downloads/nexusagent/nexus-agent-main
source venv/bin/activate 2>/dev/null || echo "No virtualenv found, using system python"
python3 -m uvicorn app.api.app:app --host 0.0.0.0 --port 7860 &
BACKEND_PID=$!

# Start the Frontend App
echo "Starting Frontend React App on port 5173..."
cd frontend
npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

echo "============================================================"
echo "Nexus AI Agent is now running!"
echo "Backend API: http://localhost:7860"
echo "Frontend UI: http://localhost:5173"
echo "============================================================"
echo "Press Ctrl+C to stop both servers."

# Trap Ctrl+C and kill both background processes
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
