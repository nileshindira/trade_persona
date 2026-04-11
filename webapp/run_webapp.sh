#!/bin/bash

# Trade Persona Analyzer - WebApp Runner
# This script starts the FastAPI backend and Next.js frontend

# 1. Start Backend
echo "Starting FastAPI Backend..."
cd webapp/backend
# Install requirements if needed
# pip install -r requirements.txt
python3 main.py &
BACKEND_PID=$!

# 2. Start Frontend
echo "Starting Next.js Frontend..."
cd ../frontend
# npm install
npm run dev &
FRONTEND_PID=$!

echo "----------------------------------------------------"
echo "Trade Persona Analyzer is running!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "----------------------------------------------------"

# Trap SIGINT to kill background processes on exit
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT

wait
