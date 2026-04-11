#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Stockk Persona Analyzer - WebApp   ${NC}"
echo -e "${BLUE}========================================${NC}"

# Get the script direction
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# 1. Start the Backend
echo -e "${GREEN}1. Launching Diagnostic Backend (FastAPI)...${NC}"
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run backend in background
cd webapp/backend
python3 api.py &
BACKEND_PID=$!
echo -e "   [PID: $BACKEND_PID] Backend is starting on http://localhost:8100"

# 2. Start the Frontend
echo -e "${GREEN}2. Launching Persona Lab (Next.js)...${NC}"
cd ../frontend

# Kill backend on script exit
trap "echo -e '${BLUE}Shutting down...${NC}'; kill $BACKEND_PID; exit" SIGINT SIGTERM EXIT

npm run dev
