import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import json
import uvicorn
import os

app = FastAPI(title="Telemetry UI Server")

METRICS_FILE = "metrics.jsonl"

async def event_generator(request: Request):
    if not os.path.exists(METRICS_FILE):
        open(METRICS_FILE, "w").close()
        
    with open(METRICS_FILE, "r") as f:
        # We process from the beginning of the file to load historical chart lines rapidly,
        # then we tail the file infinitely.
        while True:
            # Gracefully handle client connection loss
            if await request.is_disconnected():
                break
                
            line = f.readline()
            if not line:
                await asyncio.sleep(0.1) # Wait for new content from evaluate.py
                continue
            
            # Send Server-Sent Event formatted correctly for JS consumers
            yield f"data: {line}\n\n"

@app.get("/stream")
async def stream(request: Request):
    return StreamingResponse(event_generator(request), media_type="text/event-stream")

@app.get("/")
async def get_index():
    with open("src/ui/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    uvicorn.run("src.ui.dashboard:app", host="127.0.0.1", port=8001, log_level="info")
