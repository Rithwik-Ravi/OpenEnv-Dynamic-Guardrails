from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import os
import uvicorn
import logging

# Disable uvicorn access logs to prevent stdout pollution for OpenEnv
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(title="Dynamic Guardrail Env")


METRICS_FILE = "metrics.jsonl"

async def event_generator(request: Request):
    if not os.path.exists(METRICS_FILE):
        open(METRICS_FILE, "w").close()
        
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        while True:
            if await request.is_disconnected():
                break
                
            line = f.readline()
            if not line:
                await asyncio.sleep(0.1)
                continue
            
            yield f"data: {line}\n\n"

@app.get("/stream")
async def stream(request: Request):
    return StreamingResponse(event_generator(request), media_type="text/event-stream")

@app.get("/")
async def get_index():
    ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {"detail": "Frontend not found"}

if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, log_level="warning")
