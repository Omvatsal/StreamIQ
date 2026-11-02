from fastapi import FastAPI

app = FastAPI(
    title="StreamIQ API",
    description="RL-driven adaptive video streaming platform",
    version="0.1.0"
)


@app.get("/")
async def root():
    return {
        "project": "StreamIQ",
        "status": "running",
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }