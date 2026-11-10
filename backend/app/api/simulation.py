"""Simulation API endpoints for batch comparison and WebSocket streaming."""

import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.app.services.sim_service import sim_service

router = APIRouter(prefix="/simulation", tags=["Simulation"])


class BatchSimulationRequest(BaseModel):
    profile: str = Field(default="sudden_drop", description="Network trace profile name")
    duration_seconds: float = Field(default=60.0, ge=10.0, le=300.0)
    interval_seconds: float = Field(default=1.0, ge=0.5, le=5.0)
    seed: Optional[int] = Field(default=42)
    policies: Optional[List[str]] = Field(
        default=["dqn", "buffer_based", "throughput_based", "fixed_240p"],
        description="List of policies to compare",
    )


@router.post("/batch")
async def run_batch_simulation(request: BatchSimulationRequest) -> Dict[str, Any]:
    """Run full synchronous multi-policy comparative simulation."""
    return sim_service.run_batch(
        profile=request.profile,
        duration_seconds=request.duration_seconds,
        interval_seconds=request.interval_seconds,
        seed=request.seed,
        policies=request.policies,
    )


@router.websocket("/live")
async def live_simulation_websocket(websocket: WebSocket):
    """Real-time bi-directional WebSocket session for interactive streaming simulation."""
    await websocket.accept()
    try:
        # Wait for initialization parameters from client
        data = await websocket.receive_text()
        params = json.loads(data)

        profile = params.get("profile", "sudden_drop")
        duration = float(params.get("duration_seconds", 60.0))
        interval = float(params.get("interval_seconds", 1.0))
        seed = params.get("seed", 42)
        policies = params.get("policies", ["dqn", "buffer_based", "throughput_based"])
        step_delay = float(params.get("step_delay", 0.15))

        async for step_message in sim_service.stream_live(
            profile=profile,
            duration_seconds=duration,
            interval_seconds=interval,
            seed=seed,
            policies=policies,
            step_delay=step_delay,
        ):
            await websocket.send_text(json.dumps(step_message))

    except WebSocketDisconnect:
        pass
    except Exception as error:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(error)}))
        except Exception:
            pass
        finally:
            await websocket.close()

