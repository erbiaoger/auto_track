from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .runtime import ReplayController


def create_app(
    controller: ReplayController,
    *,
    frontend_dist: str | Path | None = None,
    video_dir: str | Path | None = None,
    access_token: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await controller.ensure_worker()
        if not controller.state.session_id:
            try:
                await controller.start(0.0)
            except Exception as exc:
                # A missing environment/checkpoint must make only that method
                # unavailable; keep the health/API page alive so the user can
                # select Classic Graph Search or inspect the reason.
                controller.state.session_id = f"startup-{uuid.uuid4().hex[:8]}"
                controller.state.playing = False
                controller.state.error = str(exc)
        yield
        await controller.close()

    app = FastAPI(title="Vehicle Recognition Replay", version="0.2.0", lifespan=lifespan)

    if video_dir is not None and Path(video_dir).is_dir():
        video_root = Path(video_dir).resolve()
        app.mount("/media", StaticFiles(directory=video_root), name="media")

        @app.get("/api/videos")
        async def videos(request: Request) -> dict[str, object]:
            check_token(request)
            rows = []
            for path in sorted(video_root.iterdir()):
                if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
                    rows.append({"id": path.name, "label": path.stem, "url": f"/media/{path.name}", "size_bytes": path.stat().st_size})
            return {"videos": rows}

    def check_token(request: Request) -> None:
        if access_token and request.headers.get("x-hvt-token") != access_token:
            raise HTTPException(status_code=401, detail="invalid access token")

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, object]:
        check_token(request)
        tracker = controller.tracker
        active_worker = getattr(controller.method_manager, "active", None)
        device = str(tracker.device) if tracker is not None else str(getattr(active_worker, "device", getattr(controller.method_manager, "device", "worker")))
        requested_cuda = bool(device.startswith("cuda"))
        cuda_available = bool(__import__("torch").cuda.is_available()) if requested_cuda else False
        return {
            "service": "ok",
            "method_id": controller.state.method_id,
            "preset_id": controller.state.preset_id,
            "protocol": "method-worker/v1",
            "model_loaded": bool(tracker.has_checkpoint) if tracker is not None else bool(controller.method_manager and controller.method_manager.active is not None),
            "device": device,
            "cuda_available": cuda_available,
            "cuda_device": __import__("torch").cuda.get_device_name(0) if cuda_available else None,
            "cache": {
                "manifest": str(getattr(controller.source, "manifest_path", "")),
                "duration_s": controller.source.duration_s,
                "station_count": controller.source.station_count,
                "sample_rate_hz": controller.source.sample_rate_hz,
                "station_positions_m": controller.snapshot()["station_positions_m"],
            },
            "subscribers": len(controller._subscribers),
            "methods": controller.method_manager.descriptors() if controller.method_manager is not None else [],
        }

    @app.get("/api/methods")
    async def methods(request: Request) -> dict[str, object]:
        check_token(request)
        return {
            "active_method_id": controller.state.method_id,
            "active_preset_id": controller.state.preset_id,
            "methods": controller.method_manager.descriptors() if controller.method_manager is not None else [],
        }

    @app.get("/api/replay/state")
    async def replay_state(request: Request) -> dict[str, object]:
        check_token(request)
        return controller.snapshot()

    @app.post("/api/replay/start")
    async def replay_start(request: Request) -> dict[str, object]:
        check_token(request)
        payload = await request.json()
        await controller.start(
            float(payload.get("start_s", 0.0)),
            speed=payload.get("speed"),
            window_s=float(payload["window_s"]) if payload.get("window_s") is not None else None,
            stride_s=float(payload["stride_s"]) if payload.get("stride_s") is not None else None,
            waveform_downsample=int(payload["waveform_downsample"]) if payload.get("waveform_downsample") is not None else None,
            method_id=str(payload["method_id"]) if payload.get("method_id") is not None else None,
            separation=dict(payload["separation"]) if isinstance(payload.get("separation"), dict) else None,
            hybrid_config=dict(payload["hybrid_config"]) if isinstance(payload.get("hybrid_config"), dict) else None,
        )
        return controller.snapshot()

    @app.patch("/api/replay/control")
    async def replay_control(request: Request) -> dict[str, object]:
        check_token(request)
        await controller.control(await request.json())
        return controller.snapshot()

    @app.post("/api/replay/export")
    async def replay_export(request: Request) -> dict[str, object]:
        check_token(request)
        return await controller.export()

    @app.post("/api/replay/separation-plot")
    async def replay_separation_plot(request: Request) -> FileResponse:
        check_token(request)
        payload = await request.json()
        separation = dict(payload["separation"]) if isinstance(payload.get("separation"), dict) else None
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(
            controller._executor,
            controller.render_separation_plot,
            float(payload.get("start_s", 0.0)),
            float(payload.get("duration_s", 600.0)),
            separation,
        )
        return FileResponse(path, media_type="image/png", filename=path.name)

    @app.websocket("/ws/replay")
    async def replay_socket(websocket: WebSocket) -> None:
        if access_token and websocket.query_params.get("token") != access_token:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        controller.subscribe(websocket)
        try:
            await websocket.send_json({"event": "status", **controller.snapshot()})
            while True:
                message = await websocket.receive_json()
                if isinstance(message, dict):
                    await controller.control(message)
        except WebSocketDisconnect:
            pass
        finally:
            controller.unsubscribe(websocket)

    if frontend_dist:
        dist = Path(frontend_dist)
        if dist.exists():
            assets = dist / "assets"
            if assets.exists():
                app.mount("/assets", StaticFiles(directory=assets), name="assets")

            @app.get("/{path:path}")
            async def frontend(path: str) -> Any:
                candidate = (dist / path).resolve()
                if path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
                    return FileResponse(candidate)
                index = dist / "index.html"
                if index.exists():
                    return FileResponse(index)
                return HTMLResponse("<h1>Hybrid Vehicle Tracker</h1><p>Frontend not built.</p>")

    return app
