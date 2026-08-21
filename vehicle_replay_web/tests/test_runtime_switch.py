from __future__ import annotations

import asyncio
from types import MethodType

from vehicle_replay_web.runtime import ReplayController, ReplayState


def test_select_method_restarts_at_current_cursor_and_preserves_play_state() -> None:
    controller = object.__new__(ReplayController)
    controller.state = ReplayState(
        cursor_s=3360.0,
        window_s=120.0,
        stride_s=60.0,
        waveform_downsample=20,
        playing=False,
        speed=5.0,
        method_id="graph_search",
    )
    calls: list[tuple[float, dict[str, object]]] = []

    async def fake_start(self: ReplayController, start_s: float = 0.0, **kwargs: object) -> None:
        calls.append((start_s, kwargs))

    controller.start = MethodType(fake_start, controller)
    asyncio.run(
        controller.control(
            {
                "action": "select_method",
                "method_id": "kalman_seed",
            }
        )
    )

    assert calls == [
        (
            3360.0,
            {
                "speed": 5.0,
                "window_s": None,
                "stride_s": None,
                "waveform_downsample": None,
                "method_id": "kalman_seed",
                "playing": False,
            },
        )
    ]
