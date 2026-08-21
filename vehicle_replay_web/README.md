# Vehicle Replay Web

The coordinator and frontend are shared consumers. Model code is launched in
the selected method worker and is never imported into this process.

From the repository root:

```bash
./vehicle_replay_web/scripts/run_web.sh --host 127.0.0.1 --port 8000
```

The canonical launcher is `vehicle_replay_web/scripts/run_web.sh`; the root
旧的根目录启动入口已移除；请使用上面的规范路径。

The service exposes `GET /api/methods`; select one of the four GPU methods in
the dropdown, then use “应用识别参数” to create a new session. Exports are
written to the selected method's `results/web_sessions/<session_id>/`.

Hybrid is the default method.  “展开 Hybrid 参数” exposes the complete
`data`/`model`/`association`/`runtime` configuration from
`methods/hybrid_vehicle_tracker/configs/day11_120s_v9.yaml`; editable values
are passed into the isolated Hybrid worker when “应用识别参数” is clicked.
The three-way large-vehicle separation controls are applied at the same time.

When the default `shared_data/2025-08-09` video directory is present, the page
also lists `2026-05-07 02.mp4` and `2026-05-07 07.mp4`.  They share the replay
cursor, play/pause state, seek position, and playback speed.  Use
`--video-dir <directory>` to select another metadata/video directory.

## Gauss-only field inputs

The two 2025-08-09 field archives can be converted into the current replay
layout with:

```bash
uv run python tools/convert_gauss_zip_to_web_input.py \
  --archive shared_data/2025-08-09/gauss_output_DAY02.zip \
  --output-dir shared_data/2025-08-09/web_input/DAY02

uv run python tools/convert_gauss_zip_to_web_input.py \
  --archive shared_data/2025-08-09/gauss_output_DAY06.zip \
  --output-dir shared_data/2025-08-09/web_input/DAY06
```

Start either dataset by pointing `--cache-dir` at its generated `cache`
directory; the mapping is picked up from the manifest automatically:

```bash
./vehicle_replay_web/scripts/run_web.sh \
  --cache-dir shared_data/2025-08-09/web_input/DAY02/cache
```

The Gauss and matching PRE/pRE archives provide the Gauss and Pre planes. The
matching SAC archive and deployment workbook can add the normalized Raw plane,
after which Hybrid is available too.

## Large-vehicle separation

The web worker can run the three-way separation script's large-vehicle stage
before every recognition window.  The UI exposes the network Pick threshold,
the normalized-RMS large-vehicle threshold, and the split method.  Click
“应用识别参数” after changing them; the selected method then receives the
large-vehicle Gaussian plane only.

When the matching prediction ZIP is available, add it to the generated cache
so Pick threshold is applied to the network probability plane rather than to
Gauss itself:

```bash
uv run python tools/convert_prediction_zip_to_web_input.py \
  --archive shared_data/2025-08-09/PRE.zip \
  --manifest shared_data/2025-08-09/web_input/DAY02/cache/full_day_manifest.json

uv run python tools/convert_prediction_zip_to_web_input.py \
  --archive shared_data/2025-08-09/pRE06.zip \
  --manifest shared_data/2025-08-09/web_input/DAY06/cache/full_day_manifest.json

uv run python tools/convert_sac_zip_to_raw_web_input.py \
  --archive shared_data/2025-08-09/2026-05-07_sca.zip \
  --manifest shared_data/2025-08-09/web_input/DAY02/cache/full_day_manifest.json \
  --workbook shared_data/2025-08-09/415现场布设详表20260312.xlsx
```
