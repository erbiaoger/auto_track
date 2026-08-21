import { useEffect, useMemo, useRef, useState } from 'react'
import * as echarts from 'echarts'

type Point = { channel_index: number; station_id: string; position_m: number; time_s: number; observed: boolean; ambiguous?: boolean }
type Track = { global_vehicle_id: string; points: Point[]; median_speed_kmh: number; confidence: number | null; observed_count: number; span_m: number; max_gap_m: number; enters_window: boolean; exits_window: boolean; ambiguous_crossing: boolean }
type Counters = { window_candidates: number; window_unique?: number; tentative_tracks?: number; gap_reconnections?: number; id_merges?: number; current_unique: number; cumulative_unique: number; mean_speed_kmh: number | null; median_speed_kmh: number | null; speed_range_kmh: [number|null, number|null] }
type Frame = { times: number[]; gauss: number[]; largeGauss: number[]; rawMin: number[]; rawMax: number[]; rawWave: number[]; waveformRate: number; stationCount: number; start: number; duration: number }
type MethodDescriptor = { method_id: string; label: string; description: string; available: boolean; reason?: string; preset_id: string; checkpoint?: string; version?: string; device?: string; engine?: string }
type HybridConfig = Record<string, Record<string, unknown>>
type VideoItem = { id: string; label: string; url: string; duration_s?: number }

const COLORS = ['#42d3ff','#ffbf69','#ff6b9d','#9bff8a','#c8a7ff','#f4f36b','#ff8c69','#55e6c1','#7aa7ff','#f78fb3','#63cdda','#e77f67']
const VIDEO_CHANNEL_INDEX = 40
const VIDEO_OFFSET_M = 4000
const colorForId = (id: string) => { let hash = 0; for (const char of id) hash = (hash * 31 + char.charCodeAt(0)) >>> 0; return COLORS[hash % COLORS.length] }

function decodeFrame(buffer: ArrayBuffer): Frame | null {
  const bytes = new Uint8Array(buffer)
  if (bytes.length < 8 || String.fromCharCode(...bytes.slice(0, 4)) !== 'HVT1') return null
  const view = new DataView(buffer)
  const headerLength = view.getUint32(4, true)
  const header = JSON.parse(new TextDecoder().decode(bytes.slice(8, 8 + headerLength)))
  const lengths: number[] = header.array_lengths ?? [header.times_s.length * header.station_count, header.times_s.length * header.station_count, header.times_s.length * header.station_count]
  const names: string[] = header.array_names ?? ['gauss', 'raw_min', 'raw_max', 'raw_wave']
  // JSON header length is not guaranteed to be 4-byte aligned; slice first so
  // the Float32Array constructor always receives an aligned byte offset.
  const payload = new Float32Array(buffer.slice(8 + headerLength))
  let offset = 0
  const take = (length: number) => { const values = Array.from(payload.slice(offset, offset + length)); offset += length; return values }
  const arrays: Record<string, number[]> = {}
  names.forEach((name, index) => { arrays[name] = take(lengths[index] ?? 0) })
  return { times: header.times_s, gauss: arrays.gauss ?? [], largeGauss: arrays.large_gauss ?? [], rawMin: arrays.raw_min ?? [], rawMax: arrays.raw_max ?? [], rawWave: arrays.raw_wave ?? [], waveformRate: Number(header.waveform_rate_hz ?? 0), stationCount: header.station_count, start: header.start_s, duration: header.duration_s }
}

function App() {
  const chartRef = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)
  const frames = useRef<Frame[]>([])
  const [tracks, setTracks] = useState<Record<string, Track>>({})
  const [counters, setCounters] = useState<Counters>({ window_candidates: 0, current_unique: 0, cumulative_unique: 0, mean_speed_kmh: null, median_speed_kmh: null, speed_range_kmh: [null, null] })
  const [status, setStatus] = useState<any>({ state: { cursor_s: 0, playing: false, speed: 1 }, station_positions_m: [] })
  const [mode, setMode] = useState<'gauss'|'large'|'raw'|'both'>('large')
  const [viewMode, setViewMode] = useState<'tracks'|'separation'>('tracks')
  const [separationImageUrl, setSeparationImageUrl] = useState<string | null>(null)
  const [loadingSeparation, setLoadingSeparation] = useState(false)
  const [speed, setSpeed] = useState<number|string>(1)
  const [selected, setSelected] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const [inference, setInference] = useState<number | null>(null)
  const [windowDuration, setWindowDuration] = useState(120)
  const [strideDuration, setStrideDuration] = useState(60)
  const [waveformDownsample, setWaveformDownsample] = useState(20)
  const [separationEnabled, setSeparationEnabled] = useState(true)
  const [pickThreshold, setPickThreshold] = useState(0.5)
  const [largeThreshold, setLargeThreshold] = useState(0.35)
  const [splitPercentile, setSplitPercentile] = useState(75)
  const [minGap, setMinGap] = useState(3)
  const [energyWindow, setEnergyWindow] = useState(1)
  const [gaussianWidth, setGaussianWidth] = useState(0.5)
  const [amplitudeMin, setAmplitudeMin] = useState(0.25)
  const [amplitudeMax, setAmplitudeMax] = useState(1)
  const [splitMethod, setSplitMethod] = useState<'threshold'|'percentile'|'kmeans'>('percentile')
  const [methods, setMethods] = useState<MethodDescriptor[]>([])
  const [selectedMethod, setSelectedMethod] = useState('hybrid')
  const [switchingMethod, setSwitchingMethod] = useState(false)
  const [hybridConfig, setHybridConfig] = useState<HybridConfig>({})
  const [showHybridConfig, setShowHybridConfig] = useState(false)
  const [videos, setVideos] = useState<VideoItem[]>([])
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({})
  const videoLastSeekAt = useRef<Record<string, number>>({})
  const displayVideos = useMemo(() => {
    const candidates = videos.filter((video) => /02(?:\.web)?\.mp4$/i.test(video.id))
    const browserCopy = candidates.find((video) => video.id.toLowerCase().includes('.web.'))
    return browserCopy ? [browserCopy] : candidates.slice(0, 1)
  }, [videos])
  const sessionRef = useRef('')
  const windowDurationRef = useRef(windowDuration)
  windowDurationRef.current = windowDuration

  const hasRaw = status.data_capabilities?.has_raw ?? true

  const send = (payload: object) => wsRef.current?.send(JSON.stringify(payload))
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/replay`)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onmessage = (event) => {
      if (typeof event.data !== 'string') {
        const frame = decodeFrame(event.data)
        if (frame) {
          frames.current = [...frames.current.filter((item) => item.start >= frame.start - windowDurationRef.current), frame]
          setStatus((old: any) => ({ ...old, state: { ...old.state, cursor_s: frame.start + frame.duration, buffer_start_s: Math.max(0, frame.start + frame.duration - (old.state?.window_s ?? windowDuration)), buffer_end_s: frame.start + frame.duration } }))
        }
        return
      }
      const message = JSON.parse(event.data)
      if (message.event === 'status' || message.event === 'end') {
        const isInitialSession = !sessionRef.current
        const incomingSession = String(message.state?.session_id ?? '')
        const isNewSession = Boolean(incomingSession && incomingSession !== sessionRef.current)
        if (incomingSession && sessionRef.current && isNewSession) {
          frames.current = []
          setTracks({})
          setCounters({ window_candidates: 0, current_unique: 0, cumulative_unique: 0, mean_speed_kmh: null, median_speed_kmh: null, speed_range_kmh: [null, null] })
          setSelected(null)
          setInference(null)
        }
        if (incomingSession) sessionRef.current = incomingSession
        if (Array.isArray(message.methods)) setMethods(message.methods)
        // Do not overwrite a user's pending dropdown choice with status frames
        // from the old session.  Sync only on the initial status or after the
        // server has confirmed a newly-created session.
        if (message.state?.method_id && (isInitialSession || isNewSession)) setSelectedMethod(String(message.state.method_id))
        if (isNewSession) setSwitchingMethod(false)
        setStatus(message)
        if (message.state?.window_s != null) setWindowDuration(Number(message.state.window_s))
        if (message.state?.stride_s != null) setStrideDuration(Number(message.state.stride_s))
        if (message.state?.waveform_downsample != null) setWaveformDownsample(Number(message.state.waveform_downsample))
        const separation = message.large_vehicle_separation ?? message.state?.large_vehicle_separation
        if (separation && (isInitialSession || isNewSession)) {
          if (separation.enabled != null) setSeparationEnabled(Boolean(separation.enabled))
          if (separation.pick_threshold != null) setPickThreshold(Number(separation.pick_threshold))
          if (separation.split_threshold != null) setLargeThreshold(Number(separation.split_threshold))
          if (separation.split_percentile != null) setSplitPercentile(Number(separation.split_percentile))
          if (separation.min_gap_s != null) setMinGap(Number(separation.min_gap_s))
          if (separation.energy_window_s != null) setEnergyWindow(Number(separation.energy_window_s))
          if (separation.gaussian_width_s != null) setGaussianWidth(Number(separation.gaussian_width_s))
          if (separation.amplitude_min != null) setAmplitudeMin(Number(separation.amplitude_min))
          if (separation.amplitude_max != null) setAmplitudeMax(Number(separation.amplitude_max))
          if (separation.split_method) setSplitMethod(separation.split_method)
        }
        if (message.hybrid_config && Object.keys(message.hybrid_config).length && (isInitialSession || isNewSession)) setHybridConfig(message.hybrid_config)
      }
      if (message.event === 'prediction' || message.event === 'vehicle_update') {
        if (message.counters) setCounters(message.counters)
        if (message.elapsed_s != null) setInference(message.elapsed_s)
        if (message.tracks && (!message.session_id || message.session_id === sessionRef.current)) {
          const aliases: Record<string, string> = message.id_aliases ?? {}
          const removed = new Set<string>(message.removed_track_ids ?? [])
          setTracks((old) => {
            const next = { ...old }
            Object.entries(aliases).forEach(([oldId, canonicalId]) => {
              if (next[oldId] && !next[canonicalId]) next[canonicalId] = { ...next[oldId], global_vehicle_id: canonicalId }
              delete next[oldId]
            })
            removed.forEach((id) => delete next[id])
            message.tracks.forEach((item: Track) => { next[item.global_vehicle_id] = item })
            return next
          })
          setSelected((current) => current && aliases[current] ? aliases[current] : current)
        }
      }
      if (message.event === 'error') setSwitchingMethod(false)
    }
    return () => { ws.close(); wsRef.current = null }
  }, [])

  const positions: number[] = status.station_positions_m?.length ? status.station_positions_m : Array.from({ length: 50 }, (_, i) => i * 100)
  const cursor = status.state?.cursor_s ?? 0
  const state = status.state ?? {}

  useEffect(() => {
    fetch('/api/videos').then((response) => response.ok ? response.json() : Promise.reject(new Error('video catalog unavailable')))
      .then((payload) => setVideos(Array.isArray(payload.videos) ? payload.videos : []))
      .catch(() => setVideos([]))
  }, [])

  useEffect(() => {
    const rate = speed === 'fastest' ? 16 : Math.max(0.25, Math.min(16, Number(speed) || 1))
    const playing = Boolean(state.playing)
    Object.values(videoRefs.current).forEach((video) => {
      if (!video) return
      video.playbackRate = rate
      if (playing) void video.play().catch(() => undefined)
      else video.pause()
    })
  }, [speed, status.state?.playing, displayVideos])

  useEffect(() => {
    // Let the media element advance naturally while the replay is playing.
    // Re-seeking on every cursor update causes a new Range request for this
    // large MP4 and can leave hundreds of requests open in the dev server.
    if (state.playing) return
    const requested = cursor
    const now = Date.now()
    Object.entries(videoRefs.current).forEach(([videoId, video]) => {
      if (!video || !Number.isFinite(video.duration)) return
      const target = Math.min(requested, Math.max(0, video.duration - 0.05))
      if (Math.abs(video.currentTime - target) > 0.45 && now - (videoLastSeekAt.current[videoId] ?? 0) >= 250) {
        videoLastSeekAt.current[videoId] = now
        video.currentTime = target
      }
    })
  }, [cursor, status.state?.playing, displayVideos])

  const plotTracks = useMemo(() => Object.values(tracks).filter((track) => track.points.some((point) => point.time_s >= Math.max(0, cursor - windowDuration) && point.time_s <= cursor + 1)), [tracks, cursor, windowDuration])
  const displayFrames = frames.current

  useEffect(() => {
    if (viewMode === 'separation') return
    if (!chartRef.current) return
    chart.current ??= echarts.init(chartRef.current)
    const xMin = Math.min(...positions), xMax = Math.max(...positions)
    const yMin = Math.max(0, cursor - windowDuration), yMax = Math.max(windowDuration, cursor)
    const heatmap: any[] = []
    const largeHeatmap: any[] = []
    const rawHeatmap: any[] = []
    const waveform: Array<Array<[number, number]>> = Array.from({ length: positions.length }, () => [])
    for (const frame of frames.current) {
      frame.times.forEach((time, t) => { for (let station = 0; station < frame.stationCount; station++) { const value = frame.gauss[t * frame.stationCount + station]; if (value > 0.005) heatmap.push([positions[station], time, value]) } })
      if (frame.largeGauss.length > 0) frame.times.forEach((time, t) => { for (let station = 0; station < frame.stationCount; station++) { const value = frame.largeGauss[t * frame.stationCount + station]; if (value > 0.005) largeHeatmap.push([positions[station], time, value]) } })
      if (mode !== 'gauss') frame.times.forEach((time, t) => { for (let station = 0; station < frame.stationCount; station++) { const value = frame.rawMax[t * frame.stationCount + station]; const low = frame.rawMin[t * frame.stationCount + station]; rawHeatmap.push([positions[station], time, Math.min(1, Math.abs(value - low) * 0.15)]) } })
      if (mode !== 'gauss' && frame.rawWave.length > 0 && frame.waveformRate > 0) {
        const waveformSamples = Math.floor(frame.rawWave.length / frame.stationCount)
        for (let t = 0; t < waveformSamples; t++) {
          const time = frame.start + t / frame.waveformRate
          for (let station = 0; station < frame.stationCount; station++) {
            const value = Math.max(-1, Math.min(1, frame.rawWave[t * frame.stationCount + station] ?? 0))
            const spacing = positions.length > 1 ? (station === 0 ? (positions[1] - positions[0]) : (positions[station] - positions[station - 1])) : 100
            waveform[station].push([positions[station] + value * spacing * 0.28, time])
          }
        }
      }
    }
    const series: any[] = []
    if (mode === 'gauss' || mode === 'both') series.push({ type: 'heatmap', data: heatmap, progressive: 5000, itemStyle: { opacity: 0.72 }, coordinateSystem: 'cartesian2d' })
    if (mode === 'large') series.push({ type: 'heatmap', data: largeHeatmap.length ? largeHeatmap : heatmap, progressive: 5000, itemStyle: { opacity: 0.72 }, coordinateSystem: 'cartesian2d' })
    if (mode === 'raw' || mode === 'both') series.push({ type: 'heatmap', data: rawHeatmap, progressive: 5000, itemStyle: { opacity: 0.3 }, coordinateSystem: 'cartesian2d' })
    if (mode === 'raw' || mode === 'both') waveform.forEach((values, station) => { if (values.length) series.push({ type: 'line', data: values, showSymbol: false, connectNulls: false, lineStyle: { color: '#b8d7ff', width: 0.7, opacity: 0.55 }, progressive: 10000, large: true, silent: true }) })
    plotTracks.forEach((track) => {
      const color = colorForId(track.global_vehicle_id)
      const points = track.points.filter((point) => point.time_s >= yMin - 2 && point.time_s <= yMax + 2)
      series.push({ type: 'line', name: track.global_vehicle_id, data: points.map((point) => [point.position_m, point.time_s]), showSymbol: false, lineStyle: { color, width: selected === track.global_vehicle_id ? 4 : 2 }, endLabel: { show: true, formatter: `${track.global_vehicle_id}  ${track.median_speed_kmh.toFixed(1)} km/h`, color, fontWeight: 'bold' }, silent: true })
      series.push({ type: 'scatter', data: points.filter((point) => point.observed).map((point) => [point.position_m, point.time_s]), symbol: 'rect', symbolSize: selected === track.global_vehicle_id ? 10 : 7, itemStyle: { color }, silent: true })
      series.push({ type: 'scatter', data: points.filter((point) => !point.observed).map((point) => [point.position_m, point.time_s]), symbol: 'circle', symbolSize: 9, itemStyle: { color: 'transparent', borderColor: color, borderWidth: 2 }, silent: true })
    })
    chart.current.setOption({ animation: false, grid: { left: 58, right: 120, top: 18, bottom: 46 }, tooltip: { trigger: 'item' }, xAxis: { type: 'value', min: xMin, max: xMax, name: '物理位置 (m)', nameLocation: 'middle', nameGap: 32, axisLabel: { color: '#a9b9d2' }, splitLine: { lineStyle: { color: '#22314b' } } }, yAxis: { type: 'value', min: yMin, max: yMax, inverse: true, name: '绝对时间 (s)', axisLabel: { color: '#a9b9d2' }, splitLine: { lineStyle: { color: '#22314b' } } }, visualMap: { show: false, min: 0, max: 1, inRange: { color: ['#08111f', '#165d87', '#42d3ff', '#ffe66d'] } }, series }, true)
  }, [positions, cursor, mode, plotTracks, selected, windowDuration, viewMode])

  const speedRange = counters.speed_range_kmh ?? [null, null]
  const activeMethod = methods.find((item) => item.method_id === (state.method_id ?? selectedMethod))
  const methodLabel = activeMethod?.label ?? (state.method_id ?? selectedMethod)
  useEffect(() => {
    if (!hasRaw && (mode === 'raw' || mode === 'both')) setMode('large')
  }, [hasRaw, mode])
  const separationPayload = { enabled: separationEnabled, pick_threshold: pickThreshold, min_gap_s: minGap, energy_window_s: energyWindow, split_method: splitMethod, split_percentile: splitPercentile, split_threshold: largeThreshold, gaussian_width_s: gaussianWidth, amplitude_min: amplitudeMin, amplitude_max: amplitudeMax }
  const loadSeparationPlot = async () => {
    setViewMode('separation')
    setLoadingSeparation(true)
    try {
      // The original program uses a full-record percentile split (75%) and
      // only crops the requested 10-minute display window afterwards.
      const response = await fetch('/api/replay/separation-plot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start_s: 0, duration_s: 600, separation: { ...separationPayload, split_method: 'percentile', split_percentile: 75 } }),
      })
      if (!response.ok) throw new Error(await response.text())
      const url = URL.createObjectURL(await response.blob())
      setSeparationImageUrl((old) => { if (old) URL.revokeObjectURL(old); return url })
    } catch (error) {
      setViewMode('tracks')
      alert(`三分类图生成失败：${String(error)}`)
    } finally {
      setLoadingSeparation(false)
    }
  }
  const switchMethod = (methodId: string) => {
    setSelectedMethod(methodId)
    if (methodId === state.method_id || switchingMethod) return
    setSwitchingMethod(true)
    send({ action: 'select_method', method_id: methodId, start_s: cursor, speed: state.speed, window_s: windowDuration, stride_s: strideDuration, waveform_downsample: waveformDownsample, separation: separationPayload, hybrid_config: hybridConfig })
  }
  const updateHybridConfig = (section: string, key: string, rawValue: string) => {
    setHybridConfig((old) => ({ ...old, [section]: { ...old[section], [key]: rawValue === '' ? '' : (/^-?\d+(\.\d+)?$/.test(rawValue) ? Number(rawValue) : rawValue) } }))
  }
  const configFieldReadOnly = (section: string, key: string) => section === 'runtime' && key === 'device' || ['raw_path', 'pre_path', 'gauss_path', 'mapping_path', 'checkpoint', 'output_dir'].includes(key)
  const inferenceLabel = state.method_id === 'graph_search' ? '推理耗时' : '推理耗时'
  return <main className="app-shell">
    <header><div><h1>车辆识别回放</h1><span className="subtitle">{status.dataset_label ?? 'DAY11'} · {methodLabel} · {Number(state.window_s ?? windowDuration)} 秒窗口 · {activeMethod?.version ?? 'worker/v1'}</span></div><span className={connected ? 'pill online' : 'pill'}>{connected ? '● 已连接' : '○ 未连接'}</span></header>
    <section className="toolbar"><button onClick={() => { if (!state.playing && selectedMethod !== state.method_id) switchMethod(selectedMethod); else send({ action: state.playing ? 'pause' : 'resume' }) }}>{state.playing ? '暂停' : '播放'}</button><button onClick={() => send({ action: 'restart', method_id: selectedMethod, separation: separationPayload, hybrid_config: hybridConfig })}>重新开始</button><label>识别方法 <select value={selectedMethod} disabled={switchingMethod} onChange={(e) => switchMethod(e.target.value)}>{(methods.length ? methods : [{ method_id: 'hybrid', label: 'Hybrid', description: '', available: true, preset_id: 'hybrid' }]).map((item) => <option key={item.method_id} value={item.method_id} disabled={!item.available} title={item.reason}>{item.label}{item.available ? '' : '（不可用）'}</option>)}</select></label><label>倍速 <select value={String(speed)} onChange={(e) => { const value = e.target.value === 'fastest' ? 'fastest' : Number(e.target.value); setSpeed(value); send({ action: 'speed', value }) }}><option value="1">1×</option><option value="5">5×</option><option value="10">10×</option><option value="60">60×</option><option value="fastest">最快</option></select></label><span className="base-label">底图：大车分离后波形</span><label>轨迹窗口长度 <input className="number-input" type="number" min="1" step="1" value={windowDuration} onChange={(e) => setWindowDuration(Math.max(1, Number(e.target.value) || 1))} /> s</label><label>每隔多久识别 <input className="number-input" type="number" min="1" step="1" value={strideDuration} onChange={(e) => setStrideDuration(Math.max(1, Number(e.target.value) || 1))} /> s</label><label>波形降采样 <input className="number-input" type="number" min="1" step="1" value={waveformDownsample} onChange={(e) => setWaveformDownsample(Math.max(1, Math.round(Number(e.target.value) || 1)))} /> ×</label><label className="check-label"><input type="checkbox" checked={separationEnabled} onChange={(e) => setSeparationEnabled(e.target.checked)} /> 仅使用大车信号</label><label>Pick阈值 <input className="number-input" type="number" min="0" max="1" step="0.01" value={pickThreshold} onChange={(e) => setPickThreshold(Math.min(1, Math.max(0, Number(e.target.value) || 0)))} /></label><label>大车阈值 <input className="number-input" type="number" min="0" max="1" step="0.01" value={largeThreshold} disabled={splitMethod !== 'threshold'} onChange={(e) => setLargeThreshold(Math.min(1, Math.max(0, Number(e.target.value) || 0)))} /></label><label>最小峰间隔 <input className="number-input" type="number" min="0.01" step="0.1" value={minGap} onChange={(e) => setMinGap(Math.max(0.01, Number(e.target.value) || 0.01))} /> s</label><label>能量窗 <input className="number-input" type="number" min="0.01" step="0.1" value={energyWindow} onChange={(e) => setEnergyWindow(Math.max(0.01, Number(e.target.value) || 0.01))} /> s</label><label>分离方式 <select value={splitMethod} onChange={(e) => setSplitMethod(e.target.value as 'threshold'|'percentile'|'kmeans')}><option value="threshold">固定阈值</option><option value="percentile">百分位</option><option value="kmeans">KMeans</option></select></label><label>百分位 <input className="number-input" type="number" min="0" max="100" step="1" value={splitPercentile} disabled={splitMethod !== 'percentile'} onChange={(e) => setSplitPercentile(Math.min(100, Math.max(0, Number(e.target.value) || 0)))} /></label><label>高斯宽度 <input className="number-input" type="number" min="0.001" step="0.01" value={gaussianWidth} onChange={(e) => setGaussianWidth(Math.max(0.001, Number(e.target.value) || 0.001))} /> s</label><label>振幅范围 <input className="number-input" type="number" min="0" step="0.01" value={amplitudeMin} onChange={(e) => setAmplitudeMin(Math.max(0, Number(e.target.value) || 0))} />–<input className="number-input" type="number" min="0" step="0.01" value={amplitudeMax} onChange={(e) => setAmplitudeMax(Math.max(amplitudeMin + 0.001, Number(e.target.value) || amplitudeMin + 0.001))} /></label><button onClick={() => send({ action: 'configure', method_id: selectedMethod, window_s: windowDuration, stride_s: strideDuration, waveform_downsample: waveformDownsample, separation: separationPayload, hybrid_config: hybridConfig, start_s: 0 })}>应用识别参数</button><button onClick={() => setShowHybridConfig((value) => !value)}>{showHybridConfig ? '收起 Hybrid 参数' : '展开 Hybrid 参数'}</button><span className="schedule-hint">服务端当前：每 {Number(status.prediction_schedule?.interval_s ?? strideDuration)}s 识别，窗口 {Number(status.prediction_schedule?.window_length_s ?? windowDuration)}s，波形 {Number(status.prediction_schedule?.waveform_downsample ?? waveformDownsample)}× · 仅使用大车信号{switchingMethod ? ' · 正在切换算法…' : ''}</span><input type="range" min="0" max={Number(state.total_duration_s ?? 3600)} step="1" value={cursor} onChange={(e) => send({ action: 'seek', value: Number(e.target.value) })} /><span className="timecode">{cursor.toFixed(0)} / {Number(state.total_duration_s ?? 3600).toFixed(0)} s</span></section>
    {showHybridConfig && <section className="hybrid-config"><div className="config-title"><strong>Hybrid 全部配置</strong><span>可编辑项会在点击“应用识别参数”后重新加载 Hybrid worker；数据路径、模型路径和设备为当前数据固定项。</span></div>{Object.entries(hybridConfig).map(([section, values]) => <div className="config-section" key={section}><h3>{section}</h3>{Object.entries(values).map(([key, value]) => <label className="config-field" key={`${section}.${key}`}><span>{key}</span><input type={typeof value === 'number' ? 'number' : 'text'} step="any" value={String(value ?? '')} readOnly={configFieldReadOnly(section, key)} onChange={(e) => updateHybridConfig(section, key, e.target.value)} /></label>)}</div>)}</section>}
    <div className="view-switch"><button className={viewMode === 'tracks' ? 'selected' : ''} onClick={() => setViewMode('tracks')}>网页轨迹图</button><button className={viewMode === 'separation' ? 'selected' : ''} onClick={loadSeparationPlot}>{loadingSeparation ? '正在生成三分类图…' : '原始程序三分类图'}</button><span>三分类图按原始 Python 程序：整段分类后显示前 10 分钟</span></div>
    <div className="layout"><section className="plot-card">{viewMode === 'separation' ? <div className="separation-view">{separationImageUrl ? <img src={separationImageUrl} alt="大车、正向小车、反向小车三分类分离图" /> : <div className="empty">正在生成原始程序风格的三分类图…</div>}</div> : <WaveformCanvas frames={displayFrames} positions={positions} cursor={cursor} windowDuration={windowDuration} tracks={plotTracks} selected={selected} />}</section><aside>{displayVideos.length > 0 && <section className="video-panel"><div className="video-title"><strong>现场视频同步回放</strong><span>02.mp4 · 视频道 {VIDEO_OFFSET_M} m · {cursor.toFixed(1)} s</span></div><div className="video-grid">{displayVideos.map((video) => <div className="video-card" key={video.id}><video ref={(element) => { videoRefs.current[video.id] = element }} src={video.url} controls muted preload="metadata" aria-label="02.mp4 现场视频" onLoadedMetadata={(event) => { const element = event.currentTarget; const target = Math.min(cursor, Math.max(0, element.duration - 0.05)); if (Math.abs(element.currentTime - target) > 0.45) { videoLastSeekAt.current[video.id] = Date.now(); element.currentTime = target } }} /><span>02.mp4</span></div>)}</div></section>}<div className="metric-grid"><Metric label="窗口候选" value={counters.window_candidates} /><Metric label="窗口去重" value={counters.window_unique ?? '—'} /><Metric label="待确认" value={counters.tentative_tracks ?? 0} /><Metric label="当前去重" value={counters.current_unique} /><Metric label="累计唯一" value={counters.cumulative_unique} /><Metric label="重连" value={counters.gap_reconnections ?? 0} /><Metric label="ID合并" value={counters.id_merges ?? 0} /><Metric label="中位速度" value={counters.median_speed_kmh == null ? '—' : `${counters.median_speed_kmh.toFixed(1)} km/h`} /><Metric label="平均速度" value={counters.mean_speed_kmh == null ? '—' : `${counters.mean_speed_kmh.toFixed(1)} km/h`} /><Metric label="速度范围" value={speedRange[0] == null ? '—' : `${speedRange[0].toFixed(1)}–${speedRange[1]!.toFixed(1)}`} /><Metric label={inferenceLabel} value={inference == null ? '—' : `${inference.toFixed(3)} s`} /><Metric label="队列" value={state.prediction_queue ?? 0} /></div><div className="vehicle-head"><h2>车辆列表</h2><button className="export" onClick={async () => { const response = await fetch('/api/replay/export', { method: 'POST' }); const result = await response.json(); alert(`已导出 ${result.track_count} 条车辆轨迹`) }}>导出</button></div><div className="vehicle-table">{plotTracks.length === 0 && <div className="empty">等待当前方法的首次推理…</div>}{plotTracks.map((track) => <button className={`vehicle-row ${selected === track.global_vehicle_id ? 'selected' : ''}`} key={track.global_vehicle_id} onClick={() => setSelected(track.global_vehicle_id)}><span className="vehicle-id">{track.global_vehicle_id}</span><strong>{track.median_speed_kmh.toFixed(1)} <small>km/h</small></strong><span>置信 {track.confidence == null ? '—' : track.confidence.toFixed(2)} · {track.observed_count}站</span><span>{track.span_m.toFixed(0)}m · 缺口 {track.max_gap_m.toFixed(0)}m</span><span>{track.enters_window ? '入窗 ' : ''}{track.exits_window ? '出窗 ' : ''}{track.ambiguous_crossing ? '交叉歧义' : ''}</span></button>)}</div></aside></div>
  </main>
}

function Metric({ label, value }: { label: string; value: string|number }) { return <div className="metric"><span>{label}</span><strong>{value}</strong></div> }

function WaveformCanvas({ frames, positions, cursor, windowDuration, tracks, selected }: { frames: Frame[]; positions: number[]; cursor: number; windowDuration: number; tracks: Track[]; selected: string | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const width = Math.max(640, canvas.clientWidth || 640)
    const height = Math.max(600, canvas.clientHeight || 600)
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    const context = canvas.getContext('2d')
    if (!context) return
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    context.clearRect(0, 0, width, height)
    context.fillStyle = '#fbfcfe'
    context.fillRect(0, 0, width, height)

    const left = 58, right = 22, top = 28, bottom = 42
    const plotWidth = width - left - right
    const plotHeight = height - top - bottom
    const xValues = positions.length ? positions : Array.from({ length: 50 }, (_, index) => index * 100)
    const xMin = Math.min(...xValues), xMax = Math.max(...xValues)
    const yMin = Math.max(0, cursor - windowDuration), yMax = Math.max(windowDuration, cursor)
    const xScale = (value: number) => left + (value - xMin) / Math.max(1, xMax - xMin) * plotWidth
    const yScale = (value: number) => top + (value - yMin) / Math.max(1, yMax - yMin) * plotHeight
    const spacing = xValues.length > 1 ? (xValues[1] - xValues[0]) : 100
    const wiggleScale = spacing / Math.max(1, xMax - xMin) * plotWidth * 0.30

    context.strokeStyle = '#dfe4ea'
    context.lineWidth = 1
    context.font = '11px system-ui, sans-serif'
    context.fillStyle = '#66717e'
    for (let station = 0; station < xValues.length; station += 5) {
      const x = xScale(xValues[station])
      context.beginPath(); context.moveTo(x, top); context.lineTo(x, top + plotHeight); context.stroke()
      context.textAlign = 'center'; context.fillText(String(station), x, height - 22)
    }
    for (let time = Math.ceil(yMin / 20) * 20; time <= yMax; time += 20) {
      const y = yScale(time)
      context.beginPath(); context.moveTo(left, y); context.lineTo(left + plotWidth, y); context.stroke()
      context.textAlign = 'right'; context.fillText(`${time}`, left - 8, y + 4)
    }
    context.fillStyle = '#26313e'; context.textAlign = 'left'; context.font = '600 13px system-ui, sans-serif'
    context.fillText('大车分离后底图 · 红色 = 大车 · 灰色 = 原始波形参考', left, 17)

    // The thin wiggles reproduce the original program's reference traces.
    context.save()
    context.beginPath(); context.rect(left, top, plotWidth, plotHeight); context.clip()
    context.strokeStyle = 'rgba(80, 90, 102, 0.44)'; context.lineWidth = 0.7
    for (let station = 0; station < xValues.length; station++) {
      let open = false
      for (const frame of frames) {
        const samples = frame.waveformRate > 0 ? Math.floor(frame.rawWave.length / frame.stationCount) : 0
        for (let index = 0; index < samples; index++) {
          const time = frame.start + index / frame.waveformRate
          if (time < yMin || time > yMax) continue
          const value = Math.max(-1, Math.min(1, frame.rawWave[index * frame.stationCount + station] ?? 0))
          const x = xScale(xValues[station]) + value * wiggleScale
          const y = yScale(time)
          if (!open) { context.beginPath(); open = true }
          context.lineTo(x, y)
        }
      }
      if (open) context.stroke()
    }

    // Only the large-vehicle classification plane is drawn as the colored base.
    for (const frame of frames) {
      for (let index = 0; index < frame.times.length; index++) {
        const time = frame.times[index]
        if (time < yMin || time > yMax) continue
        for (let station = 0; station < frame.stationCount; station++) {
          const value = frame.largeGauss[index * frame.stationCount + station] ?? 0
          if (value <= 0.008) continue
          const alpha = Math.min(0.92, 0.16 + value * 0.95)
          const x = xScale(xValues[station]), y = yScale(time)
          context.fillStyle = `rgba(211, 47, 47, ${alpha})`
          context.fillRect(x - 3.2, y - 2.2, 6.4, 4.4)
        }
      }
    }
    context.restore()

    const trackColors = ['#246bce', '#e67e22', '#8e44ad', '#159957', '#c0392b']
    tracks.forEach((track, trackIndex) => {
      const points = track.points.filter((point) => point.time_s >= yMin - 2 && point.time_s <= yMax + 2).sort((a, b) => a.time_s - b.time_s)
      if (!points.length) return
      const color = trackColors[trackIndex % trackColors.length]
      context.strokeStyle = color; context.lineWidth = selected === track.global_vehicle_id ? 3 : 1.6; context.globalAlpha = 0.9
      context.beginPath(); points.forEach((point, index) => { const x = xScale(point.position_m), y = yScale(point.time_s); if (index === 0) context.moveTo(x, y); else context.lineTo(x, y) }); context.stroke()
      context.globalAlpha = 1
      const last = points[points.length - 1]
      context.fillStyle = color; context.font = '600 11px system-ui, sans-serif'; context.textAlign = 'left'
      context.fillText(`${track.global_vehicle_id}  ${track.median_speed_kmh.toFixed(1)} km/h`, xScale(last.position_m) + 5, yScale(last.time_s) - 4)
    })

    // One station is 100 m, so 4000 m is the 40th channel/road. Use the
    // station array directly so the marker stays on 道 40 even when the
    // absolute survey coordinates do not start at zero.
    const videoPositionM = xValues[VIDEO_CHANNEL_INDEX]
    if (videoPositionM != null && videoPositionM >= xMin && videoPositionM <= xMax) {
      const videoX = xScale(videoPositionM)
      context.save()
      context.beginPath(); context.rect(left, top, plotWidth, plotHeight); context.clip()
      context.strokeStyle = '#f08c00'; context.lineWidth = 3; context.setLineDash([10, 7])
      context.beginPath(); context.moveTo(videoX, top); context.lineTo(videoX, top + plotHeight); context.stroke()
      context.restore()
      context.setLineDash([])
      context.fillStyle = '#f08c00'; context.font = '700 12px system-ui, sans-serif'; context.textAlign = 'center'
      context.fillText(`视频所在道 · ${VIDEO_OFFSET_M} m`, videoX, top + 16)
      context.fillStyle = '#fff7e6'; context.fillRect(videoX - 31, height - 34, 62, 18)
      context.fillStyle = '#b35f00'; context.font = '700 11px system-ui, sans-serif'; context.fillText(`${VIDEO_OFFSET_M} m`, videoX, height - 21)
    }
    context.strokeStyle = '#9da7b3'; context.lineWidth = 1; context.strokeRect(left, top, plotWidth, plotHeight)
    context.fillStyle = '#5e6874'; context.font = '11px system-ui, sans-serif'; context.textAlign = 'center'; context.fillText('物理位置 / 道号', left + plotWidth / 2, height - 5)
    context.save(); context.translate(14, top + plotHeight / 2); context.rotate(-Math.PI / 2); context.fillText('时间 (s)', 0, 0); context.restore()
  }, [frames, positions, cursor, windowDuration, tracks, selected])

  return <canvas ref={canvasRef} className="waveform-canvas" aria-label="大车分离后的波形底图，从测线起点起算4000米处为视频所在道" />
}

export default App
