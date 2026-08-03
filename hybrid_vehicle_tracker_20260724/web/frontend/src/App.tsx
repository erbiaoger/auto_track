import { useEffect, useMemo, useRef, useState } from 'react'
import * as echarts from 'echarts'

type Point = { channel_index: number; station_id: string; position_m: number; time_s: number; observed: boolean; ambiguous?: boolean }
type Track = { global_vehicle_id: string; points: Point[]; median_speed_kmh: number; confidence: number | null; observed_count: number; span_m: number; max_gap_m: number; enters_window: boolean; exits_window: boolean; ambiguous_crossing: boolean }
type Counters = { window_candidates: number; current_unique: number; cumulative_unique: number; mean_speed_kmh: number | null; median_speed_kmh: number | null; speed_range_kmh: [number|null, number|null] }
type Frame = { times: number[]; gauss: number[]; rawMin: number[]; rawMax: number[]; rawWave: number[]; waveformRate: number; stationCount: number; start: number; duration: number }
type MethodDescriptor = { method_id: string; label: string; description: string; available: boolean; reason?: string; preset_id: string; checkpoint?: string; version?: string }

const COLORS = ['#42d3ff','#ffbf69','#ff6b9d','#9bff8a','#c8a7ff','#f4f36b','#ff8c69','#55e6c1','#7aa7ff','#f78fb3','#63cdda','#e77f67']
const colorForId = (id: string) => { let hash = 0; for (const char of id) hash = (hash * 31 + char.charCodeAt(0)) >>> 0; return COLORS[hash % COLORS.length] }

function decodeFrame(buffer: ArrayBuffer): Frame | null {
  const bytes = new Uint8Array(buffer)
  if (bytes.length < 8 || String.fromCharCode(...bytes.slice(0, 4)) !== 'HVT1') return null
  const view = new DataView(buffer)
  const headerLength = view.getUint32(4, true)
  const header = JSON.parse(new TextDecoder().decode(bytes.slice(8, 8 + headerLength)))
  const lengths: number[] = header.array_lengths ?? [header.times_s.length * header.station_count, header.times_s.length * header.station_count, header.times_s.length * header.station_count]
  // JSON header length is not guaranteed to be 4-byte aligned; slice first so
  // the Float32Array constructor always receives an aligned byte offset.
  const payload = new Float32Array(buffer.slice(8 + headerLength))
  let offset = 0
  const take = (length: number) => { const values = Array.from(payload.slice(offset, offset + length)); offset += length; return values }
  const gauss = take(lengths[0] ?? 0)
  const rawMin = take(lengths[1] ?? 0)
  const rawMax = take(lengths[2] ?? 0)
  const rawWave = take(lengths[3] ?? 0)
  return { times: header.times_s, gauss, rawMin, rawMax, rawWave, waveformRate: Number(header.waveform_rate_hz ?? 0), stationCount: header.station_count, start: header.start_s, duration: header.duration_s }
}

function App() {
  const chartRef = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)
  const frames = useRef<Frame[]>([])
  const [tracks, setTracks] = useState<Record<string, Track>>({})
  const [counters, setCounters] = useState<Counters>({ window_candidates: 0, current_unique: 0, cumulative_unique: 0, mean_speed_kmh: null, median_speed_kmh: null, speed_range_kmh: [null, null] })
  const [status, setStatus] = useState<any>({ state: { cursor_s: 0, playing: false, speed: 1 }, station_positions_m: [] })
  const [mode, setMode] = useState<'gauss'|'raw'|'both'>('gauss')
  const [speed, setSpeed] = useState<number|string>(1)
  const [selected, setSelected] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const [inference, setInference] = useState<number | null>(null)
  const [windowDuration, setWindowDuration] = useState(120)
  const [strideDuration, setStrideDuration] = useState(60)
  const [waveformDownsample, setWaveformDownsample] = useState(20)
  const [methods, setMethods] = useState<MethodDescriptor[]>([])
  const [selectedMethod, setSelectedMethod] = useState('hybrid')
  const sessionRef = useRef('')
  const windowDurationRef = useRef(windowDuration)
  windowDurationRef.current = windowDuration

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
        if (message.state?.method_id && isNewSession) setSelectedMethod(String(message.state.method_id))
        setStatus(message)
        if (message.state?.window_s != null) setWindowDuration(Number(message.state.window_s))
        if (message.state?.stride_s != null) setStrideDuration(Number(message.state.stride_s))
        if (message.state?.waveform_downsample != null) setWaveformDownsample(Number(message.state.waveform_downsample))
      }
      if (message.event === 'prediction' || message.event === 'vehicle_update') {
        if (message.counters) setCounters(message.counters)
        if (message.elapsed_s != null) setInference(message.elapsed_s)
        if (message.tracks && (!message.session_id || message.session_id === sessionRef.current)) setTracks((old) => ({ ...old, ...Object.fromEntries(message.tracks.map((item: Track) => [item.global_vehicle_id, item])) }))
      }
    }
    return () => { ws.close(); wsRef.current = null }
  }, [])

  const positions: number[] = status.station_positions_m?.length ? status.station_positions_m : Array.from({ length: 50 }, (_, i) => i * 100)
  const cursor = status.state?.cursor_s ?? 0
  const plotTracks = useMemo(() => Object.values(tracks).filter((track) => track.points.some((point) => point.time_s >= Math.max(0, cursor - windowDuration) && point.time_s <= cursor + 1)), [tracks, cursor, windowDuration])

  useEffect(() => {
    if (!chartRef.current) return
    chart.current ??= echarts.init(chartRef.current)
    const xMin = Math.min(...positions), xMax = Math.max(...positions)
    const yMin = Math.max(0, cursor - windowDuration), yMax = Math.max(windowDuration, cursor)
    const heatmap: any[] = []
    const rawHeatmap: any[] = []
    const waveform: Array<Array<[number, number]>> = Array.from({ length: positions.length }, () => [])
    for (const frame of frames.current) {
      frame.times.forEach((time, t) => { for (let station = 0; station < frame.stationCount; station++) { const value = frame.gauss[t * frame.stationCount + station]; if (value > 0.005) heatmap.push([positions[station], time, value]) } })
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
  }, [positions, cursor, mode, plotTracks, selected, windowDuration])

  const state = status.state ?? {}
  const speedRange = counters.speed_range_kmh ?? [null, null]
  const activeMethod = methods.find((item) => item.method_id === (state.method_id ?? selectedMethod))
  const methodLabel = activeMethod?.label ?? (state.method_id ?? selectedMethod)
  const inferenceLabel = state.method_id === 'graph_search' ? '推理耗时' : '推理耗时'
  return <main className="app-shell">
    <header><div><h1>车辆识别回放</h1><span className="subtitle">DAY11 · {methodLabel} · {Number(state.window_s ?? windowDuration)} 秒窗口 · {activeMethod?.version ?? 'worker/v1'}</span></div><span className={connected ? 'pill online' : 'pill'}>{connected ? '● 已连接' : '○ 未连接'}</span></header>
    <section className="toolbar"><button onClick={() => { if (!state.playing && selectedMethod !== state.method_id) send({ action: 'configure', method_id: selectedMethod, window_s: windowDuration, stride_s: strideDuration, waveform_downsample: waveformDownsample, start_s: 0 }); else send({ action: state.playing ? 'pause' : 'resume' }) }}>{state.playing ? '暂停' : '播放'}</button><button onClick={() => send({ action: 'restart', method_id: selectedMethod })}>重新开始</button><label>识别方法 <select value={selectedMethod} onChange={(e) => setSelectedMethod(e.target.value)}>{(methods.length ? methods : [{ method_id: 'hybrid', label: 'Hybrid', description: '', available: true, preset_id: 'hybrid' }]).map((item) => <option key={item.method_id} value={item.method_id} disabled={!item.available} title={item.reason}>{item.label}{item.available ? '' : '（不可用）'}</option>)}</select></label><label>倍速 <select value={String(speed)} onChange={(e) => { const value = e.target.value === 'fastest' ? 'fastest' : Number(e.target.value); setSpeed(value); send({ action: 'speed', value }) }}><option value="1">1×</option><option value="5">5×</option><option value="10">10×</option><option value="60">60×</option><option value="fastest">最快</option></select></label><label>底图 <select value={mode} onChange={(e) => setMode(e.target.value as any)}><option value="gauss">Gauss</option><option value="raw">Raw</option><option value="both">Gauss + Raw</option></select></label><label>轨迹窗口长度 <input className="number-input" type="number" min="1" step="1" value={windowDuration} onChange={(e) => setWindowDuration(Math.max(1, Number(e.target.value) || 1))} /> s</label><label>每隔多久识别 <input className="number-input" type="number" min="1" step="1" value={strideDuration} onChange={(e) => setStrideDuration(Math.max(1, Number(e.target.value) || 1))} /> s</label><label>波形降采样 <input className="number-input" type="number" min="1" step="1" value={waveformDownsample} onChange={(e) => setWaveformDownsample(Math.max(1, Math.round(Number(e.target.value) || 1)))} /> ×</label><button onClick={() => send({ action: 'configure', method_id: selectedMethod, window_s: windowDuration, stride_s: strideDuration, waveform_downsample: waveformDownsample, start_s: 0 })}>应用识别参数</button><span className="schedule-hint">服务端当前：每 {Number(status.prediction_schedule?.interval_s ?? strideDuration)}s 识别，窗口 {Number(status.prediction_schedule?.window_length_s ?? windowDuration)}s，波形 {Number(status.prediction_schedule?.waveform_downsample ?? waveformDownsample)}×</span><input type="range" min="0" max={Number(state.total_duration_s ?? 3600)} step="1" value={cursor} onChange={(e) => send({ action: 'seek', value: Number(e.target.value) })} /><span className="timecode">{cursor.toFixed(0)} / {Number(state.total_duration_s ?? 3600).toFixed(0)} s</span></section>
    <div className="layout"><section className="plot-card"><div ref={chartRef} className="chart" /></section><aside><div className="metric-grid"><Metric label="窗口候选" value={counters.window_candidates} /><Metric label="当前去重" value={counters.current_unique} /><Metric label="累计唯一" value={counters.cumulative_unique} /><Metric label="中位速度" value={counters.median_speed_kmh == null ? '—' : `${counters.median_speed_kmh.toFixed(1)} km/h`} /><Metric label="平均速度" value={counters.mean_speed_kmh == null ? '—' : `${counters.mean_speed_kmh.toFixed(1)} km/h`} /><Metric label="速度范围" value={speedRange[0] == null ? '—' : `${speedRange[0].toFixed(1)}–${speedRange[1]!.toFixed(1)}`} /><Metric label={inferenceLabel} value={inference == null ? '—' : `${inference.toFixed(3)} s`} /><Metric label="队列" value={state.prediction_queue ?? 0} /></div><div className="vehicle-head"><h2>车辆列表</h2><button className="export" onClick={async () => { const response = await fetch('/api/replay/export', { method: 'POST' }); const result = await response.json(); alert(`已导出 ${result.track_count} 条车辆轨迹`) }}>导出</button></div><div className="vehicle-table">{plotTracks.length === 0 && <div className="empty">等待当前方法的首次推理…</div>}{plotTracks.map((track) => <button className={`vehicle-row ${selected === track.global_vehicle_id ? 'selected' : ''}`} key={track.global_vehicle_id} onClick={() => setSelected(track.global_vehicle_id)}><span className="vehicle-id">{track.global_vehicle_id}</span><strong>{track.median_speed_kmh.toFixed(1)} <small>km/h</small></strong><span>置信 {track.confidence == null ? '—' : track.confidence.toFixed(2)} · {track.observed_count}站</span><span>{track.span_m.toFixed(0)}m · 缺口 {track.max_gap_m.toFixed(0)}m</span><span>{track.enters_window ? '入窗 ' : ''}{track.exits_window ? '出窗 ' : ''}{track.ambiguous_crossing ? '交叉歧义' : ''}</span></button>)}</div></aside></div>
  </main>
}

function Metric({ label, value }: { label: string; value: string|number }) { return <div className="metric"><span>{label}</span><strong>{value}</strong></div> }

export default App
