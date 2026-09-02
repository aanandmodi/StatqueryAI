'use client';

import {
  AlertTriangle,
  ArrowDownToLine,
  ArrowUpRight,
  Boxes,
  Check,
  ChevronDown,
  CircleDot,
  FileImage,
  Layers3,
  LoaderCircle,
  Map,
  Orbit,
  RadioTower,
  ScanSearch,
  Sparkles,
  UploadCloud,
  X,
} from 'lucide-react';
import Image from 'next/image';
import { ChangeEvent, CSSProperties, useEffect, useMemo, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';

type Task = 'single_vqa' | 'caption' | 'grounding' | 'change_vqa' | 'optical_sar_fusion';
type AnalysisStatus =
  | 'queued'
  | 'validating'
  | 'planning'
  | 'running'
  | 'integrating'
  | 'succeeded'
  | 'failed'
  | 'cancelled';

type EvidenceItem = {
  id: string;
  type: string;
  label: string;
  score: number;
  coordinate_space: string;
  geometry: Record<string, number>;
};

type TraceEvent = {
  step_id: string;
  task: string;
  tool: string;
  model_version?: string;
  status: string;
  policy_reason: string;
  duration_ms?: number;
};

type AnalysisResult = {
  answer: string;
  facts: Array<Record<string, unknown>>;
  evidence: EvidenceItem[];
  confidence: {
    score: number;
    level: 'low' | 'medium' | 'high';
    calibration_version: string;
    meaning: string;
  };
  trace: TraceEvent[];
  warnings: string[];
  report_url?: string;
  provenance: Record<string, unknown>;
};

type AnalysisRecord = {
  id: string;
  status: AnalysisStatus;
  progress: number;
  result?: AnalysisResult;
  error_code?: string;
  error_message?: string;
};

type AssetRecord = { id: string; original_name: string; size_bytes: number };

const taskOptions: Array<{ value: Task; label: string; note: string }> = [
  { value: 'single_vqa', label: 'Ask one scene', note: 'Qwen3-VL · released' },
  { value: 'caption', label: 'Describe the scene', note: 'Qwen3-VL · released' },
  { value: 'grounding', label: 'Locate a feature', note: 'Qwen3-VL · released' },
  { value: 'change_vqa', label: 'Compare two dates', note: 'Specialist training gate' },
  { value: 'optical_sar_fusion', label: 'Fuse optical + SAR', note: 'Specialist training gate' },
];

const statusCopy: Record<AnalysisStatus, string> = {
  queued: 'Queued for the free model',
  validating: 'Checking raster integrity',
  planning: 'Selecting a permitted specialist',
  running: 'Reading the visible evidence',
  integrating: 'Structuring the answer and trace',
  succeeded: 'Analysis complete',
  failed: 'Analysis could not complete',
  cancelled: 'Analysis cancelled',
};

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function jsonRequest<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await fetch(input, { ...init, cache: 'no-store' });
  const payload = (await response.json().catch(() => ({}))) as {
    error?: { message?: string };
  };
  if (!response.ok) {
    const message = payload?.error?.message || `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return payload as T;
}

function delay(milliseconds: number) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function EvidenceOverlay({ item }: { item: EvidenceItem }) {
  if (item.type !== 'box' || item.coordinate_space !== 'normalized') return null;
  const { x, y, width, height } = item.geometry;
  if (![x, y, width, height].every(Number.isFinite)) return null;
  const style = {
    left: `${Math.max(0, x) * 100}%`,
    top: `${Math.max(0, y) * 100}%`,
    width: `${Math.min(1, width) * 100}%`,
    height: `${Math.min(1, height) * 100}%`,
  } as CSSProperties;
  return (
    <div className="evidence-box" style={style}>
      <span>{item.label}</span>
    </div>
  );
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [task, setTask] = useState<Task>('single_vqa');
  const [query, setQuery] = useState('Which land-cover features are visible in this scene?');
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  const [altitude, setAltitude] = useState('');
  const [sensor, setSensor] = useState('');
  const [availableTasks, setAvailableTasks] = useState<Task[]>([
    'single_vqa',
    'caption',
    'grounding',
  ]);
  const [systemState, setSystemState] = useState<'checking' | 'ready' | 'sleeping'>('checking');
  const [asset, setAsset] = useState<AssetRecord | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisRecord | null>(null);
  const [phase, setPhase] = useState<'idle' | 'uploading' | 'working' | 'succeeded' | 'failed'>('idle');
  const [error, setError] = useState('');

  const busy = phase === 'uploading' || phase === 'working';
  const result = analysis?.result;
  const previewSource = asset ? `/api/satquery/assets/${asset.id}/preview` : '';
  const selectedOption = taskOptions.find((option) => option.value === task)!;
  const progress = phase === 'uploading' ? 0.04 : analysis?.progress ?? 0;
  const uncalibrated = result?.confidence.calibration_version.includes('uncalibrated');
  const confidenceLabel = result
    ? uncalibrated
      ? `Evidence quality · ${result.confidence.level}`
      : `${Math.round(result.confidence.score * 100)}% calibrated`
    : 'No score calculated';

  const displayTrace = useMemo(() => {
    if (result?.trace.length) return result.trace;
    if (phase === 'idle') return [];
    const rows: TraceEvent[] = [];
    if (asset) {
      rows.push({
        step_id: 'upload',
        task: 'validation',
        tool: 'immutable-upload',
        status: 'succeeded',
        policy_reason: 'Asset stored and inspected.',
      });
    }
    if (analysis) {
      rows.push({
        step_id: analysis.status,
        task: analysis.status,
        tool: 'satquery-controller',
        status: busy ? 'started' : analysis.status,
        policy_reason: statusCopy[analysis.status],
      });
    }
    return rows;
  }, [analysis, asset, busy, phase, result]);

  useEffect(() => {
    let active = true;
    async function checkSystems() {
      try {
        const [, capabilities] = await Promise.all([
          jsonRequest<{ status: string }>('/api/satquery/health/ready'),
          jsonRequest<{ tasks: Task[] }>('/api/satquery/capabilities'),
        ]);
        if (!active) return;
        setAvailableTasks(capabilities.tasks);
        setSystemState('ready');
      } catch {
        if (!active) return;
        setSystemState('sleeping');
      }
    }
    void checkSystems();
    return () => {
      active = false;
    };
  }, []);

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setError('');
    setAsset(null);
    setAnalysis(null);
    setPhase('idle');
    if (!selected) {
      setFile(null);
      return;
    }
    if (!/\.(tif|tiff)$/i.test(selected.name)) {
      setFile(null);
      setError('Use a georeferenced .tif or .tiff. PNG/JPEG is reserved for named benchmark imports.');
      event.target.value = '';
      return;
    }
    if (selected.size > 50 * 1024 * 1024) {
      setFile(null);
      setError('The local profile accepts files up to 50 MB. Tile or crop this raster first.');
      event.target.value = '';
      return;
    }
    setFile(selected);
  }

  async function runAnalysis() {
    if (!file || busy) return;
    if (!availableTasks.includes(task)) {
      setError(`${selectedOption.label} is visible in the architecture but its specialist has not passed release yet.`);
      return;
    }
    if (query.trim().length < 2) {
      setError('Write a specific question before running the analysis.');
      return;
    }
    const hasPartialLocation = [latitude, longitude, altitude].some((value) => value.trim());
    if (hasPartialLocation && (!latitude.trim() || !longitude.trim())) {
      setError('Latitude and longitude must be supplied together. Altitude is optional.');
      return;
    }
    const latitudeNumber = latitude.trim() ? Number(latitude) : null;
    const longitudeNumber = longitude.trim() ? Number(longitude) : null;
    const altitudeNumber = altitude.trim() ? Number(altitude) : null;
    if (latitudeNumber !== null && (!Number.isFinite(latitudeNumber) || latitudeNumber < -90 || latitudeNumber > 90)) {
      setError('Latitude must be a number between -90 and 90.');
      return;
    }
    if (longitudeNumber !== null && (!Number.isFinite(longitudeNumber) || longitudeNumber < -180 || longitudeNumber > 180)) {
      setError('Longitude must be a number between -180 and 180.');
      return;
    }
    if (altitudeNumber !== null && (!Number.isFinite(altitudeNumber) || altitudeNumber < -500 || altitudeNumber > 100000)) {
      setError('Altitude must be between -500 and 100000 metres.');
      return;
    }

    setError('');
    setAnalysis(null);
    setAsset(null);
    setPhase('uploading');
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('modality', 'optical');
      form.append('role', 'primary');
      const uploaded = await jsonRequest<AssetRecord>('/api/satquery/assets', {
        method: 'POST',
        body: form,
      });
      setAsset(uploaded);
      setPhase('working');

      const created = await jsonRequest<AnalysisRecord>('/api/satquery/analyses', {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'idempotency-key': crypto.randomUUID() },
        body: JSON.stringify({
          query: query.trim(),
          asset_ids: [uploaded.id],
          requested_tasks: [task],
          context: latitudeNumber !== null && longitudeNumber !== null ? {
            latitude: latitudeNumber,
            longitude: longitudeNumber,
            altitude_m: altitudeNumber,
            sensor: sensor.trim() || null,
            source: 'user',
            metadata: {},
          } : null,
          parameters: {},
        }),
      });
      setAnalysis(created);

      const deadline = Date.now() + 10 * 60 * 1000;
      let current = created;
      while (!['succeeded', 'failed', 'cancelled'].includes(current.status)) {
        if (Date.now() >= deadline) {
          throw new Error('The model exceeded the ten-minute local job limit. Check the model-service terminal.');
        }
        await delay(1200);
        current = await jsonRequest<AnalysisRecord>(`/api/satquery/analyses/${created.id}`);
        setAnalysis(current);
      }
      if (current.status !== 'succeeded' || !current.result) {
        throw new Error(current.error_message || 'The analysis ended without a result.');
      }
      setPhase('succeeded');
    } catch (caught) {
      setPhase('failed');
      setError(caught instanceof Error ? caught.message : 'Analysis failed unexpectedly.');
    }
  }

  function clearFile() {
    setFile(null);
    setAsset(null);
    setAnalysis(null);
    setPhase('idle');
    if (inputRef.current) inputRef.current.value = '';
  }

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />

      <header className="topbar glass-panel">
        <a className="brand" href="#workspace" aria-label="SatQuery home">
          <span className="brand-mark"><Orbit /></span>
          <span><strong>satquery</strong><small>orbital intelligence</small></span>
        </a>
        <div className="project-switcher" aria-label="Current workspace">
          <span className="status-dot" />
          SIH / Evidence workspace
          <ChevronDown size={14} aria-hidden="true" />
        </div>
        <div className="topbar-actions">
          <span className={`live-pill ${systemState}`}>
            {systemState === 'checking' ? <LoaderCircle size={13} /> : <RadioTower size={13} />}
            {systemState === 'ready'
              ? 'Local controller ready'
              : systemState === 'checking'
                ? 'Checking systems'
                : 'Local services offline'}
          </span>
          <span className="zero-cost-badge">Local-first · ₹0</span>
        </div>
      </header>

      <section className="workspace" id="workspace">
        <aside className="rail glass-panel" aria-label="Workspace sections">
          <nav>
            <a className="rail-item active" href="#query-card" aria-label="Query workspace"><ScanSearch /><span>Query</span></a>
            <a className="rail-item" href="#evidence" aria-label="Evidence canvas"><Map /><span>Evidence</span></a>
            <a className="rail-item" href="#answer" aria-label="Analysis answer"><Boxes /><span>Answer</span></a>
            <a className="rail-item" href="#trace" aria-label="Execution trace"><Layers3 /><span>Trace</span></a>
          </nav>
          <div className="rail-orbit" aria-hidden="true"><span /><CircleDot /></div>
        </aside>

        <div className="main-column">
          <div className="eyebrow"><Sparkles size={14} /> Evidence-first remote sensing</div>
          <div className="heading-row">
            <div>
              <h1>Ask the landscape.</h1>
              <p>A specialist answer, grounded in the pixels and honest about uncertainty.</p>
            </div>
            <div className="mission-badge"><span>SIH 26167</span><strong>Auditable by design</strong></div>
          </div>

          <section className="query-card glass-panel" id="query-card" aria-labelledby="query-title">
            <div className="query-card-head">
              <div>
                <span className="step-number">01</span>
                <div><h2 id="query-title">Build your evidence set</h2><p>GeoTIFF on this machine · 50 MB maximum.</p></div>
              </div>
              <span className="secure-label">Immutable upload · local workspace</span>
            </div>

            <input ref={inputRef} className="sr-only" type="file" accept=".tif,.tiff,image/tiff" onChange={chooseFile} />
            <div className="upload-grid single-upload">
              <button className={`upload-tile ${file ? 'filled' : ''}`} type="button" onClick={() => inputRef.current?.click()} disabled={busy}>
                <span className={`file-icon ${file ? 'optical' : ''}`}>{file ? <FileImage /> : <UploadCloud />}</span>
                <span className="upload-copy">
                  <strong>{file ? file.name : 'Choose an optical scene'}</strong>
                  <small>{file ? `${formatBytes(file.size)} · ready for validation` : 'Georeferenced TIFF / GeoTIFF'}</small>
                </span>
                {file ? <span className="verified">Selected</span> : <ArrowUpRight size={17} />}
              </button>
              {file && <button className="clear-file" type="button" onClick={clearFile} disabled={busy}><X size={15} /> Remove scene</button>}
            </div>

            <div className="task-picker">
              <label htmlFor="task">Specialist route</label>
              <select id="task" value={task} onChange={(event) => setTask(event.target.value as Task)} disabled={busy}>
                {taskOptions.map((option) => (
                  <option key={option.value} value={option.value} disabled={!availableTasks.includes(option.value)}>
                    {option.label} — {option.note}
                  </option>
                ))}
              </select>
              <span>{selectedOption.note}</span>
            </div>

            <fieldset className="location-context">
              <legend>Location context <span>optional but recommended</span></legend>
              <label>Latitude<input inputMode="decimal" value={latitude} onChange={(event) => setLatitude(event.target.value)} placeholder="28.6139" disabled={busy} /></label>
              <label>Longitude<input inputMode="decimal" value={longitude} onChange={(event) => setLongitude(event.target.value)} placeholder="77.2090" disabled={busy} /></label>
              <label>Altitude (m)<input inputMode="decimal" value={altitude} onChange={(event) => setAltitude(event.target.value)} placeholder="216" disabled={busy} /></label>
              <label>Sensor<input value={sensor} onChange={(event) => setSensor(event.target.value)} placeholder="Sentinel-2 / drone" maxLength={120} disabled={busy} /></label>
              <p>Stored as user-provided metadata; the model is never allowed to present it as pixel-derived evidence.</p>
            </fieldset>

            <div className="prompt-box">
              <label htmlFor="query">Your investigation</label>
              <textarea id="query" value={query} onChange={(event) => setQuery(event.target.value)} disabled={busy} maxLength={2000} aria-describedby="query-help" />
              <div className="prompt-footer">
                <div className="chips" id="query-help" aria-label="Execution properties"><span>{task.replaceAll('_', ' ')}</span><span>pinned model</span><span>trace on</span></div>
                <Button className="run-button" size="lg" onClick={runAnalysis} disabled={!file || busy}>
                  {busy ? <><LoaderCircle className="spin" /> {phase === 'uploading' ? 'Validating' : 'Analyzing'}</> : <>Run analysis <ArrowUpRight /></>}
                </Button>
              </div>
            </div>

            {(busy || analysis) && (
              <output className="run-progress" aria-live="polite">
                <div><span>{phase === 'uploading' ? 'Uploading safely' : analysis ? statusCopy[analysis.status] : 'Starting'}</span><strong>{Math.round(progress * 100)}%</strong></div>
                <div className="progress-track"><span style={{ width: `${Math.max(3, progress * 100)}%` }} /></div>
              </output>
            )}
            {error && <div className="error-banner" role="alert"><AlertTriangle size={16} /><span>{error}</span></div>}
          </section>
        </div>

        <aside className="insight-column" aria-label="Analysis output">
          <section className="evidence-card glass-panel" id="evidence">
            <div className="section-label">
              <span><span className={`pulse ${phase === 'succeeded' ? 'complete' : ''}`} /> Evidence canvas</span>
              <span>{result?.evidence.length ?? 0} regions</span>
            </div>
            <figure className={`evidence-map ${asset ? 'has-preview' : ''}`} aria-label={asset ? 'RGB preview of the uploaded raster with model evidence overlays' : 'Empty satellite evidence canvas'}>
              {asset && previewSource ? <Image src={previewSource} alt="RGB preview generated from the uploaded raster" fill sizes="(max-width: 1050px) 60vw, 38vw" unoptimized /> : <div className="empty-orbit" aria-hidden="true"><Orbit /><span>Awaiting scene</span></div>}
              <div className="map-grid" />
              {result?.evidence.map((item) => <EvidenceOverlay key={item.id} item={item} />)}
              <div className="map-coordinates">SOURCE LOCKED<br />MODEL OVERLAY</div>
            </figure>
          </section>

          <section className={`answer-card glass-panel ${phase}`} id="answer" aria-live="polite">
            <div className="answer-topline">
              <span>{phase === 'succeeded' ? 'Model result' : busy ? 'Analysis in progress' : phase === 'failed' ? 'Run interrupted' : 'Awaiting analysis'}</span>
              <strong>{confidenceLabel}</strong>
            </div>
            <h2>{result?.answer || (busy ? (analysis ? statusCopy[analysis.status] : 'Inspecting the upload…') : phase === 'failed' ? 'No model answer was produced.' : 'Your evidence-backed answer will appear here.')}</h2>
            <p>{result?.confidence.meaning || 'Scores stay hidden until a real specialist returns; SatQuery never invents a confidence percentage.'}</p>
            {result && !uncalibrated && <div className="confidence-track"><span style={{ width: `${result.confidence.score * 100}%` }} /></div>}
            {result?.warnings.length ? <details className="warning-list"><summary>{result.warnings.length} model warning{result.warnings.length === 1 ? '' : 's'}</summary><ul>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></details> : null}
            {analysis?.status === 'succeeded' && (
              <div className="artifact-links">
                <a className="report-link" href={`/api/satquery/analyses/${analysis.id}/overlay`}><ArrowDownToLine size={14} /> Download marked image</a>
                {result?.report_url && <a className="report-link" href={`/api/satquery/analyses/${analysis.id}/report`}><ArrowDownToLine size={14} /> Download audit report</a>}
              </div>
            )}
          </section>

          <section className="trace-card glass-panel" id="trace">
            <div className="section-label"><span>Observable execution trace</span><span>{displayTrace.length} events</span></div>
            {displayTrace.length ? (
              <ol>
                {displayTrace.map((step, index) => (
                  <li key={`${step.step_id}-${index}`}>
                    <span className={`trace-node ${step.status === 'succeeded' ? 'mint' : step.status === 'failed' ? 'coral' : 'blue'}`}>
                      {step.status === 'succeeded' ? <Check size={11} /> : index + 1}
                    </span>
                    <span><strong>{step.tool.replaceAll('-', ' ')}</strong><small>{step.model_version || step.task} {step.duration_ms ? `· ${step.duration_ms} ms` : ''}</small><em>{step.policy_reason}</em></span>
                  </li>
                ))}
              </ol>
            ) : <div className="trace-empty">Validation, routing, model revision and timing will be recorded here—never hidden reasoning.</div>}
          </section>
        </aside>
      </section>
    </main>
  );
}
