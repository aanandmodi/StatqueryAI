'use client';

import {
  AlertTriangle,
  ArrowDownToLine,
  ArrowUpRight,
  Boxes,
  Check,
  CircleDot,
  FileImage,
  Layers3,
  LoaderCircle,
  Map,
  Orbit,
  RadioTower,
  ScanSearch,
  UploadCloud,
  X,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import {
  ChangeEvent,
  CSSProperties,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import { Button } from '@/components/ui/button';
import { StructuredAnswer } from '@/components/structured-answer';
import { jsonRequest } from '@/lib/api-client';

type Task =
  | 'single_vqa'
  | 'caption'
  | 'grounding'
  | 'change_vqa'
  | 'optical_sar_fusion';
type RouteChoice = 'auto' | Task;
type InputMode = 'single' | 'temporal' | 'fusion';
type Modality = 'optical' | 'multispectral' | 'sar';
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
  asset_id: string;
  type: string;
  label: string;
  score: number;
  coordinate_space: string;
  geometry: Record<string, unknown>;
  artifact_url?: string;
};

type TraceEvent = {
  step_id: string;
  task: string;
  tool: string;
  model_version?: string;
  status: string;
  policy_reason: string;
  permitted_params?: Record<string, unknown>;
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
  sections?: Array<{ title: string; source: string; paragraphs: string[] }>;
};

type AnalysisRecord = {
  id: string;
  status: AnalysisStatus;
  progress: number;
  result?: AnalysisResult;
  error_code?: string;
  error_message?: string;
};

type AssetRecord = {
  id: string;
  original_name: string;
  size_bytes: number;
  role: string;
  registration_basis: 'geospatial' | 'pixel_grid';
  metadata: {
    width: number;
    height: number;
    crs?: string | null;
    warnings?: string[];
  } | null;
};

type Readiness = { status: string; checks: Record<string, boolean> };

const taskOptions: Array<{ value: Task; label: string; note: string }> = [
  { value: 'single_vqa', label: 'Ask one scene', note: 'Qwen3-VL · released' },
  {
    value: 'caption',
    label: 'Describe the scene',
    note: 'Qwen3-VL · released',
  },
  {
    value: 'grounding',
    label: 'Locate a feature',
    note: 'Qwen3-VL · released',
  },
  {
    value: 'change_vqa',
    label: 'Compare two dates',
    note: 'Local analytical baseline · runnable',
  },
  {
    value: 'optical_sar_fusion',
    label: 'Fuse optical + SAR',
    note: 'Local analytical baseline · runnable',
  },
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

function delay(milliseconds: number) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function EvidenceOverlay({ item }: { item: EvidenceItem }) {
  if (item.type === 'mask' && item.artifact_url?.startsWith('/v1/analyses/')) {
    return (
      <Image
        className="evidence-mask"
        src={`/api/satquery/${item.artifact_url.slice(4)}?colored=true`}
        alt={`${item.label} — unverified candidate mask`}
        fill
        unoptimized
        sizes="60vw"
      />
    );
  }
  if (item.type !== 'box' || item.coordinate_space !== 'normalized')
    return null;
  const [x, y, width, height] = ['x', 'y', 'width', 'height'].map((key) =>
    Number(item.geometry[key]),
  );
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
  const secondInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [secondFile, setSecondFile] = useState<File | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>('single');
  const [modality, setModality] = useState<Modality>('optical');
  const [route, setRoute] = useState<RouteChoice>('auto');
  const [query, setQuery] = useState(
    'Which land-cover features are visible in this scene?',
  );
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  const [altitude, setAltitude] = useState('');
  const [sensor, setSensor] = useState('');
  const [availableTasks, setAvailableTasks] = useState<Task[]>([
    'single_vqa',
    'caption',
    'grounding',
  ]);
  const [systemState, setSystemState] = useState<
    'checking' | 'ready' | 'degraded' | 'offline'
  >('checking');
  const [systemDetail, setSystemDetail] = useState(
    'Checking controller and model connection.',
  );
  const [assets, setAssets] = useState<AssetRecord[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState('');
  const [analysis, setAnalysis] = useState<AnalysisRecord | null>(null);
  const [phase, setPhase] = useState<
    'idle' | 'uploading' | 'working' | 'succeeded' | 'failed'
  >('idle');
  const [error, setError] = useState('');
  const [showOverlay, setShowOverlay] = useState(true);
  const [pixelAligned, setPixelAligned] = useState(false);
  const [inputProfile, setInputProfile] = useState<'strict' | 'exploration'>(
    'exploration',
  );
  const [outputView, setOutputView] = useState<'evidence' | 'report' | 'trace'>(
    'evidence',
  );
  const [sideBySide, setSideBySide] = useState(false);
  const [maskOpacity, setMaskOpacity] = useState(0.8);
  const displayFormats =
    inputProfile === 'exploration' &&
    (inputMode === 'fusion' || modality === 'optical');
  const acceptedFiles = displayFormats
    ? '.tif,.tiff,.jpg,.jpeg,.png,.webp'
    : '.tif,.tiff,image/tiff';

  const busy = phase === 'uploading' || phase === 'working';
  const result = analysis?.result;
  const asset = assets.find((item) => item.id === selectedAssetId) ?? assets[0];
  const previewSource = asset ? `/api/satquery/assets/${asset.id}/preview` : '';
  const visibleEvidence =
    result?.evidence.filter((item) => item.asset_id === asset?.id) ?? [];
  const ranGrounding = result?.trace.some((step) => step.task === 'grounding');
  const ranChange = result?.trace.some((step) => step.task === 'change_vqa');
  const previewStyle = asset?.metadata
    ? { aspectRatio: `${asset.metadata.width} / ${asset.metadata.height}` }
    : undefined;
  const selectedOption =
    route === 'auto'
      ? null
      : taskOptions.find((option) => option.value === route)!;
  const requiresPair = inputMode !== 'single';
  const filesReady = Boolean(file && (!requiresPair || secondFile));
  const compatibleTasks =
    inputMode === 'single'
      ? new Set<Task>(['single_vqa', 'caption', 'grounding'])
      : inputMode === 'temporal'
        ? new Set<Task>(['change_vqa'])
        : new Set<Task>(['optical_sar_fusion']);
  const progress = phase === 'uploading' ? 0.04 : (analysis?.progress ?? 0);
  const uncalibrated =
    result?.confidence.calibration_version.includes('uncalibrated');
  const confidenceLabel = result
    ? uncalibrated
      ? 'Accuracy not calibrated'
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
        const [readiness, capabilities] = await Promise.all([
          jsonRequest<Readiness>('/api/satquery/health/ready'),
          jsonRequest<{ tasks: Task[] }>('/api/satquery/capabilities'),
        ]);
        if (!active) return;
        setAvailableTasks(capabilities.tasks);
        const checks = Object.entries(readiness.checks ?? {});
        const ready =
          readiness.status === 'ok' &&
          checks.length > 0 &&
          checks.every(([, value]) => value === true);
        const failed = checks
          .filter(([, value]) => value !== true)
          .map(([name]) => name.replaceAll('_', ' '));
        setSystemState(ready ? 'ready' : 'degraded');
        setSystemDetail(
          ready
            ? 'Controller, database and model gateway are ready.'
            : `Controller reachable; not ready: ${failed.join(', ') || readiness.status}. Check the backend and Kaggle session.`,
        );
      } catch {
        if (!active) return;
        setSystemState('offline');
        setSystemDetail(
          'The local controller or its authenticated capabilities cannot be reached. Check the backend terminal and proxy configuration.',
        );
      }
    }
    void checkSystems();
    const interval = window.setInterval(() => void checkSystems(), 20_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  function chooseFile(
    event: ChangeEvent<HTMLInputElement>,
    slot: 'primary' | 'secondary',
  ) {
    const selected = event.target.files?.[0] ?? null;
    setError('');
    setAssets([]);
    setSelectedAssetId('');
    setAnalysis(null);
    setPhase('idle');
    if (!selected) {
      if (slot === 'primary') setFile(null);
      else setSecondFile(null);
      return;
    }
    if (
      !(
        displayFormats ? /\.(tif|tiff|jpg|jpeg|png|webp)$/i : /\.(tif|tiff)$/i
      ).test(selected.name)
    ) {
      if (slot === 'primary') setFile(null);
      else setSecondFile(null);
      setError(
        displayFormats
          ? 'Choose TIFF, JPG, PNG or WebP. Display images must be RGB or grayscale.'
          : 'This profile needs TIFF/GeoTIFF. For ordinary optical photos, select Exploration and Optical.',
      );
      event.target.value = '';
      return;
    }
    if (selected.size > 50 * 1024 * 1024) {
      if (slot === 'primary') setFile(null);
      else setSecondFile(null);
      setError(
        'The local profile accepts files up to 50 MB. Tile or crop this raster first.',
      );
      event.target.value = '';
      return;
    }
    if (slot === 'primary') setFile(selected);
    else setSecondFile(selected);
  }

  async function runAnalysis() {
    if (!file || (requiresPair && !secondFile) || busy) return;
    const expectedTask: Task =
      inputMode === 'single'
        ? 'single_vqa'
        : inputMode === 'temporal'
          ? 'change_vqa'
          : 'optical_sar_fusion';
    if (route !== 'auto' && !compatibleTasks.has(route)) {
      setError(
        'The selected route is incompatible with this evidence-set configuration.',
      );
      return;
    }
    if (!availableTasks.includes(route === 'auto' ? expectedTask : route)) {
      setError(
        'The required specialist is not available from the local controller.',
      );
      return;
    }
    if (query.trim().length < 2) {
      setError('Write a specific question before running the analysis.');
      return;
    }
    const hasPartialLocation = [latitude, longitude, altitude].some((value) =>
      value.trim(),
    );
    if (hasPartialLocation && (!latitude.trim() || !longitude.trim())) {
      setError(
        'Latitude and longitude must be supplied together. Altitude is optional.',
      );
      return;
    }
    const latitudeNumber = latitude.trim() ? Number(latitude) : null;
    const longitudeNumber = longitude.trim() ? Number(longitude) : null;
    const altitudeNumber = altitude.trim() ? Number(altitude) : null;
    if (
      latitudeNumber !== null &&
      (!Number.isFinite(latitudeNumber) ||
        latitudeNumber < -90 ||
        latitudeNumber > 90)
    ) {
      setError('Latitude must be a number between -90 and 90.');
      return;
    }
    if (
      longitudeNumber !== null &&
      (!Number.isFinite(longitudeNumber) ||
        longitudeNumber < -180 ||
        longitudeNumber > 180)
    ) {
      setError('Longitude must be a number between -180 and 180.');
      return;
    }
    if (
      altitudeNumber !== null &&
      (!Number.isFinite(altitudeNumber) ||
        altitudeNumber < -500 ||
        altitudeNumber > 100000)
    ) {
      setError('Altitude must be between -500 and 100000 metres.');
      return;
    }

    setError('');
    setAnalysis(null);
    setAssets([]);
    setSelectedAssetId('');
    setPhase('uploading');
    try {
      const upload = async (
        source: File,
        uploadModality: Modality,
        role: string,
      ) => {
        const form = new FormData();
        form.append('file', source);
        form.append('modality', uploadModality);
        form.append('role', role);
        form.append(
          'input_profile',
          inputProfile,
        );
        form.append(
          'registration_basis',
          requiresPair && pixelAligned
            ? 'pixel_grid'
            : 'geospatial',
        );
        return jsonRequest<AssetRecord>('/api/satquery/assets', {
          method: 'POST',
          body: form,
        });
      };
      const primaryRole =
        inputMode === 'temporal'
          ? 'time_a'
          : inputMode === 'fusion'
            ? 'optical'
            : 'primary';
      const uploads = [upload(file, modality, primaryRole)];
      if (requiresPair && secondFile) {
        uploads.push(
          upload(
            secondFile,
            inputMode === 'fusion' ? 'sar' : modality,
            inputMode === 'fusion' ? 'sar' : 'time_b',
          ),
        );
      }
      const uploaded = await Promise.all(uploads);
      setAssets(uploaded);
      setSelectedAssetId(uploaded[0].id);
      setPhase('working');

      const created = await jsonRequest<AnalysisRecord>(
        '/api/satquery/analyses',
        {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            'idempotency-key': crypto.randomUUID(),
          },
          body: JSON.stringify({
            query: query.trim(),
            asset_ids: uploaded.map((item) => item.id),
            requested_tasks: route === 'auto' ? null : [route],
            context:
              latitudeNumber !== null && longitudeNumber !== null
                ? {
                    latitude: latitudeNumber,
                    longitude: longitudeNumber,
                    altitude_m: altitudeNumber,
                    sensor: sensor.trim() || null,
                    source: 'user',
                    metadata: {},
                  }
                : null,
            parameters: {},
          }),
        },
      );
      setAnalysis(created);

      const deadline = Date.now() + 10 * 60 * 1000;
      let current = created;
      while (!['succeeded', 'failed', 'cancelled'].includes(current.status)) {
        if (Date.now() >= deadline) {
          throw new Error(
            'The model exceeded the ten-minute local job limit. Check the model-service terminal.',
          );
        }
        await delay(1200);
        current = await jsonRequest<AnalysisRecord>(
          `/api/satquery/analyses/${created.id}`,
        );
        setAnalysis(current);
      }
      if (current.status !== 'succeeded' || !current.result) {
        throw new Error(
          current.error_message || 'The analysis ended without a result.',
        );
      }
      const evidenceAsset = current.result.evidence.find((item) =>
        uploaded.some((source) => source.id === item.asset_id),
      );
      if (evidenceAsset) setSelectedAssetId(evidenceAsset.asset_id);
      setPhase('succeeded');
    } catch (caught) {
      setPhase('failed');
      setError(
        caught instanceof Error
          ? caught.message
          : 'Analysis failed unexpectedly.',
      );
    }
  }

  function clearFile() {
    setFile(null);
    setSecondFile(null);
    setAssets([]);
    setSelectedAssetId('');
    setAnalysis(null);
    setPhase('idle');
    if (inputRef.current) inputRef.current.value = '';
    if (secondInputRef.current) secondInputRef.current.value = '';
  }

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />

      <header className="topbar glass-panel">
        <a className="brand" href="#workspace" aria-label="SatQuery home">
          <span className="brand-mark">
            <Orbit />
          </span>
          <span>
            <strong>
              SATQUERY<span className="brand-period">.</span>
            </strong>
            <small>Earth evidence studio</small>
          </span>
        </a>
        <div className="project-switcher" aria-label="Current workspace">
          <span className="status-dot" />
          Local investigation workspace
        </div>
        <div className="topbar-actions">
          <span className={`live-pill ${systemState}`} title={systemDetail}>
            {systemState === 'checking' ? (
              <LoaderCircle size={13} />
            ) : (
              <RadioTower size={13} />
            )}
            {systemState === 'ready'
              ? 'Local controller ready'
              : systemState === 'checking'
                ? 'Checking systems'
                : systemState === 'degraded'
                  ? 'Controller reachable · not ready'
                  : 'Local services offline'}
          </span>
          <span className="zero-cost-badge">Local-first · ₹0</span>
        </div>
      </header>

      <section className="workspace" id="workspace">
        <aside className="rail glass-panel" aria-label="Workspace sections">
          <nav>
            <a
              className="rail-item active"
              href="#query-card"
              aria-label="Query workspace"
            >
              <ScanSearch />
              <span>Query</span>
            </a>
            <Link className="rail-item" href="/cases" aria-label="Casebook">
              <Boxes />
              <span>Cases</span>
            </Link>
            <Link
              className="rail-item"
              href="/archive"
              aria-label="Historical evidence"
            >
              <Map />
              <span>Archive</span>
            </Link>
            <Link
              className="rail-item"
              href="/method"
              aria-label="Methods and limits"
            >
              <Layers3 />
              <span>Methods</span>
            </Link>
          </nav>
          <div className="rail-orbit" aria-hidden="true">
            <span />
            <CircleDot />
          </div>
        </aside>

        <div className="main-column">
          <div className="eyebrow">01 / Investigation</div>
          <div className="heading-row">
            <div>
              <h1>Read the changing Earth.</h1>
              <p>Inspect a scene. Compare dates. Follow the evidence.</p>
            </div>
          </div>

          <section
            className="query-card glass-panel"
            id="query-card"
            aria-labelledby="query-title"
          >
            <div className="query-card-head">
              <div>
                <span className="step-number">01</span>
                <div>
                  <h2 id="query-title">Evidence set</h2>
                  <p>50 MB per file · originals preserved locally</p>
                </div>
              </div>
            </div>

            <div className="task-picker">
              <label htmlFor="input-profile">Input profile</label>
              <select
                id="input-profile"
                value={inputProfile}
                disabled={busy}
                onChange={(e) => {
                  setInputProfile(e.target.value as 'strict' | 'exploration');
                  clearFile();
                }}
              >
                <option value="exploration">
                  Exploration · JPG / PNG / WebP / TIFF
                </option>
                <option value="strict">SIH strict · geospatial TIFF</option>
              </select>
            </div>
            <p className="field-note">
              {inputMode === 'fusion'
                ? inputProfile === 'exploration'
                  ? 'Accepts a documented, co-registered optical/SAR display pair. Results use relative image intensity only—no calibrated backscatter or metric area.'
                  : 'Use genuine georeferenced optical and calibrated SAR sensor rasters, not two RGB photos.'
                : inputProfile === 'exploration'
                  ? 'RGB or grayscale images. Not the graded SIH benchmark profile. A photo without georeferencing cannot establish area or location.'
                  : 'Preserves the geospatial input contract. Non-TIFF benchmarks use the controlled API import.'}
            </p>

            <div className="task-picker evidence-mode-picker">
              <label htmlFor="input-mode">Evidence set</label>
              <select
                id="input-mode"
                value={inputMode}
                onChange={(event) => {
                  const nextMode = event.target.value as InputMode;
                  setInputMode(nextMode);
                  setModality(
                    nextMode === 'fusion' ? 'multispectral' : 'optical',
                  );
                  setRoute('auto');
                  setPixelAligned(false);
                  clearFile();
                }}
                disabled={busy}
              >
                <option value="single">
                  One optical, multispectral, or SAR scene
                </option>
                <option value="temporal">Bi-temporal co-registered pair</option>
                <option value="fusion">Co-registered optical + SAR pair</option>
              </select>
              <span>
                {inputMode === 'single' ? '1 raster' : '2 aligned rasters'}
              </span>
            </div>

            <div className="task-picker modality-picker">
              <label htmlFor="modality">
                {inputMode === 'temporal'
                  ? 'Pair modality'
                  : inputMode === 'fusion'
                    ? 'Optical modality'
                    : 'Scene modality'}
              </label>
              <select
                id="modality"
                value={modality}
                onChange={(event) =>
                  setModality(event.target.value as Modality)
                }
                disabled={busy}
              >
                {inputMode !== 'fusion' && (
                  <option value="optical">Optical</option>
                )}
                <option value="multispectral">Multispectral</option>
                {inputMode !== 'fusion' && <option value="sar">SAR</option>}
                {inputMode === 'fusion' && (
                  <option value="optical">Optical RGB</option>
                )}
              </select>
              <span>Declared, never guessed</span>
            </div>

            <input
              ref={inputRef}
              aria-label="Primary image file"
              className="sr-only"
              type="file"
              accept={acceptedFiles}
              onChange={(event) => chooseFile(event, 'primary')}
            />
            <input
              ref={secondInputRef}
              aria-label="Secondary image file"
              className="sr-only"
              type="file"
              accept={acceptedFiles}
              onChange={(event) => chooseFile(event, 'secondary')}
            />
            <div
              className={`upload-grid ${requiresPair ? '' : 'single-upload'}`}
            >
              <button
                className={`upload-tile ${file ? 'filled' : ''}`}
                type="button"
                onClick={() => inputRef.current?.click()}
                disabled={busy}
              >
                <span className={`file-icon ${file ? 'optical' : ''}`}>
                  {file ? <FileImage /> : <UploadCloud />}
                </span>
                <span className="upload-copy">
                  <strong>
                    {file
                      ? file.name
                      : inputMode === 'temporal'
                        ? 'Choose time A / before'
                        : inputMode === 'fusion'
                          ? 'Choose optical scene'
                          : 'Choose source scene'}
                  </strong>
                  <small>
                    {file
                      ? `${formatBytes(file.size)} · ready for validation`
                      : displayFormats
                        ? 'JPG · PNG · WebP · TIFF'
                        : 'Georeferenced TIFF / GeoTIFF'}
                  </small>
                </span>
                {file ? (
                  <span className="verified">Selected</span>
                ) : (
                  <ArrowUpRight size={17} />
                )}
              </button>
              {requiresPair && (
                <button
                  className={`upload-tile ${secondFile ? 'filled' : ''}`}
                  type="button"
                  onClick={() => secondInputRef.current?.click()}
                  disabled={busy}
                >
                  <span
                    className={`file-icon ${secondFile ? (inputMode === 'fusion' ? 'sar' : 'optical') : ''}`}
                  >
                    {secondFile ? <FileImage /> : <UploadCloud />}
                  </span>
                  <span className="upload-copy">
                    <strong>
                      {secondFile
                        ? secondFile.name
                        : inputMode === 'temporal'
                          ? 'Choose time B / after'
                          : 'Choose SAR scene'}
                    </strong>
                    <small>
                      {secondFile
                        ? `${formatBytes(secondFile.size)} · ready for pair validation`
                        : 'Same CRS, extent, resolution and grid'}
                    </small>
                  </span>
                  {secondFile ? (
                    <span className="verified">Selected</span>
                  ) : (
                    <ArrowUpRight size={17} />
                  )}
                </button>
              )}
              {!requiresPair && file && (
                <button
                  className="clear-file"
                  type="button"
                  onClick={clearFile}
                  disabled={busy}
                >
                  <X size={15} /> Remove scene
                </button>
              )}
            </div>
            {requiresPair && (file || secondFile) && (
              <button
                className="clear-file pair-clear"
                type="button"
                onClick={clearFile}
                disabled={busy}
              >
                <X size={15} /> Remove evidence set
              </button>
            )}
            {requiresPair && inputProfile === 'exploration' && (
              <label
                className="pixel-grid-consent"
                aria-label="Confirm pixel-for-pixel alignment"
              >
                <input
                  type="checkbox"
                  checked={pixelAligned}
                  onChange={(event) => setPixelAligned(event.target.checked)}
                  disabled={busy}
                />
                <span>
                  <strong>These images are aligned pixel-for-pixel</strong>
                  <small>
                    For documented paired exports without georeferencing.
                    Matching size alone is not proof of alignment. No geographic
                    area or calibrated SAR value will be claimed.
                  </small>
                </span>
              </label>
            )}

            <div className="task-picker">
              <label htmlFor="task">Specialist route</label>
              <select
                id="task"
                value={route}
                onChange={(event) =>
                  setRoute(event.target.value as RouteChoice)
                }
                disabled={busy}
              >
                <option value="auto">
                  Auto route from question + validated inputs
                </option>
                {taskOptions.map((option) => (
                  <option
                    key={option.value}
                    value={option.value}
                    disabled={
                      !availableTasks.includes(option.value) ||
                      !compatibleTasks.has(option.value)
                    }
                  >
                    {option.label} — {option.note}
                  </option>
                ))}
              </select>
              <span>
                {selectedOption?.note || 'Policy router · observable'}
              </span>
            </div>

            <details className="context-details">
              <summary>
                Location context <span>optional · user supplied</span>
              </summary>
              <fieldset className="location-context">
                <legend>
                  Location context <span>optional but recommended</span>
                </legend>
                <label>
                  Latitude
                  <input
                    inputMode="decimal"
                    value={latitude}
                    onChange={(event) => setLatitude(event.target.value)}
                    placeholder="28.6139"
                    disabled={busy}
                  />
                </label>
                <label>
                  Longitude
                  <input
                    inputMode="decimal"
                    value={longitude}
                    onChange={(event) => setLongitude(event.target.value)}
                    placeholder="77.2090"
                    disabled={busy}
                  />
                </label>
                <label>
                  Altitude (m)
                  <input
                    inputMode="decimal"
                    value={altitude}
                    onChange={(event) => setAltitude(event.target.value)}
                    placeholder="216"
                    disabled={busy}
                  />
                </label>
                <label>
                  Sensor
                  <input
                    value={sensor}
                    onChange={(event) => setSensor(event.target.value)}
                    placeholder="Sentinel-2 / drone"
                    maxLength={120}
                    disabled={busy}
                  />
                </label>
                <p>
                  Stored as user-provided metadata; the model is never allowed
                  to present it as pixel-derived evidence.
                </p>
              </fieldset>
            </details>

            <div className="prompt-box">
              <label htmlFor="query">Your investigation</label>
              <div
                className="question-presets"
                aria-label="Investigation presets"
              >
                {[
                  [
                    'Water',
                    'Outline visible water bodies. Describe their spatial pattern, possible confounders and unresolved questions.',
                  ],
                  [
                    'Forestry',
                    'Outline visible vegetation. Describe canopy pattern, bare-ground patches and visible fragmentation. Do not invent species or causes.',
                  ],
                  [
                    'Journalism',
                    'Describe the visible scene for an evidence-led report. Separate observations from hypotheses; identify what cannot be confirmed.',
                  ],
                  [
                    'Change',
                    'Compare the before and after scenes. Mark changed pixels, quantify the candidate coverage and explain cloud, seasonal and registration limitations.',
                  ],
                ].map(([label, prompt]) => (
                  <button
                    key={label}
                    type="button"
                    disabled={busy}
                    onClick={() => setQuery(prompt)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <textarea
                id="query"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                disabled={busy}
                maxLength={2000}
                aria-describedby="query-help"
              />
              <div className="prompt-footer">
                <div
                  className="chips"
                  id="query-help"
                  aria-label="Execution properties"
                >
                  <span>{route.replaceAll('_', ' ')}</span>
                  <span>{inputMode.replaceAll('_', ' ')}</span>
                  <span>trace on</span>
                </div>
                <Button
                  className="run-button"
                  size="lg"
                  onClick={runAnalysis}
                  disabled={!filesReady || busy}
                >
                  {busy ? (
                    <>
                      <LoaderCircle className="spin" />{' '}
                      {phase === 'uploading' ? 'Validating' : 'Analyzing'}
                    </>
                  ) : (
                    <>
                      Run analysis <ArrowUpRight />
                    </>
                  )}
                </Button>
              </div>
            </div>

            {(busy || analysis) && (
              <output className="run-progress" aria-live="polite">
                <div>
                  <span>
                    {phase === 'uploading'
                      ? 'Uploading safely'
                      : analysis
                        ? statusCopy[analysis.status]
                        : 'Starting'}
                  </span>
                  <strong>{Math.round(progress * 100)}%</strong>
                </div>
                <div className="progress-track">
                  <span style={{ width: `${Math.max(3, progress * 100)}%` }} />
                </div>
              </output>
            )}
            {error && (
              <div className="error-banner" role="alert">
                <AlertTriangle size={16} />
                <span>{error}</span>
              </div>
            )}
            {(systemState === 'degraded' || systemState === 'offline') && (
              <output className="error-banner">
                <AlertTriangle size={16} />
                <span>
                  {systemDetail}
                  {systemState === 'degraded'
                    ? ' Local pair tools may still be available.'
                    : ''}
                </span>
              </output>
            )}
          </section>
        </div>

        <aside className="insight-column" aria-label="Analysis output">
          <div className="output-tabs" aria-label="Output view">
            {(['evidence', 'report', 'trace'] as const).map((view) => (
              <button
                key={view}
                type="button"
                aria-pressed={outputView === view}
                onClick={() => setOutputView(view)}
              >
                {view === 'evidence'
                  ? 'Scene inspector'
                  : view === 'report'
                    ? 'Analysis report'
                    : 'Execution trace'}
              </button>
            ))}
            {analysis && <a href={`/cases/${analysis.id}`}>Open case ↗</a>}
          </div>
          <section
            hidden={outputView !== 'evidence'}
            className="evidence-card glass-panel"
            id="evidence"
          >
            <div className="section-label">
              <span>
                <span
                  className={`pulse ${phase === 'succeeded' ? 'complete' : ''}`}
                />{' '}
                Evidence canvas
              </span>
              <span>{visibleEvidence.length} regions in selected scene</span>
            </div>
            {assets.length > 0 && (
              <div className="evidence-selector">
                <label htmlFor="evidence-asset">Preview source</label>
                <select
                  id="evidence-asset"
                  value={asset?.id ?? ''}
                  onChange={(event) => setSelectedAssetId(event.target.value)}
                >
                  {assets.map((source) => (
                    <option key={source.id} value={source.id}>
                      {source.role.replaceAll('_', ' ')} ·{' '}
                      {source.original_name}
                    </option>
                  ))}
                </select>
              </div>
            )}
            {assets.length === 2 && (
              <label className="pair-view-toggle">
                <input
                  type="checkbox"
                  checked={sideBySide}
                  onChange={(e) => setSideBySide(e.target.checked)}
                />{' '}
                Compare sources side by side
              </label>
            )}
            {sideBySide && assets.length === 2 ? (
              <div className="paired-canvases">
                {assets.map((source) => (
                  <figure key={source.id}>
                    <figcaption>
                      {source.role.replaceAll('_', ' ')} ·{' '}
                      {source.original_name}
                    </figcaption>
                    <div
                      className="evidence-map has-preview"
                      style={
                        {
                          aspectRatio: `${source.metadata?.width ?? 1}/${source.metadata?.height ?? 1}`,
                          '--mask-opacity': maskOpacity,
                        } as CSSProperties
                      }
                    >
                      <Image
                        src={`/api/satquery/assets/${source.id}/preview`}
                        alt={source.original_name}
                        fill
                        unoptimized
                        sizes="40vw"
                      />
                      {showOverlay &&
                        result?.evidence
                          .filter((item) => item.asset_id === source.id)
                          .map((item) => (
                            <EvidenceOverlay key={item.id} item={item} />
                          ))}
                    </div>
                  </figure>
                ))}
              </div>
            ) : (
              <figure
                className={`evidence-map ${asset ? 'has-preview' : ''}`}
                style={
                  {
                    ...previewStyle,
                    '--mask-opacity': maskOpacity,
                  } as CSSProperties
                }
                aria-label={
                  asset
                    ? 'Uncropped raster preview with evidence for this source only'
                    : 'Empty satellite evidence canvas'
                }
              >
                {asset && previewSource ? (
                  <Image
                    src={previewSource}
                    alt={`Uncropped preview of ${asset.original_name}`}
                    fill
                    sizes="(max-width: 1050px) 60vw, 38vw"
                    unoptimized
                  />
                ) : (
                  <div className="empty-orbit" aria-hidden="true">
                    <Orbit />
                    <span>Awaiting scene</span>
                  </div>
                )}
                <div className="map-grid" />
                {showOverlay &&
                  visibleEvidence.map((item) => (
                    <EvidenceOverlay key={item.id} item={item} />
                  ))}
                {!asset && (
                  <div className="map-coordinates">
                    No source loaded · no invented map
                  </div>
                )}
              </figure>
            )}
            {result && (
              <div className="mask-controls">
                <label>
                  <input
                    type="checkbox"
                    checked={showOverlay}
                    onChange={(event) => setShowOverlay(event.target.checked)}
                  />{' '}
                  Show candidate overlays
                </label>
                <label>
                  Overlay opacity
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={maskOpacity}
                    onChange={(e) => setMaskOpacity(Number(e.target.value))}
                  />
                </label>
                <p>
                  {visibleEvidence.some((item) => item.type === 'mask')
                    ? 'Cyan pixels are mask predictions, not verified boundaries. Toggle off to inspect the source. Areas measure the mask, not ground truth.'
                    : visibleEvidence.length
                      ? 'Coarse model boxes only — not pixel-level segmentation.'
                      : result.evidence.length
                        ? 'The returned overlay belongs to the other source image. Change Preview source to inspect its reference grid.'
                        : ranChange
                          ? 'No shared valid pixels exceeded the change threshold. This does not prove absence of physical change; inspect the paired report and source images.'
                          : !ranGrounding
                            ? 'This run selected text-only analysis, so no mask was requested. Choose Auto or Locate a feature and run the analysis again to request an overlay.'
                            : 'The grounding step ran but returned no usable mask or region. Check the model warnings below for details.'}
                </p>
                {visibleEvidence
                  .filter((item) => item.type === 'mask')
                  .map((item) => (
                    <div key={item.id} className="mask-stat">
                      <strong>{item.label}</strong>
                      <span>
                        {Number(item.geometry.coverage_percent ?? 0).toFixed(2)}
                        % of valid grid · {String(item.geometry.width)} ×{' '}
                        {String(item.geometry.height)} mask pixels
                        {item.geometry.resampled ? ' · resampled' : ''}
                      </span>
                      {item.artifact_url?.startsWith('/v1/analyses/') && (
                        <a
                          href={`/api/satquery/${item.artifact_url.slice(4)}`}
                          download={`${item.id}.png`}
                        >
                          Download binary mask (PNG)
                        </a>
                      )}
                    </div>
                  ))}
              </div>
            )}
          </section>

          {outputView === 'evidence' && (
            <div className="inspector-summary">
              <span>
                {phase === 'succeeded'
                  ? 'Analysis ready'
                  : 'Evidence before interpretation'}
              </span>
              <p>
                {result
                  ? (() => {
                      const clean = result.answer
                        .replace(/^#+\s+/gm, '')
                        .replace(/\|[^\n]+\|/g, '')
                        .replace(/[*`_~]/g, '')
                        .replace(/\s+/g, ' ')
                        .trim();
                      return clean.slice(0, 240) + (clean.length > 240 ? '…' : '');
                    })()
                  : 'Upload a source on the left. Candidate masks, measured coverage and source-aligned comparison appear here.'}
              </p>
              <button type="button" onClick={() => setOutputView('report')}>
                Read full report →
              </button>
            </div>
          )}
          <section
            hidden={outputView !== 'report'}
            className={`answer-card glass-panel ${phase}`}
            id="answer"
            aria-live="polite"
          >
            <div className="answer-topline">
              <span>
                {phase === 'succeeded'
                  ? 'Model result'
                  : busy
                    ? 'Analysis in progress'
                    : phase === 'failed'
                      ? 'Run interrupted'
                      : 'Awaiting analysis'}
              </span>
              <strong>{confidenceLabel}</strong>
            </div>
            <h2>
              {result
                ? 'Visual interpretation'
                : busy
                  ? analysis
                    ? statusCopy[analysis.status]
                    : 'Inspecting the upload…'
                  : phase === 'failed'
                    ? 'No model answer was produced.'
                    : 'Your evidence-backed answer will appear here.'}
            </h2>
            {result && (
              <StructuredAnswer
                content={result.answer}
                className="narrative-output"
              />
            )}
            {result?.sections?.map((section, index) => (
              <section
                className="region-report-section"
                key={`${section.title}-${index}`}
              >
                <h3>{section.title}</h3>
                <small>{section.source}</small>
                {section.paragraphs.map((paragraph, paragraphIndex) => (
                  <StructuredAnswer key={paragraphIndex} content={paragraph} />
                ))}
              </section>
            ))}
            <p>
              {result?.confidence.meaning ||
                'Scores stay hidden until a real specialist returns; SatQuery never invents a confidence percentage.'}
            </p>
            {result && !uncalibrated && (
              <div className="confidence-track">
                <span style={{ width: `${result.confidence.score * 100}%` }} />
              </div>
            )}
            {result?.warnings.length ? (
              <details className="warning-list">
                <summary>
                  {result.warnings.length} model warning
                  {result.warnings.length === 1 ? '' : 's'}
                </summary>
                <ul>
                  {result.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </details>
            ) : null}
            {result?.facts.length ? (
              <details className="structured-facts">
                <summary>
                  Structured observations and provenance ({result.facts.length})
                </summary>
                <pre>{JSON.stringify(result.facts, null, 2)}</pre>
              </details>
            ) : null}
            {analysis?.status === 'succeeded' && (
              <div className="artifact-links">
                <a
                  className="report-link"
                  href={`/api/satquery/analyses/${analysis.id}/overlay${asset ? `?asset_id=${encodeURIComponent(asset.id)}` : ''}`}
                >
                  <ArrowDownToLine size={14} /> Download selected marked image
                </a>
                {result?.report_url && (
                  <a
                    className="report-link"
                    href={`/api/satquery/analyses/${analysis.id}/report`}
                  >
                    <ArrowDownToLine size={14} /> Download audit report
                  </a>
                )}
              </div>
            )}
          </section>

          <section
            hidden={outputView !== 'trace'}
            className="trace-card glass-panel"
            id="trace"
          >
            <div className="section-label">
              <span>Observable execution trace</span>
              <span>{displayTrace.length} events</span>
            </div>
            {displayTrace.length ? (
              <ol>
                {displayTrace.map((step, index) => (
                  <li key={`${step.step_id}-${index}`}>
                    <span
                      className={`trace-node ${step.status === 'succeeded' ? 'mint' : step.status === 'failed' ? 'coral' : 'blue'}`}
                    >
                      {step.status === 'succeeded' ? (
                        <Check size={11} />
                      ) : (
                        index + 1
                      )}
                    </span>
                    <span>
                      <strong>{step.tool.replaceAll('-', ' ')}</strong>
                      <small>
                        {step.task} · {step.model_version || 'controller'}{' '}
                        {step.duration_ms ? `· ${step.duration_ms} ms` : ''}
                      </small>
                      <em>{step.policy_reason}</em>
                      <code className="trace-params">
                        Permitted parameters:{' '}
                        {JSON.stringify(step.permitted_params ?? {})}
                      </code>
                    </span>
                  </li>
                ))}
              </ol>
            ) : (
              <div className="trace-empty">
                Validation, routing, model revision and timing will be recorded
                here—never hidden reasoning.
              </div>
            )}
          </section>
        </aside>
      </section>
    </main>
  );
}
