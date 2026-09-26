'use client';

import {
  Activity,
  AlertTriangle,
  ArrowDownToLine,
  ArrowUpRight,
  BarChart3,
  Check,
  FileClock,
  FileImage,
  Focus,
  Layers3,
  LoaderCircle,
  MoveHorizontal,
  Minus,
  Orbit,
  Plus,
  RadioTower,
  ShieldCheck,
  UploadCloud,
  X,
} from 'lucide-react';
import Image from 'next/image';
import {
  ChangeEvent,
  CSSProperties,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { StructuredAnswer } from '@/components/structured-answer';
import { StudioNav } from '@/components/studio-nav';
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
    correctness_probability?: number | null;
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
  { value: 'single_vqa', label: 'Ask one scene', note: 'Vision-language route' },
  {
    value: 'caption',
    label: 'Describe the scene',
    note: 'Vision-language route',
  },
  {
    value: 'grounding',
    label: 'Locate a feature',
    note: 'Grounding and mask route',
  },
  {
    value: 'change_vqa',
    label: 'Compare two dates',
    note: 'Controller-advertised paired route',
  },
  {
    value: 'optical_sar_fusion',
    label: 'Fuse optical + SAR',
    note: 'Controller-advertised fusion route',
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

const evidencePalette: Record<string, string> = {
  water: '#2dcef2',
  vegetation: '#63e39a',
  'built-up': '#ffc857',
  'burn-scar': '#f06a6a',
  change: '#c78cff',
  unknown: '#8bd0ff',
};

function evidenceColor(item: EvidenceItem) {
  const rawClass = item.geometry.class_name;
  const declared = typeof rawClass === 'string' ? rawClass.toLowerCase() : '';
  return evidencePalette[declared] ?? evidencePalette.unknown;
}

function vectorPaths(item: EvidenceItem) {
  if (!Array.isArray(item.geometry.polygons)) return [];
  return item.geometry.polygons.flatMap((candidate, polygonIndex) => {
    if (!candidate || typeof candidate !== 'object') return [];
    const rings = (candidate as { rings?: unknown }).rings;
    if (!Array.isArray(rings)) return [];
    const commands = rings
      .map((ring) => {
        if (!Array.isArray(ring)) return '';
        const points = ring
          .map((point) => {
            if (!Array.isArray(point) || point.length !== 2) return null;
            const x = Number(point[0]);
            const y = Number(point[1]);
            return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
          })
          .filter((point): point is number[] => point !== null);
        if (points.length < 4) return '';
        return `M ${points.map(([x, y]) => `${x} ${y}`).join(' L ')} Z`;
      })
      .filter(Boolean)
      .join(' ');
    return commands ? [{ id: `${item.id}-${polygonIndex}`, commands }] : [];
  });
}

function EvidenceOverlay({ item }: { item: EvidenceItem }) {
  if (item.type === 'mask' && item.artifact_url?.startsWith('/v1/analyses/')) {
    const paths = vectorPaths(item);
    if (paths.length) {
      const color = evidenceColor(item);
      return (
        <svg
          className="evidence-vector"
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          aria-label={`${item.label} — candidate polygon overlay`}
        >
          <title>{`${item.label} — candidate polygon overlay`}</title>
          {paths.map((path) => (
            <path
              key={path.id}
              d={path.commands}
              fill={color}
              fillRule="evenodd"
              stroke={color}
              strokeWidth="0.0025"
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      );
    }
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

function ScenePixelLayer({
  previewSource,
  alt,
  evidence,
  showOverlay,
  zoom,
  sizes,
}: {
  previewSource: string;
  alt: string;
  evidence: EvidenceItem[];
  showOverlay: boolean;
  zoom: number;
  sizes: string;
}) {
  return (
    <div
      className="scene-pixel-layer"
      style={{ transform: `scale(${zoom})` }}
    >
      <Image
        src={previewSource}
        alt={alt}
        fill
        unoptimized
        sizes={sizes}
      />
      {showOverlay &&
        evidence.map((item) => <EvidenceOverlay key={item.id} item={item} />)}
    </div>
  );
}

export default function Home() {
  const canvasRef = useRef<HTMLElement>(null);
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
  const [sideBySide, setSideBySide] = useState(false);
  const [pairSplit, setPairSplit] = useState(50);
  const [maskOpacity, setMaskOpacity] = useState(0.8);
  const [canvasZoom, setCanvasZoom] = useState(1);
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
    result?.confidence.correctness_probability == null;
  const confidenceLabel = result
    ? uncalibrated
      ? 'Accuracy not calibrated'
      : `${Math.round((result.confidence.correctness_probability ?? 0) * 100)}% calibrated`
    : 'No score calculated';
  const maskEvidence = visibleEvidence.filter((item) => item.type === 'mask');
  const measuredArea = maskEvidence.find(
    (item) =>
      typeof item.geometry.area_m2 === 'number' &&
      Number.isFinite(item.geometry.area_m2),
  );
  const coverageEvidence = maskEvidence.find(
    (item) =>
      typeof item.geometry.coverage_percent === 'number' &&
      Number.isFinite(item.geometry.coverage_percent),
  );
  const evidenceClasses = Array.from(
    new Set(
      visibleEvidence.map((item) => {
        const value = item.geometry.class_name;
        return typeof value === 'string' && value.trim()
          ? value.trim().toLowerCase()
          : 'unknown';
      }),
    ),
  );

  async function toggleCanvasFullscreen() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (document.fullscreenElement) await document.exitFullscreen();
    else await canvas.requestFullscreen();
  }

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
    setPairSplit(50);
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
    setPairSplit(50);
    setSideBySide(false);
    if (inputRef.current) inputRef.current.value = '';
    if (secondInputRef.current) secondInputRef.current.value = '';
  }

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />

      <StudioNav active="/workspace" />

      <header className="topbar glass-panel">
        <div className="workspace-context">
          <span>Orbital intelligence</span>
          <strong>Evidence workspace</strong>
        </div>
        <div className="project-switcher" aria-label="Current workspace">
          <span>Active case</span>
          <strong>{analysis ? analysis.id : 'New investigation'}</strong>
        </div>
        <div className="topbar-actions">
          <span className={`live-pill ${systemState}`} title={systemDetail}>
            {systemState === 'checking' ? (
              <LoaderCircle size={13} />
            ) : (
              <RadioTower size={13} />
            )}
            {systemState === 'ready'
              ? 'Services online'
              : systemState === 'checking'
                ? 'Checking systems'
                : systemState === 'degraded'
                  ? 'Controller reachable · not ready'
                  : 'Local services offline'}
          </span>
          <span className="zero-cost-badge">Local-first · zero-cost path</span>
        </div>
      </header>

      <section className="workspace" id="workspace">
        <div className="main-column">
          <div className="eyebrow">Source workspace</div>
          <div className="heading-row">
            <div>
              <span className="workspace-kicker">New investigation</span>
              <h1>Evidence desk</h1>
              <p>Build an inspectable source set before interpretation.</p>
            </div>
          </div>

          <div className="workflow-ribbon" aria-label="Analysis workflow">
            <div className="active"><span>01</span><strong>Configure</strong></div>
            <i aria-hidden="true" />
            <div className={file ? 'active' : ''}><span>02</span><strong>Attach</strong></div>
            <i aria-hidden="true" />
            <div className={analysis ? 'active' : ''}><span>03</span><strong>Interpret</strong></div>
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
                  <h2 id="query-title">Evidence contract</h2>
                  <p>50 MB per file · originals preserved locally</p>
                </div>
              </div>
            </div>

            <div className="task-picker profile-picker">
              <label htmlFor="input-profile">Input profile</label>
              <Select
                value={inputProfile}
                disabled={busy}
                onValueChange={(value) => {
                  if (!value) return;
                  setInputProfile(value as 'strict' | 'exploration');
                  clearFile();
                }}
              >
                <SelectTrigger id="input-profile" className="studio-select-trigger">
                  <SelectValue>{inputProfile === 'exploration' ? 'Exploration · common formats' : 'SIH strict · GeoTIFF'}</SelectValue>
                </SelectTrigger>
                <SelectContent className="studio-select-content" align="start">
                  <SelectItem value="exploration" className="studio-select-item">
                    <span className="select-option-copy"><strong>Exploration</strong><small>JPG · PNG · WebP · TIFF</small></span>
                  </SelectItem>
                  <SelectItem value="strict" className="studio-select-item">
                    <span className="select-option-copy"><strong>SIH strict</strong><small>Geospatial TIFF contract</small></span>
                  </SelectItem>
                </SelectContent>
              </Select>
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
              <Select
                value={inputMode}
                onValueChange={(value) => {
                  if (!value) return;
                  const nextMode = value as InputMode;
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
                <SelectTrigger id="input-mode" className="studio-select-trigger">
                  <SelectValue>{inputMode === 'single' ? 'Single scene' : inputMode === 'temporal' ? 'Bi-temporal pair' : 'Optical + SAR pair'}</SelectValue>
                </SelectTrigger>
                <SelectContent className="studio-select-content" align="start">
                  <SelectItem value="single" className="studio-select-item"><span className="select-option-copy"><strong>Single scene</strong><small>Optical, multispectral or SAR</small></span></SelectItem>
                  <SelectItem value="temporal" className="studio-select-item"><span className="select-option-copy"><strong>Bi-temporal pair</strong><small>Co-registered before and after</small></span></SelectItem>
                  <SelectItem value="fusion" className="studio-select-item"><span className="select-option-copy"><strong>Optical + SAR</strong><small>Co-registered sensor pair</small></span></SelectItem>
                </SelectContent>
              </Select>
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
              <Select
                value={modality}
                onValueChange={(value) => value && setModality(value as Modality)}
                disabled={busy}
              >
                <SelectTrigger id="modality" className="studio-select-trigger">
                  <SelectValue>{inputMode === 'fusion' && modality === 'optical' ? 'Optical RGB' : modality === 'sar' ? 'SAR' : modality === 'multispectral' ? 'Multispectral' : 'Optical'}</SelectValue>
                </SelectTrigger>
                <SelectContent className="studio-select-content" align="start">
                {inputMode !== 'fusion' && (
                  <SelectItem value="optical" className="studio-select-item"><span className="select-option-copy"><strong>Optical</strong><small>Visible colour imagery</small></span></SelectItem>
                )}
                <SelectItem value="multispectral" className="studio-select-item"><span className="select-option-copy"><strong>Multispectral</strong><small>Declared spectral bands</small></span></SelectItem>
                {inputMode !== 'fusion' && <SelectItem value="sar" className="studio-select-item"><span className="select-option-copy"><strong>SAR</strong><small>Radar intensity imagery</small></span></SelectItem>}
                {inputMode === 'fusion' && (
                  <SelectItem value="optical" className="studio-select-item"><span className="select-option-copy"><strong>Optical RGB</strong><small>Visible reference source</small></span></SelectItem>
                )}
                </SelectContent>
              </Select>
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

            <div className="task-picker route-picker">
              <label htmlFor="task">Specialist route</label>
              <Select
                value={route}
                onValueChange={(value) => value && setRoute(value as RouteChoice)}
                disabled={busy}
              >
                <SelectTrigger id="task" className="studio-select-trigger">
                  <SelectValue>{route === 'auto' ? 'Automatic specialist' : selectedOption?.label}</SelectValue>
                </SelectTrigger>
                <SelectContent className="studio-select-content route-select-content" align="start">
                  <SelectItem value="auto" className="studio-select-item"><span className="select-option-copy"><strong>Automatic specialist</strong><small>Question + validated input routing</small></span></SelectItem>
                  {taskOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value} disabled={!availableTasks.includes(option.value) || !compatibleTasks.has(option.value)} className="studio-select-item">
                      <span className="select-option-copy"><strong>{option.label}</strong><small>{option.note}</small></span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
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
                placeholder="Ask about visible land cover, locate a feature, compare aligned dates, or fuse optical and SAR evidence…"
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
          <div className="studio-commandbar">
            <div>
              <span className="eyebrow">Orbital Intelligence Studio</span>
              <strong>
                {asset ? asset.original_name : 'No evidence loaded'}
              </strong>
            </div>
            <div>
              {asset?.metadata?.crs && <code>{asset.metadata.crs}</code>}
              {analysis && <a href={`/cases/${analysis.id}`}>Open case ↗</a>}
            </div>
          </div>
          <section
            className="evidence-card glass-panel"
            id="evidence"
            ref={canvasRef}
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
            <div className="canvas-toolbar" aria-label="Scene controls">
              <button
                type="button"
                onClick={() => setCanvasZoom((value) => Math.max(1, value - 0.25))}
                disabled={!asset || canvasZoom <= 1}
                aria-label="Zoom out"
              >
                <Minus />
              </button>
              <output>{Math.round(canvasZoom * 100)}%</output>
              <button
                type="button"
                onClick={() => setCanvasZoom((value) => Math.min(3, value + 0.25))}
                disabled={!asset || canvasZoom >= 3}
                aria-label="Zoom in"
              >
                <Plus />
              </button>
              <button
                type="button"
                onClick={() => setCanvasZoom(1)}
                disabled={!asset}
                aria-label="Reset scene zoom"
              >
                <Focus />
              </button>
              <button
                type="button"
                onClick={() => void toggleCanvasFullscreen()}
                disabled={!asset}
                aria-label="Open scene fullscreen"
              >
                <Layers3 />
              </button>
            </div>
            {assets.length > 0 && (
              <div className="evidence-selector">
                <label htmlFor="evidence-asset">Preview source</label>
                <Select
                  value={asset?.id ?? ''}
                  onValueChange={(value) => value && setSelectedAssetId(value)}
                >
                  <SelectTrigger id="evidence-asset" className="canvas-select-trigger"><SelectValue>{asset ? `${asset.role.replaceAll('_', ' ')} · ${asset.original_name}` : 'Select source'}</SelectValue></SelectTrigger>
                  <SelectContent className="studio-select-content" align="start">
                  {assets.map((source) => (
                    <SelectItem className="studio-select-item" key={source.id} value={source.id}><span className="select-option-copy"><strong>{source.role.replaceAll('_', ' ')}</strong><small>{source.original_name}</small></span></SelectItem>
                  ))}
                  </SelectContent>
                </Select>
              </div>
            )}
            {assets.length === 2 && (
              <fieldset className="pair-view-toggle">
                <legend className="sr-only">Pair display mode</legend>
                <button
                  type="button"
                  aria-pressed={!sideBySide}
                  onClick={() => setSideBySide(false)}
                >
                  Swipe compare
                </button>
                <button
                  type="button"
                  aria-pressed={sideBySide}
                  onClick={() => setSideBySide(true)}
                >
                  Side by side
                </button>
              </fieldset>
            )}
            {assets.length === 2 && !sideBySide ? (
              <div className="pair-comparison-shell">
                <div
                  className="pair-comparison"
                  style={{
                    aspectRatio: `${assets[0].metadata?.width ?? 1}/${assets[0].metadata?.height ?? 1}`,
                    '--mask-opacity': maskOpacity,
                  } as CSSProperties}
                  >
                    <div className="pair-comparison-layer">
                    <ScenePixelLayer
                      previewSource={`/api/satquery/assets/${assets[0].id}/preview`}
                      alt={`${assets[0].role.replaceAll('_', ' ')} source: ${assets[0].original_name}`}
                      evidence={result?.evidence.filter((item) => item.asset_id === assets[0].id) ?? []}
                      showOverlay={showOverlay}
                      zoom={canvasZoom}
                      sizes="(max-width: 1080px) 90vw, 55vw"
                    />
                  </div>
                  <div
                    className="pair-comparison-layer pair-comparison-reveal"
                    style={{ clipPath: `inset(0 0 0 ${pairSplit}%)` }}
                  >
                    <ScenePixelLayer
                      previewSource={`/api/satquery/assets/${assets[1].id}/preview`}
                      alt={`${assets[1].role.replaceAll('_', ' ')} source: ${assets[1].original_name}`}
                      evidence={result?.evidence.filter((item) => item.asset_id === assets[1].id) ?? []}
                      showOverlay={showOverlay}
                      zoom={canvasZoom}
                      sizes="(max-width: 1080px) 90vw, 55vw"
                    />
                  </div>
                  <div className="pair-comparison-labels" aria-hidden="true">
                    <span>{assets[0].role.replaceAll('_', ' ')}</span>
                    <span>{assets[1].role.replaceAll('_', ' ')}</span>
                  </div>
                  <div
                    className="pair-comparison-divider"
                    style={{ left: `${pairSplit}%` }}
                    aria-hidden="true"
                  >
                    <span><MoveHorizontal /></span>
                  </div>
                </div>
                <label className="pair-comparison-control">
                  <span>Compare sources</span>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={pairSplit}
                    onChange={(event) => setPairSplit(Number(event.target.value))}
                    aria-label={`Compare ${assets[0].role.replaceAll('_', ' ')} and ${assets[1].role.replaceAll('_', ' ')}`}
                    aria-valuetext={`${pairSplit}% ${assets[0].role.replaceAll('_', ' ')} visible`}
                  />
                  <output>{pairSplit}%</output>
                </label>
              </div>
            ) : sideBySide && assets.length === 2 ? (
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
                      <ScenePixelLayer
                        previewSource={`/api/satquery/assets/${source.id}/preview`}
                        alt={source.original_name}
                        evidence={result?.evidence.filter((item) => item.asset_id === source.id) ?? []}
                        showOverlay={showOverlay}
                        zoom={canvasZoom}
                        sizes="40vw"
                      />
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
                  <ScenePixelLayer
                    previewSource={previewSource}
                    alt={`Uncropped preview of ${asset.original_name}`}
                    evidence={visibleEvidence}
                    showOverlay={showOverlay}
                    zoom={canvasZoom}
                    sizes="(max-width: 1050px) 60vw, 38vw"
                  />
                ) : (
                  <div className="empty-canvas-state">
                    <span className="canvas-corner corner-nw" aria-hidden="true" />
                    <span className="canvas-corner corner-ne" aria-hidden="true" />
                    <span className="canvas-corner corner-sw" aria-hidden="true" />
                    <span className="canvas-corner corner-se" aria-hidden="true" />
                    <div className="empty-scan-field" aria-hidden="true">
                      <span className="orbit-ring ring-one" />
                      <span className="orbit-ring ring-two" />
                      <span className="orbit-ring ring-three" />
                      <span className="orbit-axis axis-x" />
                      <span className="orbit-axis axis-y" />
                      <span className="orbit-node node-one" />
                      <span className="orbit-node node-two" />
                      <span className="empty-signal-core"><Orbit /></span>
                    </div>
                    <div className="empty-canvas-copy">
                      <span>Evidence channel open</span>
                      <strong>Bring a scene into focus.</strong>
                      <p>Attach one source—or a verified pair—to unlock source-locked inspection.</p>
                      <button type="button" onClick={() => inputRef.current?.click()} disabled={busy}>
                        <UploadCloud /> Choose evidence
                      </button>
                    </div>
                    <div className="canvas-capabilities" aria-label="Accepted evidence modes">
                      <span>Single scene</span><i />
                      <span>Temporal pair</span><i />
                      <span>Optical + SAR</span>
                    </div>
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
                {asset && (
                  <div className="map-coordinates">
                    {asset.metadata
                      ? `${asset.metadata.width} × ${asset.metadata.height}${asset.metadata.crs ? ` · ${asset.metadata.crs}` : ' · no CRS supplied'}`
                      : 'Raster metadata unavailable'}
                  </div>
                )}
              </figure>
            )}
            {evidenceClasses.length > 0 && (
              <div className="map-legend" aria-label="Evidence class legend">
                <strong>Returned layers</strong>
                {evidenceClasses.map((name) => (
                  <span key={name}>
                    <i style={{ background: evidencePalette[name] ?? evidencePalette.unknown }} />
                    {name}
                  </span>
                ))}
              </div>
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
                    ? 'Class-colored polygons trace candidate mask regions: blue water, green vegetation, amber built-up, red burn-scar, violet change. Toggle off to inspect the source; areas measure predictions, not ground truth.'
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
            <div className="scene-timeline" aria-label="Uploaded evidence sequence">
              <div className="timeline-heading">
                <FileClock />
                <span>Evidence sequence</span>
                <strong>{assets.length} source{assets.length === 1 ? '' : 's'}</strong>
              </div>
              {assets.length ? (
                <div className="timeline-items">
                  {assets.map((source, index) => (
                    <button
                      key={source.id}
                      type="button"
                      aria-pressed={asset?.id === source.id}
                      onClick={() => setSelectedAssetId(source.id)}
                    >
                      <span className="timeline-preview">
                        <Image
                          src={`/api/satquery/assets/${source.id}/preview`}
                          alt=""
                          fill
                          unoptimized
                          sizes="100px"
                        />
                      </span>
                      <span>
                        <small>{source.role.replaceAll('_', ' ')}</small>
                        <strong>{source.original_name}</strong>
                        <em>
                          {source.metadata
                            ? `${source.metadata.width} × ${source.metadata.height}`
                            : 'Metadata unavailable'}
                        </em>
                      </span>
                      <i>{String(index + 1).padStart(2, '0')}</i>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="timeline-empty">
                  <span><FileClock /></span>
                  <div><strong>Sequence waiting for evidence</strong><p>Validated source roles and dimensions will appear here.</p></div>
                </div>
              )}
            </div>
          </section>
          <section
            className={`answer-card glass-panel ${phase}`}
            id="answer"
            aria-live="polite"
          >
            <div className="report-heading">
              <span className="eyebrow">Evidence-led output</span>
              <h1>Intelligence brief</h1>
              <p>Satellite data · human context · inspectable provenance</p>
            </div>
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
            <h2 className="executive-title">
              {result
                ? 'Executive assessment'
                : busy
                  ? analysis
                    ? statusCopy[analysis.status]
                    : 'Inspecting the upload…'
                  : phase === 'failed'
                    ? 'No model answer was produced.'
                    : 'Your evidence-backed answer will appear here.'}
            </h2>
            {result && (measuredArea || coverageEvidence) && (
              <div className="report-measures">
                {measuredArea && (
                  <div>
                    <BarChart3 />
                    <span>Measured candidate area</span>
                    <strong>
                      {Number(measuredArea.geometry.area_m2) >= 1_000_000
                        ? `${(Number(measuredArea.geometry.area_m2) / 1_000_000).toFixed(3)} km²`
                        : `${Number(measuredArea.geometry.area_m2).toLocaleString(undefined, { maximumFractionDigits: 1 })} m²`}
                    </strong>
                    <small>{measuredArea.label} · model-derived boundary</small>
                  </div>
                )}
                {coverageEvidence && (
                  <div>
                    <Activity />
                    <span>Selected-grid coverage</span>
                    <strong>
                      {Number(coverageEvidence.geometry.coverage_percent).toFixed(2)}%
                    </strong>
                    <small>{coverageEvidence.label} · not confidence</small>
                  </div>
                )}
              </div>
            )}
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
            {!result && !busy && phase !== 'failed' && (
              <div className="brief-empty-guide" aria-label="Analysis output stages">
                <div><span>01</span><strong>Validate</strong><small>File contract and source metadata</small></div>
                <div><span>02</span><strong>Route</strong><small>Compatible specialist selection</small></div>
                <div><span>03</span><strong>Report</strong><small>Answer, evidence and provenance</small></div>
              </div>
            )}
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
            {result && Object.keys(result.provenance ?? {}).length > 0 && (
              <section className="provenance-block">
                <h3>
                  <ShieldCheck /> Provenance
                </h3>
                <dl>
                  {Object.entries(result.provenance).map(([key, value]) => (
                    <div key={key}>
                      <dt>{key.replaceAll('_', ' ')}</dt>
                      <dd>
                        {typeof value === 'string' || typeof value === 'number'
                          ? String(value)
                          : JSON.stringify(value)}
                      </dd>
                    </div>
                  ))}
                </dl>
              </section>
            )}
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
