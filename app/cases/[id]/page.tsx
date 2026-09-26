'use client';

import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Download,
  Eye,
  EyeOff,
  FileJson,
  Gauge,
  Layers3,
  Route,
  ShieldCheck,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { use, useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { StructuredAnswer } from '@/components/structured-answer';
import { StudioNav } from '@/components/studio-nav';
import { jsonRequest } from '@/lib/api-client';

type Evidence = {
  id: string;
  type: string;
  label: string;
  asset_id: string;
  score?: number;
  coordinate_space?: string;
  geometry?: Record<string, unknown>;
};

type TraceEvent = {
  step_id?: string;
  task?: string;
  tool?: string;
  model_version?: string;
  status?: string;
  policy_reason?: string;
  duration_ms?: number;
};

type AnalysisRecord = {
  id: string;
  status: string;
  created_at?: string;
  updated_at?: string;
  error_message?: string;
  request: { query: string; asset_ids: string[]; task?: string };
  result?: {
    answer: string;
    report_url?: string;
    warnings: string[];
    facts?: Record<string, unknown>[];
    sections?: { title: string; source: string; paragraphs: string[] }[];
    provenance: Record<string, unknown>;
    trace: TraceEvent[];
    confidence?: {
      score: number;
      level: string;
      calibration_version: string;
      meaning: string;
      correctness_probability?: number | null;
      factors?: Record<string, number>;
    };
    evidence?: Evidence[];
  };
};

const factorPalette = ['#18a99a', '#e3a52b', '#4b82b1', '#d96d62', '#7d67b3', '#45a76d'];

function readable(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'Unavailable';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  return JSON.stringify(value);
}

function titleCase(value: string): string {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function CaseDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [record, setRecord] = useState<AnalysisRecord | null>(null);
  const [error, setError] = useState('');
  const [showMasks, setShowMasks] = useState(true);
  const [maskOpacity, setMaskOpacity] = useState(0.4);
  const [activeAsset, setActiveAsset] = useState('');

  useEffect(() => {
    let active = true;
    jsonRequest<AnalysisRecord>(`/api/satquery/analyses/${encodeURIComponent(id)}`)
      .then((response) => {
        if (!active) return;
        setRecord(response);
        setActiveAsset(response.request.asset_ids[0] ?? '');
      })
      .catch((requestError) => {
        if (active) setError(requestError.message);
      });
    return () => {
      active = false;
    };
  }, [id]);

  const activeEvidence = useMemo(
    () => record?.result?.evidence?.filter((item) => item.asset_id === activeAsset) ?? [],
    [activeAsset, record],
  );
  const maskEvidence = activeEvidence.filter((item) => item.type === 'mask');
  const coverageData = useMemo(
    () =>
      (record?.result?.evidence ?? [])
        .map((item) => ({
          name: item.label,
          value:
            typeof item.geometry?.coverage_percent === 'number'
              ? item.geometry.coverage_percent
              : null,
        }))
        .filter((item): item is { name: string; value: number } => item.value !== null),
    [record],
  );
  const factorData = useMemo(
    () =>
      Object.entries(record?.result?.confidence?.factors ?? {}).map(([name, value], index) => ({
        name: titleCase(name),
        value: Number(value) * 100,
        fill: factorPalette[index % factorPalette.length],
      })),
    [record],
  );
  const durationData = useMemo(
    () =>
      (record?.result?.trace ?? [])
        .filter((event) => typeof event.duration_ms === 'number')
        .map((event) => ({
          name: event.task || event.tool || event.step_id || 'Step',
          value: event.duration_ms as number,
        })),
    [record],
  );

  function downloadJson() {
    if (!record) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }),
    );
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${id}.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <main className="document-shell case-detail-shell">
      <StudioNav active="/cases" />
      <div className="document-content">
        {error && <div role="alert" className="error-banner">{error}</div>}
        {!record ? (
          <div className="case-loading">
            <span className="eyebrow">Evidence dossier</span>
            <h1>{error ? 'Case unavailable.' : 'Opening stored investigation…'}</h1>
          </div>
        ) : (
          <>
            <header className="case-hero">
              <div>
                <Link className="case-back" href="/cases"><ArrowLeft /> Casebook</Link>
                <span className="eyebrow">Stored investigation · {record.status}</span>
                <h1>{record.request.query}</h1>
                <div className="case-id-row">
                  <code>{record.id}</code>
                  {record.created_at && <span>{new Date(record.created_at).toLocaleString()}</span>}
                </div>
              </div>
              <div className="case-actions">
                <button className="studio-button" onClick={downloadJson}><FileJson /> Evidence JSON</button>
                {record.result?.report_url && (
                  <a className="studio-button primary" href={`/api/satquery/analyses/${id}/report`}>
                    <Download /> Report PDF
                  </a>
                )}
              </div>
            </header>

            {record.error_message && <div className="error-banner">{record.error_message}</div>}

            <section className="case-status-strip" aria-label="Analysis summary">
              <div><CheckCircle2 /><span>Status</span><strong>{record.status}</strong></div>
              <div><Layers3 /><span>Sources</span><strong>{record.request.asset_ids.length}</strong></div>
              <div><Route /><span>Requested route</span><strong>{record.request.task ? titleCase(record.request.task) : 'Automatic'}</strong></div>
              <div>
                <Gauge />
                <span>Score semantics</span>
                <strong>{record.result?.confidence?.calibration_version || 'No score returned'}</strong>
              </div>
            </section>

            <div className="case-dossier-grid">
              <section className="case-visual-panel">
                <div className="case-section-heading">
                  <div><span className="eyebrow">Source-locked evidence</span><h2>Scene inspection</h2></div>
                  {maskEvidence.length > 0 && (
                    <button className="icon-action" onClick={() => setShowMasks((shown) => !shown)}>
                      {showMasks ? <Eye /> : <EyeOff />}{showMasks ? 'Masks on' : 'Masks off'}
                    </button>
                  )}
                </div>

                {record.request.asset_ids.length > 1 && (
                  <div className="source-tabs" role="tablist" aria-label="Source scene">
                    {record.request.asset_ids.map((asset, index) => (
                      <button
                        role="tab"
                        aria-selected={asset === activeAsset}
                        key={asset}
                        onClick={() => setActiveAsset(asset)}
                      >
                        Source {index + 1}
                      </button>
                    ))}
                  </div>
                )}

                <figure className="case-scene-frame">
                  <div className="case-mask-canvas">
                    <Image
                      width={1280}
                      height={1280}
                      unoptimized
                      src={`/api/satquery/assets/${activeAsset}/preview`}
                      alt="Selected original source"
                    />
                    {showMasks && maskEvidence.map((item) => (
                      <Image
                        key={item.id}
                        fill
                        unoptimized
                        style={{ opacity: maskOpacity, imageRendering: 'pixelated', pointerEvents: 'none' }}
                        src={`/api/satquery/analyses/${id}/masks/${item.id}?colored=true`}
                        alt={`${item.label} candidate overlay`}
                      />
                    ))}
                  </div>
                  <figcaption>
                    <span>Original pixels with {showMasks ? maskEvidence.length : 0} visible candidate overlay{maskEvidence.length === 1 ? '' : 's'}</span>
                    <code>{activeAsset}</code>
                  </figcaption>
                </figure>

                {maskEvidence.length > 0 && (
                  <div className="mask-control-dock">
                    <label>
                      Overlay opacity
                      <input
                        aria-label="Saved case mask opacity"
                        type="range"
                        min="0"
                        max="0.8"
                        step="0.05"
                        value={maskOpacity}
                        onChange={(event) => setMaskOpacity(Number(event.target.value))}
                      />
                      <output>{Math.round(maskOpacity * 100)}%</output>
                    </label>
                    <p>Candidate boundaries, not verified truth. Disable overlays to inspect source pixels.</p>
                  </div>
                )}

                <div className="evidence-downloads">
                  {maskEvidence.map((item) => (
                    <a key={item.id} href={`/api/satquery/analyses/${id}/masks/${item.id}`}>
                      <Download /> {item.label} mask
                    </a>
                  ))}
                  {record.result && activeAsset && (
                    <a href={`/api/satquery/analyses/${id}/overlay?asset_id=${activeAsset}`}>
                      <Download /> Marked source
                    </a>
                  )}
                </div>
              </section>

              <article className="case-brief-panel">
                <div className="case-section-heading">
                  <div><span className="eyebrow">Evidence-led output</span><h2>Intelligence brief</h2></div>
                  {record.result?.confidence && (
                    <span className={`confidence-chip level-${record.result.confidence.level}`}>
                      {record.result.confidence.correctness_probability == null
                        ? 'Correctness probability unavailable'
                        : `${(record.result.confidence.correctness_probability * 100).toFixed(0)}% calibrated`}
                    </span>
                  )}
                </div>
                {record.result ? (
                  <>
                    <StructuredAnswer content={record.result.answer} className="narrative-output" />
                    {record.result.sections?.map((section, index) => (
                      <section className="report-section" key={`${section.title}-${index}`}>
                        <div><span>{String(index + 1).padStart(2, '0')}</span><h3>{section.title}</h3></div>
                        <small>{section.source}</small>
                        {section.paragraphs.map((paragraph, paragraphIndex) => (
                          <StructuredAnswer key={paragraphIndex} content={paragraph} />
                        ))}
                      </section>
                    ))}
                  </>
                ) : (
                  <div className="document-empty"><strong>No completed interpretation.</strong></div>
                )}
              </article>
            </div>

            {record.result && (
              <>
                <section className="case-analytics">
                  <div className="case-section-heading wide-heading">
                    <div><span className="eyebrow">Returned quantitative fields</span><h2>Evidence signals</h2></div>
                    <p>Charts are rendered only from numeric values stored in this analysis.</p>
                  </div>
                  <div className="analytics-grid">
                    {coverageData.length > 0 && (
                      <article className="chart-card">
                        <h3>Candidate coverage</h3><p>Percent of valid analysed pixels</p>
                        <ResponsiveContainer width="100%" height={230}>
                          <BarChart data={coverageData} layout="vertical" margin={{ left: 10, right: 16 }}>
                            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                            <XAxis type="number" domain={[0, 100]} unit="%" />
                            <YAxis type="category" dataKey="name" width={105} tick={{ fontSize: 10 }} />
                            <Tooltip formatter={(value) => [`${Number(value).toFixed(2)}%`, 'Coverage']} />
                            <Bar dataKey="value" fill="#19aa9b" radius={[0, 5, 5, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </article>
                    )}
                    {factorData.length > 0 && (
                      <article className="chart-card">
                        <h3>Quality diagnostics</h3><p>Uncalibrated integration factors—not correctness probabilities</p>
                        <ResponsiveContainer width="100%" height={230}>
                          <PieChart>
                            <Pie data={factorData} dataKey="value" nameKey="name" innerRadius={52} outerRadius={82} paddingAngle={3} />
                            <Tooltip formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Factor']} />
                          </PieChart>
                        </ResponsiveContainer>
                        <div className="chart-legend">{factorData.map((item) => <span key={item.name}><i style={{ background: item.fill }} />{item.name}: {item.value.toFixed(0)}%</span>)}</div>
                      </article>
                    )}
                    {durationData.length > 0 && (
                      <article className="chart-card">
                        <h3>Execution timing</h3><p>Recorded duration by trace task</p>
                        <ResponsiveContainer width="100%" height={230}>
                          <BarChart data={durationData} margin={{ left: 4, right: 10, bottom: 30 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="name" angle={-25} textAnchor="end" height={55} tick={{ fontSize: 9 }} />
                            <YAxis unit="ms" tick={{ fontSize: 9 }} />
                            <Tooltip formatter={(value) => [`${Number(value).toLocaleString()} ms`, 'Duration']} />
                            <Bar dataKey="value" fill="#497fa3" radius={[5, 5, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </article>
                    )}
                    {!coverageData.length && !factorData.length && !durationData.length && (
                      <div className="document-empty"><strong>No quantitative evidence was returned.</strong><p>No chart has been fabricated.</p></div>
                    )}
                  </div>
                </section>

                <div className="case-support-grid">
                  <section className="case-facts-panel">
                    <div className="case-section-heading"><div><span className="eyebrow">Structured output</span><h2>Observed facts</h2></div></div>
                    {record.result.facts?.length ? (
                      <div className="fact-grid">
                        {record.result.facts.map((fact, index) => (
                          <article key={index}>
                            <span>{String(index + 1).padStart(2, '0')}</span>
                            {Object.entries(fact).map(([key, value]) => (
                              <div key={key}><small>{titleCase(key)}</small><strong>{readable(value)}</strong></div>
                            ))}
                          </article>
                        ))}
                      </div>
                    ) : <p className="quiet">No structured facts were returned.</p>}
                  </section>

                  <aside className="case-warning-panel">
                    <AlertTriangle />
                    <span className="eyebrow">Review boundaries</span>
                    <h2>Limitations & warnings</h2>
                    {record.result.warnings.length ? (
                      <ul>{record.result.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>
                    ) : <p>No warnings were returned.</p>}
                    {record.result.confidence && <p className="confidence-meaning"><strong>Score meaning:</strong> {record.result.confidence.meaning}</p>}
                  </aside>
                </div>

                <section className="trace-ledger">
                  <div className="case-section-heading wide-heading"><div><span className="eyebrow">Observable execution</span><h2>Trace ledger</h2></div><span>{record.result.trace.length} events</span></div>
                  {record.result.trace.length ? (
                    <ol>
                      {record.result.trace.map((event, index) => (
                        <li key={`${event.step_id}-${index}`}>
                          <span className="trace-number">{String(index + 1).padStart(2, '0')}</span>
                          <div><strong>{titleCase(event.task || event.step_id || 'Trace event')}</strong><small>{event.tool || 'Tool unavailable'}{event.model_version ? ` · ${event.model_version}` : ''}</small></div>
                          <p>{event.policy_reason || 'No policy note returned.'}</p>
                          <span className={`trace-state state-${event.status}`}>{event.status || 'unknown'}</span>
                          <time>{typeof event.duration_ms === 'number' ? `${event.duration_ms.toLocaleString()} ms` : 'No duration'}</time>
                        </li>
                      ))}
                    </ol>
                  ) : <p className="quiet">No trace events were stored.</p>}
                </section>

                <section className="provenance-panel">
                  <div className="case-section-heading wide-heading"><div><span className="eyebrow">Audit record</span><h2>Provenance</h2></div><ShieldCheck /></div>
                  <div className="provenance-grid">
                    {Object.entries(record.result.provenance).map(([key, value]) => (
                      <div key={key}><span>{titleCase(key)}</span><code>{readable(value)}</code></div>
                    ))}
                    {!Object.keys(record.result.provenance).length && <p className="quiet">No provenance fields were returned.</p>}
                  </div>
                </section>
              </>
            )}
          </>
        )}
      </div>
    </main>
  );
}
