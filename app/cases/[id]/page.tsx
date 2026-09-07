'use client';
import { use, useEffect, useState } from 'react';
import Image from 'next/image';
import { StudioNav } from '@/components/studio-nav';
import { StructuredAnswer } from '@/components/structured-answer';
import { jsonRequest } from '@/lib/api-client';

type Record = {
  id: string;
  status: string;
  error_message?: string;
  request: { query: string; asset_ids: string[] };
  result?: {
    answer: string;
    report_url?: string;
    warnings: string[];
    sections?: { title: string; source: string; paragraphs: string[] }[];
    provenance: unknown;
    trace: unknown;
    evidence?: { id: string; type: string; label: string; asset_id: string }[];
  };
};
export default function CaseDetail({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [record, setRecord] = useState<Record | null>(null);
  const [error, setError] = useState('');
  const [showMasks, setShowMasks] = useState(true);
  const [maskOpacity, setMaskOpacity] = useState(0.35);
  useEffect(() => {
    let active = true;
    jsonRequest<Record>(`/api/satquery/analyses/${encodeURIComponent(id)}`)
      .then((r) => {
        if (active) setRecord(r);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [id]);
  function downloadJson() {
    if (!record) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }),
    );
    const a = document.createElement('a');
    a.href = url;
    a.download = `${id}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <main className="document-shell">
      <StudioNav active="/cases" />
      {error && (
        <div role="alert" className="error-banner">
          {error}
        </div>
      )}
      {!record ? (
        <p>{error ? 'Case unavailable.' : 'Loading case…'}</p>
      ) : (
        <>
          <header className="document-heading">
            <span className="eyebrow">Case / {record.status}</span>
            <h1>{record.request.query}</h1>
            <p>{record.id}</p>
          </header>
          <div className="toolbar">
            <button className="studio-button" onClick={downloadJson}>
              Export evidence record · JSON
            </button>
            {record.result?.report_url && (
              <a
                className="studio-button"
                href={`/api/satquery/analyses/${id}/report`}
              >
                Download report · PDF
              </a>
            )}
          </div>
          {record.error_message && (
            <div className="error-banner">{record.error_message}</div>
          )}
          {record.result?.evidence?.some((e) => e.type === 'mask') && (
            <div className="case-mask-controls">
              <label>
                <input
                  type="checkbox"
                  checked={showMasks}
                  onChange={(e) => setShowMasks(e.target.checked)}
                />{' '}
                Show candidate masks
              </label>
              <label>
                Mask opacity{' '}
                <input
                  aria-label="Saved case mask opacity"
                  type="range"
                  min="0"
                  max="0.8"
                  step="0.05"
                  value={maskOpacity}
                  onChange={(e) => setMaskOpacity(Number(e.target.value))}
                />
              </label>
              <p>
                Candidate boundaries, not verified truth. Switch off to inspect
                the original pixels.
              </p>
            </div>
          )}
          <div
            className="case-images"
            data-count={record.request.asset_ids.length}
          >
            {record.request.asset_ids.map((asset, index) => (
              <figure key={asset}>
                <figcaption>
                  Source {index + 1} · {asset}
                </figcaption>
                <div className="case-mask-canvas">
                  <Image
                    width={1280}
                    height={1280}
                    style={{ width: '100%', height: 'auto', display: 'block' }}
                    unoptimized
                    src={`/api/satquery/assets/${asset}/preview`}
                    alt={`Original source ${index + 1}`}
                  />
                  {showMasks &&
                    record.result?.evidence
                      ?.filter((e) => e.type === 'mask' && e.asset_id === asset)
                      .map((e) => (
                        <Image
                          key={e.id}
                          fill
                          unoptimized
                          style={{
                            opacity: maskOpacity,
                            imageRendering: 'pixelated',
                            pointerEvents: 'none',
                          }}
                          src={`/api/satquery/analyses/${id}/masks/${e.id}?colored=true`}
                          alt={`${e.label} — candidate overlay`}
                        />
                      ))}
                </div>
                {record.result?.evidence
                  ?.filter((e) => e.type === 'mask' && e.asset_id === asset)
                  .map((e) => (
                    <a
                      className="case-mask-download"
                      key={e.id}
                      href={`/api/satquery/analyses/${id}/masks/${e.id}`}
                    >
                      Binary mask · {e.label} ↗
                    </a>
                  ))}
                {record.result && (
                  <a
                    href={`/api/satquery/analyses/${id}/overlay?asset_id=${asset}`}
                  >
                    Download this source with candidate overlays ↗
                  </a>
                )}
              </figure>
            ))}
          </div>
          {record.result && (
            <article className="document-panel case-report">
              <h2>Visual interpretation</h2>
              <StructuredAnswer content={record.result.answer} className="narrative-output" />
              {record.result.sections?.map((section, index) => (
                <section className="region-report-section" key={index}>
                  <h3>{section.title}</h3>
                  <small>{section.source}</small>
                  {section.paragraphs.map((p, i) => (
                    <StructuredAnswer key={i} content={p} />
                  ))}
                </section>
              ))}
              <details className="warning-list">
                <summary>Warnings ({record.result.warnings.length})</summary>
                <ul>
                  {record.result.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </details>
              <details className="structured-facts">
                <summary>Full provenance & execution trace</summary>
                <pre>
                  {JSON.stringify(
                    {
                      provenance: record.result.provenance,
                      trace: record.result.trace,
                    },
                    null,
                    2,
                  )}
                </pre>
              </details>
            </article>
          )}
        </>
      )}
    </main>
  );
}
