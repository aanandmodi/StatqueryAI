'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { StudioNav } from '@/components/studio-nav';
import { jsonRequest } from '@/lib/api-client';

type Case = {
  id: string;
  status: string;
  query: string;
  created_at: string;
  asset_count: number;
  answer_excerpt?: string;
};
export default function Casebook() {
  const [items, setItems] = useState<Case[]>([]);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let active = true;
    jsonRequest<{ items: Case[] }>(
      `/api/satquery/analyses?limit=20&offset=${offset}`,
    )
      .then((data) => {
        if (active) setItems(data.items);
      })
      .catch((e) => {
        if (active) setError(e.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [offset]);
  return (
    <main className="document-shell">
      <StudioNav active="/cases" />
      <header className="document-heading">
        <span className="eyebrow">02 / Casebook</span>
        <h1>A record, not a disappearing chat.</h1>
        <p>
          Your local investigations, source evidence and reports. Stored in the
          backend database—not only in this browser.
        </p>
      </header>
      {error && (
        <div role="alert" className="error-banner">
          {error}
        </div>
      )}
      {loading ? (
        <output>Loading casebook…</output>
      ) : (
        <div className="record-list">
          {items.map((item) => (
            <a href={`/cases/${item.id}`} key={item.id} className="record-card">
              <div>
                <div className="record-meta">
                  <span>{new Date(item.created_at).toLocaleString()}</span>
                  <span>
                    {item.asset_count} source{item.asset_count === 1 ? '' : 's'}
                  </span>
                </div>
                <h2>{item.query}</h2>
                <p>
                  {item.answer_excerpt ||
                    'No completed interpretation. Open the case for status and errors.'}
                </p>
              </div>
              <span className="status-tag">{item.status} ↗</span>
            </a>
          ))}
          {!items.length && !error && (
            <p>
              No investigations on this page. <Link href="/">Start an analysis</Link>.
            </p>
          )}
        </div>
      )}
      <div className="toolbar">
        <button
          className="studio-button"
          disabled={loading || offset === 0}
          onClick={() => {setLoading(true);setError('');setOffset(Math.max(0, offset - 20));}}
        >
          Previous
        </button>
        <span>Page {offset / 20 + 1}</span>
        <button
          className="studio-button"
          disabled={loading || items.length < 20}
          onClick={() => {setLoading(true);setError('');setOffset(offset + 20);}}
        >
          Next
        </button>
      </div>
    </main>
  );
}
