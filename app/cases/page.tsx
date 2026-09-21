'use client';

import { ArrowUpRight, Clock3, Files, Search } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';

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
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('all');

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

  const statuses = useMemo(
    () => Array.from(new Set(items.map((item) => item.status))).sort(),
    [items],
  );
  const visibleItems = useMemo(() => {
    const term = search.trim().toLocaleLowerCase();
    return items.filter(
      (item) =>
        (status === 'all' || item.status === status) &&
        (!term ||
          item.query.toLocaleLowerCase().includes(term) ||
          item.answer_excerpt?.toLocaleLowerCase().includes(term) ||
          item.id.toLocaleLowerCase().includes(term)),
    );
  }, [items, search, status]);

  return (
    <main className="document-shell casebook-shell">
      <StudioNav active="/cases" />
      <div className="document-content">
        <header className="document-heading casebook-heading">
          <div>
            <span className="eyebrow">02 / Investigation casebook</span>
            <h1>Every answer leaves an evidence trail.</h1>
            <p>
              Search the investigations stored by the local controller. Every
              card below represents a real submitted analysis record.
            </p>
          </div>
          <div className="casebook-count" aria-label="Records on this page">
            <strong>{items.length}</strong>
            <span>records loaded</span>
          </div>
        </header>

        <section className="casebook-toolbar" aria-label="Filter investigations">
          <label className="case-search">
            <Search aria-hidden="true" />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search query, answer or analysis ID"
              aria-label="Search casebook"
            />
          </label>
          <label className="case-status-filter">
            <span>Status</span>
            <select value={status} onChange={(event) => setStatus(event.target.value)}>
              <option value="all">All on this page</option>
              {statuses.map((itemStatus) => (
                <option value={itemStatus} key={itemStatus}>
                  {itemStatus}
                </option>
              ))}
            </select>
          </label>
        </section>

        {error && (
          <div role="alert" className="error-banner">
            {error}
          </div>
        )}
        {loading ? (
          <output className="casebook-loading">Loading stored records…</output>
        ) : (
          <div className="record-list">
            {visibleItems.map((item, index) => (
              <Link href={`/cases/${item.id}`} key={item.id} className="record-card">
                <span className="record-index">{String(index + 1).padStart(2, '0')}</span>
                <div className="record-card-body">
                  <div className="record-meta">
                    <span><Clock3 />{new Date(item.created_at).toLocaleString()}</span>
                    <span><Files />{item.asset_count} source{item.asset_count === 1 ? '' : 's'}</span>
                  </div>
                  <h2>{item.query}</h2>
                  <p>
                    {item.answer_excerpt ||
                      'No completed interpretation is stored for this analysis.'}
                  </p>
                  <code>{item.id}</code>
                </div>
                <div className="record-card-action">
                  <span className={`status-tag status-${item.status}`}>{item.status}</span>
                  <ArrowUpRight aria-hidden="true" />
                </div>
              </Link>
            ))}
            {!visibleItems.length && !error && (
              <div className="document-empty">
                <strong>No matching investigation.</strong>
                <p>Clear the filters or create a new analysis.</p>
                <Link href="/">Open the investigation studio</Link>
              </div>
            )}
          </div>
        )}

        <div className="toolbar casebook-pagination">
          <button
            className="studio-button"
            disabled={loading || offset === 0}
            onClick={() => {
              setLoading(true);
              setError('');
              setOffset(Math.max(0, offset - 20));
            }}
          >
            Previous
          </button>
          <span>Page {offset / 20 + 1}</span>
          <button
            className="studio-button"
            disabled={loading || items.length < 20}
            onClick={() => {
              setLoading(true);
              setError('');
              setOffset(offset + 20);
            }}
          >
            Next
          </button>
        </div>
      </div>
    </main>
  );
}
