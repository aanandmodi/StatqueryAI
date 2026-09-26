import Link from 'next/link';

export function SatQueryLogo({
  className = '',
  tagline,
}: {
  className?: string;
  tagline?: string;
}) {
  return (
    <Link className={`satquery-logo ${className}`.trim()} href="/" aria-label="SatQuery home">
      <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
        <circle cx="24" cy="24" r="12" stroke="currentColor" strokeWidth="1.6" />
        <ellipse cx="24" cy="24" rx="22" ry="8" transform="rotate(-38 24 24)" stroke="currentColor" strokeWidth="1.6" />
        <circle cx="38" cy="12" r="3.4" fill="currentColor" />
      </svg>
      <span className="satquery-word">
        satquery<span className="satquery-dot">.</span>
        {tagline && <small>{tagline}</small>}
      </span>
    </Link>
  );
}
