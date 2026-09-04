export function StudioNav({ active }: { active: string }) {
  return (
    <nav className="page-nav" aria-label="Main navigation">
      <Link href="/">SATQUERY.</Link>
      {[
        ['/', 'Investigation'],
        ['/cases', 'Casebook'],
        ['/archive', 'Historical evidence'],
        ['/method', 'Methods & limits'],
      ].map(([href, label]) => (
        <Link
          key={href}
          href={href}
          aria-current={active === href ? 'page' : undefined}
        >
          {label}
        </Link>
      ))}
    </nav>
  );
}
import Link from 'next/link';
