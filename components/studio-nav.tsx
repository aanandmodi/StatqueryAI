import {
  Boxes,
  CircleDot,
  FileClock,
  ScanSearch,
  ShieldCheck,
} from 'lucide-react';
import Link from 'next/link';

const links = [
  { href: '/', label: 'Analyze', icon: ScanSearch },
  { href: '/cases', label: 'Cases', icon: Boxes },
  { href: '/archive', label: 'History', icon: FileClock },
  { href: '/method', label: 'Methods', icon: ShieldCheck },
];

export function StudioNav({ active }: { active: string }) {
  return (
    <aside className="studio-sidebar" aria-label="Primary navigation">
      <Link className="sidebar-brand" href="/" aria-label="SatQuery home">
        <span>SQ</span>
        <strong>SatQuery</strong>
      </Link>
      <nav>
        {links.map(({ href, label, icon: Icon }) => (
          <Link
            className="sidebar-link"
            key={href}
            href={href}
            aria-current={active === href ? 'page' : undefined}
          >
            <Icon aria-hidden="true" />
            <span>{label}</span>
          </Link>
        ))}
      </nav>
      <div className="sidebar-signal" aria-hidden="true">
        <span />
        <CircleDot />
      </div>
    </aside>
  );
}
