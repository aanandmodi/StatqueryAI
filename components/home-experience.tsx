'use client';

import {
  ArrowDown,
  ArrowUpRight,
  Check,
  ChevronDown,
  Eye,
  Layers3,
  Menu,
  MoveHorizontal,
  Orbit,
  ScanLine,
  ShieldCheck,
  X,
} from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';
import { useEffect, useRef, useState } from 'react';
import { SatQueryLogo } from '@/components/satquery-logo';

const workflows = [
  {
    name: 'One scene',
    tag: '01 / SINGLE IMAGE',
    title: <>Start with a place.<br />Ask a better question.</>,
    description: 'Explore a scene in your own words. Ask about visible land cover, inspect spatial evidence, and keep the original image in view.',
    prompt: 'Where are the water bodies in this scene?',
    first: 'coast.jpg',
    firstAlt: 'Satellite view of a coastline, fields, waterways, and a coastal settlement',
    second: '', secondAlt: '', leftLabel: 'OPTICAL SCENE', rightLabel: '',
    note: 'Preserved coastal demo image · source preview, not a predicted mask.',
    details: ['JPG, PNG or GeoTIFF', 'Text answers + available spatial evidence', 'Source image stays inspectable'],
  },
  {
    name: 'Across time',
    tag: '02 / BI-TEMPORAL',
    title: <>The same place.<br />A different moment.</>,
    description: 'Compare aligned before-and-after images. Inspect candidate changes alongside the two observations, with a report that separates observations from interpretation.',
    prompt: 'What changed between these two observations?',
    first: 'nepal-before.jpg',
    firstAlt: 'Earlier supplied Nepal image showing a river through a mountain valley',
    second: 'nepal-after.jpg',
    secondAlt: 'Later supplied Nepal image of the same valley for visual comparison',
    leftLabel: 'BEFORE', rightLabel: 'AFTER',
    note: 'Supplied Nepal demo pair · acquisition dates not independently verified. Drag to compare source images.',
    details: ['Two co-registered images', 'Change masks + comparative narrative', 'Alignment and metadata checks'],
  },
  {
    name: 'Optical + radar',
    tag: '03 / MULTIMODAL',
    title: <>Two ways of seeing.<br />One investigation.</>,
    description: 'Bring optical imagery and SAR into the same investigation. Examine flood-related evidence with complementary observations instead of relying on one image alone.',
    prompt: 'What flood-related evidence do both sensors show?',
    first: 'india-optical.jpg',
    firstAlt: 'Optical display of the India Sen1Floods11 river scene',
    second: 'india-sar.jpg',
    secondAlt: 'Grayscale SAR display of the matching India Sen1Floods11 river scene',
    leftLabel: 'OPTICAL', rightLabel: 'RADAR / SAR',
    note: 'Sen1Floods11 · India_1018327 · optical and SAR display previews, not model results.',
    details: ['Aligned optical and SAR observations', 'Learned flood-mask specialist', 'Sensor-specific input requirements'],
  },
];

function OrbitMark({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <circle cx="24" cy="24" r="12" stroke="currentColor" strokeWidth="1.6" />
      <ellipse cx="24" cy="24" rx="22" ry="8" transform="rotate(-38 24 24)" stroke="currentColor" strokeWidth="1.6" />
      <circle cx="38" cy="12" r="3.4" fill="currentColor" />
    </svg>
  );
}

export function HomeExperience() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [selected, setSelected] = useState(0);
  const [split, setSplit] = useState(50);
  const root = useRef<HTMLDivElement>(null);
  const tabs = useRef<Array<HTMLButtonElement | null>>([]);
  const workflow = workflows[selected];

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (media.matches || !('IntersectionObserver' in window)) return;
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('home-arrived');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });
    root.current?.querySelectorAll('[data-home-reveal]').forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  function chooseWorkflow(index: number) {
    setSelected(index);
    setSplit(50);
  }

  return (
    <div className="landing" ref={root}>
      <a className="home-skip" href="#home-main">Skip to content</a>
      <header className="home-header">
        <SatQueryLogo className="home-logo" />
        <button className="home-menu-toggle" aria-label={menuOpen ? 'Close navigation' : 'Open navigation'} aria-expanded={menuOpen} aria-controls="home-navigation" onClick={() => setMenuOpen(!menuOpen)}>
          {menuOpen ? <X /> : <Menu />}
        </button>
        <nav id="home-navigation" className={menuOpen ? 'home-nav is-open' : 'home-nav'} aria-label="Main navigation">
          <a href="#explore" onClick={() => setMenuOpen(false)}>Explore</a>
          <a href="#how-it-works" onClick={() => setMenuOpen(false)}>How it works</a>
          <a href="#about" onClick={() => setMenuOpen(false)}>About</a>
          <Link href="/method">Our methods <ArrowUpRight size={13} /></Link>
          <Link className="home-nav-cta" href="/workspace">Open workspace <ArrowUpRight size={16} /></Link>
        </nav>
      </header>

      <main id="home-main">
        <section className="home-hero" aria-labelledby="home-title">
          <div className="home-hero-copy">
            <p className="home-eyebrow"><span className="home-tiny-orbit" /> A CLOSER LOOK AT OUR CHANGING PLANET</p>
            <h1 id="home-title">Every landscape<br />has a <em>story.</em><br />Ask yours.</h1>
            <p className="home-lead">Turn satellite imagery into an investigation. Ask questions, explore changes, and see the evidence behind the answer.</p>
            <div className="home-hero-actions">
              <Link className="home-button home-button-dark" href="/workspace">Start exploring <ArrowUpRight size={19} /></Link>
              <a className="home-text-link" href="#explore">Take a closer look <ArrowDown size={16} /></a>
            </div>
            <div className="home-hero-foot"><span className="home-small-cross">+</span><p>From pixels to perspective.<br /><strong>A remote-sensing research workspace.</strong></p><svg viewBox="0 0 92 34" aria-hidden="true"><path d="M2 17h88M17 2v30M37 8v18M57 11v12M77 14v6" stroke="currentColor" fill="none" /></svg></div>
          </div>
          <div className="home-hero-art">
            <div className="home-art-topline"><span>FIELD OF VIEW — 001</span><span>EARTH / OPTICAL</span></div>
            <div className="home-orbit-ring" aria-hidden="true" />
            <figure className="home-hero-image">
              <Image src="/images/home/coast.jpg" alt="Dark green sea beside coastal dunes, fields and a settlement" width={1001} height={1001} priority sizes="(max-width: 800px) calc(100vw - 28px), 54vw" />
              <div className="home-image-grid" aria-hidden="true" />
              <span className="home-image-corner corner-tl" aria-hidden="true" /><span className="home-image-corner corner-br" aria-hidden="true" />
              <div className="home-image-caption"><span className="home-image-dot" /> COASTAL LANDSCAPE <span>RGB VIEW</span></div>
            </figure>
            <div className="home-question-float"><span className="home-question-icon"><ScanLine size={23} /></span><div><small>EVERY INVESTIGATION STARTS WITH A QUESTION</small><p>“What do you see here?”</p></div><ArrowUpRight size={21} /></div>
            <p className="home-art-foot"><span>Real imagery. Open questions.</span><span>DEMO SOURCE PREVIEW ↗</span></p>
          </div>
        </section>

        <div className="home-context-strip"><span>ONE WORKSPACE. DIFFERENT PERSPECTIVES.</span><div><span><ScanLine /> Read a scene</span><span><Layers3 /> Trace a change</span><span><Orbit /> Connect observations</span></div></div>

        <section className="home-section home-explore" id="explore" aria-labelledby="explore-title" data-home-reveal>
          <div className="home-section-heading"><p className="home-eyebrow">01 — EXPLORE THE EVIDENCE</p><div><h2 id="explore-title">More than a picture.<br /><em>A place to ask questions.</em></h2><p>One scene, two moments, or two sensors.<br />Choose the perspective your question needs.</p></div></div>
          <div className="home-workflow-tabs" role="tablist" aria-label="Explore evidence workflows">
            {workflows.map((item, index) => (
              <button key={item.tag} ref={(node) => { tabs.current[index] = node; }} id={`workflow-tab-${index}`} role="tab" aria-selected={index === selected} aria-controls="workflow-panel" tabIndex={index === selected ? 0 : -1} onClick={() => chooseWorkflow(index)} onKeyDown={(event) => {
                let next = index;
                if (event.key === 'ArrowRight') next = (index + 1) % workflows.length;
                else if (event.key === 'ArrowLeft') next = (index + workflows.length - 1) % workflows.length;
                else if (event.key === 'Home') next = 0;
                else if (event.key === 'End') next = workflows.length - 1;
                else return;
                event.preventDefault(); chooseWorkflow(next); tabs.current[next]?.focus();
              }}><span>0{index + 1}</span>{item.name}<ArrowUpRight size={17} /></button>
            ))}
          </div>
          <div id="workflow-panel" role="tabpanel" aria-labelledby={`workflow-tab-${selected}`} className="home-workflow-panel" tabIndex={0}>
            <div className="home-workflow-visual">
              <div className="home-comparison">
                <Image src={`/images/home/${workflow.first}`} alt={workflow.firstAlt} width={1100} height={1100} loading="lazy" sizes="(max-width: 800px) calc(100vw - 50px), 56vw" />
                {workflow.second && <><Image className="home-comparison-second" src={`/images/home/${workflow.second}`} alt={workflow.secondAlt} width={1100} height={1100} loading="lazy" sizes="(max-width: 800px) calc(100vw - 50px), 56vw" style={{ clipPath: `inset(0 0 0 ${split}%)` }} /><div className="home-split-line" style={{ left: `${split}%` }} aria-hidden="true"><span><MoveHorizontal size={20} /></span></div></>}
                <div className="home-compare-labels"><span>{workflow.leftLabel}</span>{workflow.second && <span>{workflow.rightLabel}</span>}</div>
                {!workflow.second && <div className="home-source-badge"><Eye size={15} /> Source imagery · no generated overlay</div>}
              </div>
              {workflow.second && <label className="home-slider-label">Compare views<input type="range" min="0" max="100" value={split} aria-label={`Compare ${workflow.leftLabel.toLowerCase()} and ${workflow.rightLabel.toLowerCase()} images`} aria-valuetext={`${split}% ${workflow.leftLabel.toLowerCase()} visible`} onChange={(event) => setSplit(Number(event.target.value))} /><span>{split}%</span></label>}
              <p className="home-source-note">{workflow.note}</p>
            </div>
            <div className="home-workflow-copy" key={selected}><p className="home-eyebrow">{workflow.tag}</p><h3>{workflow.title}</h3><p>{workflow.description}</p><ul>{workflow.details.map(detail => <li key={detail}><Check size={15} />{detail}</li>)}</ul><div className="home-example-question"><span>TRY ASKING</span><p>“{workflow.prompt}”</p></div><Link className="home-text-link" href="/workspace">Investigate in the workspace <ArrowUpRight size={18} /></Link></div>
          </div>
        </section>

        <section className="home-process" id="how-it-works" aria-labelledby="process-title">
          <div className="home-process-inner" data-home-reveal>
            <div className="home-section-heading"><p className="home-eyebrow">02 — FROM OBSERVATION TO UNDERSTANDING</p><div><h2 id="process-title">A thoughtful workflow.<br /><em>Not a black box.</em></h2><Link className="home-text-link" href="/method">Go inside the method <ArrowUpRight size={18} /></Link></div></div>
            <div className="home-process-steps">
              <article><span className="home-step-number">01</span><ScanLine /><h3>Bring your evidence.</h3><p>Upload imagery and add the location or sensor context you know. Input checks establish what can be analyzed.</p><span className="home-step-tag">IMAGES + CONTEXT</span></article>
              <article><span className="home-step-number">02</span><Layers3 /><h3>Ask in your own words.</h3><p>The planner routes your question to available vision-language, segmentation, change or fusion specialists.</p><span className="home-step-tag">QUESTION + SPECIALISTS</span></article>
              <article><span className="home-step-number">03</span><ShieldCheck /><h3>Inspect the answer.</h3><p>Review the explanation, available masks, source images and execution trace. Keep limitations with the report.</p><span className="home-step-tag">EVIDENCE + REPORT</span></article>
            </div>
          </div>
        </section>

        <section className="home-section home-about" id="about" aria-labelledby="about-title" data-home-reveal>
          <div className="home-about-visual"><Image src="/images/home/nepal-before.jpg" width={1100} height={1177} loading="lazy" sizes="(max-width: 800px) 94vw, 40vw" alt="Forested hills surrounding a river in the supplied Nepal scene" /><div className="home-about-visual-caption"><OrbitMark /><span>A wider perspective.<br /><strong>A more grounded conversation.</strong></span></div><span className="home-about-coordinate">LAND / WATER / CHANGE</span></div>
          <div className="home-about-copy"><p className="home-eyebrow">03 — ABOUT SATQUERY</p><h2 id="about-title">Built for people<br />who need to<br /><em>look closer.</em></h2><p>Earth-observation images contain a wealth of information. Making sense of them should start with a question—not a maze of tools.</p><p>SatQuery is a student-built research prototype for Smart India Hackathon. We bring specialist models and an inspectable workflow together to support environmental investigation.</p><div className="home-audiences"><span>Forest & land monitoring</span><span>Disaster research</span><span>Visual journalism</span></div><Link className="home-text-link" href="/cases">Explore saved investigations <ArrowUpRight size={18} /></Link></div>
        </section>

        <section className="home-section home-trust" aria-labelledby="trust-title" data-home-reveal>
          <div><p className="home-eyebrow">A NOTE ON TRUST</p><h2 id="trust-title">Useful answers begin<br />with <em>honest limits.</em></h2><p>This is an investigation aid, not a substitute for field verification or expert judgment.</p></div>
          <div className="home-faq">
            <details><summary>What can I upload?<ChevronDown size={18} /></summary><p>Use JPG or PNG for an ordinary image, or GeoTIFF when you have geospatial and band information. Temporal and optical/SAR workflows need compatible aligned inputs. A screenshot cannot supply missing spectral bands or reliable ground-area measurements.</p></details>
            <details><summary>Are the overlays and answers always correct?<ChevronDown size={18} /></summary><p>No. Masks and generated descriptions can miss features or misinterpret a scene. Model quality varies by task and geography. The workspace preserves warnings and distinguishes uncalibrated scores from validated confidence.</p></details>
            <details><summary>Can one image explain why a change happened?<ChevronDown size={18} /></summary><p>Not on its own. Change analysis needs comparable observations. Causes, precise acquisition-time weather and historical events need independent evidence; SatQuery should not invent them from pixels.</p></details>
            <details><summary>What do I need to run an analysis?<ChevronDown size={18} /></summary><p>The local backend and an active configured model service. The current demo uses an attended Kaggle GPU session through a protected ngrok tunnel. This home page can be explored without a running model.</p></details>
          </div>
        </section>

        <section className="home-final-cta"><div className="home-cta-orbits" aria-hidden="true"><OrbitMark /></div><p className="home-eyebrow">YOUR NEXT QUESTION STARTS HERE</p><h2>Look closer.<br /><em>See the bigger picture.</em></h2><Link className="home-button home-button-citrus" href="/workspace">Open your workspace <ArrowUpRight size={20} /></Link><p className="home-cta-note">Bring an image. Follow the evidence.</p></section>
      </main>
      <footer className="home-footer"><SatQueryLogo className="home-logo" /><p>Earth observation. Human curiosity.</p><div><Link href="/workspace">Workspace <ArrowUpRight size={13} /></Link><Link href="/method">Methods <ArrowUpRight size={13} /></Link><a href="https://github.com/aanandmodi/StatqueryAI" target="_blank" rel="noopener noreferrer">GitHub <ArrowUpRight size={13} /></a></div><span>SIH · RESEARCH PROTOTYPE</span></footer>
    </div>
  );
}
