'use client';

import React from 'react';

interface StructuredAnswerProps {
  content: string;
  className?: string;
}

// Helper to parse inline markdown (bold, italic, code, badges) into React nodes
function parseInline(text: string): React.ReactNode[] {
  if (!text) return [];

  // Tokens:
  // 1. `inline code`
  // 2. **bold** or __bold__
  // 3. *italic* or _italic_
  // 4. [Status Badge] e.g. [Verified], [Detected], [High], [Low], [Moderate], [Confirmed]
  const regex = /(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_|\[(?:Verified|Confirmed|Passed|Detected|High|Low|Moderate|Warning|Critical|Pass|Fail|Present|Absent)\])/g;

  const parts = text.split(regex);
  return parts.map((part, index) => {
    if (!part) return null;

    // Inline code: `code`
    if (part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
      const code = part.slice(1, -1);
      return (
        <code
          key={index}
          className="structured-code-pill"
        >
          {code}
        </code>
      );
    }

    // Bold: **text** or __text__
    if (
      (part.startsWith('**') && part.endsWith('**') && part.length >= 4) ||
      (part.startsWith('__') && part.endsWith('__') && part.length >= 4)
    ) {
      const inner = part.slice(2, -2);
      return (
        <strong key={index} className="structured-bold">
          {inner}
        </strong>
      );
    }

    // Italic: *text* or _text_
    if (
      (part.startsWith('*') && part.endsWith('*') && part.length >= 2 && !part.startsWith('**')) ||
      (part.startsWith('_') && part.endsWith('_') && part.length >= 2 && !part.startsWith('__'))
    ) {
      const inner = part.slice(1, -1);
      return (
        <em key={index} className="structured-italic">
          {inner}
        </em>
      );
    }

    // Badges: [Status]
    if (part.startsWith('[') && part.endsWith(']')) {
      const label = part.slice(1, -1);
      const lower = label.toLowerCase();
      let badgeClass = 'badge-neutral';
      if (['verified', 'confirmed', 'passed', 'pass', 'present'].includes(lower)) {
        badgeClass = 'badge-success';
      } else if (['high', 'detected'].includes(lower)) {
        badgeClass = 'badge-info';
      } else if (['moderate', 'warning'].includes(lower)) {
        badgeClass = 'badge-warning';
      } else if (['low', 'critical', 'fail', 'absent'].includes(lower)) {
        badgeClass = 'badge-danger';
      }
      return (
        <span key={index} className={`structured-badge ${badgeClass}`}>
          {label}
        </span>
      );
    }

    // Clean remaining loose asterisks if any
    const sanitized = part.replace(/\*/g, '');
    return <React.Fragment key={index}>{sanitized}</React.Fragment>;
  });
}

// Block token types
type BlockToken =
  | { type: 'heading'; level: number; text: string }
  | { type: 'table'; headers: string[]; rows: string[][] }
  | { type: 'list'; ordered: boolean; items: string[] }
  | { type: 'blockquote'; text: string }
  | { type: 'key_value'; key: string; value: string }
  | { type: 'paragraph'; text: string };

function parseBlocks(raw: string): BlockToken[] {
  const lines = raw.split(/\r?\n/);
  const blocks: BlockToken[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i].trim();

    // Skip blank lines
    if (!line) {
      i++;
      continue;
    }

    // 1. Markdown Table detection: line starts and ends with '|' or has '|' separated columns
    if (line.includes('|') && (line.startsWith('|') || line.split('|').length >= 3)) {
      const tableLines: string[] = [];
      while (
        i < lines.length &&
        lines[i].trim().includes('|') &&
        (lines[i].trim().startsWith('|') || lines[i].trim().split('|').length >= 3)
      ) {
        tableLines.push(lines[i].trim());
        i++;
      }

      if (tableLines.length >= 2) {
        const splitRow = (rowStr: string) => {
          let cleaned = rowStr.trim();
          if (cleaned.startsWith('|')) cleaned = cleaned.slice(1);
          if (cleaned.endsWith('|')) cleaned = cleaned.slice(0, -1);
          return cleaned.split('|').map((c) => c.trim());
        };

        const headers = splitRow(tableLines[0]);
        // Filter out separator row (e.g. |---|---|)
        const rows: string[][] = [];
        for (let r = 1; r < tableLines.length; r++) {
          const rowText = tableLines[r];
          if (/^\|?(?:\s*:?-+:?\s*\|?)+$/.test(rowText.trim())) {
            continue;
          }
          rows.push(splitRow(rowText));
        }

        if (headers.length > 0 && rows.length > 0) {
          blocks.push({ type: 'table', headers, rows });
          continue;
        }
      }
    }

    // 2. Headings: #, ##, ###, ####
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      blocks.push({
        type: 'heading',
        level: headingMatch[1].length,
        text: headingMatch[2].trim(),
      });
      i++;
      continue;
    }

    // Pseudo heading: **Heading Title:** or **Heading Title** on a single line
    const pseudoHeadingMatch = line.match(/^\*\*([^*]+)\*\*:\s*$/);
    if (pseudoHeadingMatch) {
      blocks.push({
        type: 'heading',
        level: 3,
        text: pseudoHeadingMatch[1].trim(),
      });
      i++;
      continue;
    }

    // 3. Lists: Bullet points (*, -, •) or Numbered (1., 2.)
    const bulletMatch = line.match(/^[-*•+]\s+(.+)$/);
    const numberedMatch = line.match(/^(\d+)\.\s+(.+)$/);

    if (bulletMatch || numberedMatch) {
      const isOrdered = !bulletMatch;
      const items: string[] = [];
      while (i < lines.length) {
        const curr = lines[i].trim();
        const bM = curr.match(/^[-*•+]\s+(.+)$/);
        const nM = curr.match(/^(\d+)\.\s+(.+)$/);

        if (isOrdered && nM) {
          items.push(nM[2].trim());
          i++;
        } else if (!isOrdered && bM) {
          items.push(bM[1].trim());
          i++;
        } else if (curr.startsWith('  ') && items.length > 0) {
          // Continuation of previous item
          items[items.length - 1] += ' ' + curr.trim();
          i++;
        } else {
          break;
        }
      }
      blocks.push({ type: 'list', ordered: isOrdered, items });
      continue;
    }

    // 4. Blockquotes: > quote
    if (line.startsWith('>')) {
      const quotes: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        quotes.push(lines[i].trim().replace(/^>\s*/, ''));
        i++;
      }
      blocks.push({ type: 'blockquote', text: quotes.join(' ') });
      continue;
    }

    // 5. Key-Value pairs: e.g. **Direct answer:** Yes, water is visible... or Comparison basis: ...
    const kvMatch = line.match(/^(?:(?:\*\*([^*:]+)\*\*)|([A-Z][A-Za-z0-9\s/_-]{1,30})):\s*(.+)$/);
    if (kvMatch && !kvMatch[3].startsWith('|')) {
      const key = (kvMatch[1] || kvMatch[2]).trim();
      blocks.push({
        type: 'key_value',
        key,
        value: kvMatch[3].trim(),
      });
      i++;
      continue;
    }

    // 6. Default: Paragraph
    const paraLines: string[] = [line];
    i++;
    while (i < lines.length) {
      const next = lines[i].trim();
      if (!next) break;
      if (
        next.includes('|') ||
        next.startsWith('#') ||
        next.match(/^[-*•+]\s+/) ||
        next.match(/^\d+\.\s+/) ||
        next.startsWith('>') ||
        next.match(/^(?:(?:\*\*([^*:]+)\*\*)|([A-Z][A-Za-z0-9\s/_-]{1,30})):\s*/)
      ) {
        break;
      }
      paraLines.push(next);
      i++;
    }
    blocks.push({ type: 'paragraph', text: paraLines.join(' ') });
  }

  return blocks;
}

export function StructuredAnswer({ content, className = '' }: StructuredAnswerProps) {
  if (!content || !content.trim()) {
    return null;
  }

  const blocks = parseBlocks(content);

  return (
    <div className={`structured-answer ${className}`}>
      {blocks.map((block, index) => {
        switch (block.type) {
          case 'heading': {
            if (block.level <= 2) {
              return (
                <h3 key={index} className="structured-heading level-2">
                  <span className="heading-bar" />
                  {parseInline(block.text)}
                </h3>
              );
            }
            if (block.level === 3) {
              return (
                <h4 key={index} className="structured-heading level-3">
                  <span className="heading-dot" />
                  {parseInline(block.text)}
                </h4>
              );
            }
            return (
              <h5 key={index} className="structured-heading level-4">
                {parseInline(block.text)}
              </h5>
            );
          }

          case 'table': {
            return (
              <div key={index} className="structured-table-wrapper">
                <table className="structured-table">
                  <thead>
                    <tr>
                      {block.headers.map((head, hIdx) => (
                        <th key={hIdx}>{parseInline(head)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, rIdx) => (
                      <tr key={rIdx}>
                        {row.map((cell, cIdx) => (
                          <td key={cIdx}>{parseInline(cell)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          }

          case 'list': {
            if (block.ordered) {
              return (
                <ol key={index} className="structured-ordered-list">
                  {block.items.map((item, itemIdx) => (
                    <li key={itemIdx} className="structured-ordered-item">
                      <span className="ordered-badge">{itemIdx + 1}</span>
                      <div className="item-content">{parseInline(item)}</div>
                    </li>
                  ))}
                </ol>
              );
            }
            return (
              <ul key={index} className="structured-bullet-list">
                {block.items.map((item, itemIdx) => (
                  <li key={itemIdx} className="structured-bullet-item">
                    <span className="bullet-indicator" />
                    <div className="item-content">{parseInline(item)}</div>
                  </li>
                ))}
              </ul>
            );
          }

          case 'key_value': {
            return (
              <div key={index} className="structured-kv-card">
                <span className="kv-label">{block.key}</span>
                <div className="kv-value">{parseInline(block.value)}</div>
              </div>
            );
          }

          case 'blockquote': {
            return (
              <blockquote key={index} className="structured-blockquote">
                <div className="quote-accent" />
                <div className="quote-text">{parseInline(block.text)}</div>
              </blockquote>
            );
          }

          case 'paragraph': {
            return (
              <p key={index} className="structured-paragraph">
                {parseInline(block.text)}
              </p>
            );
          }

          default:
            return null;
        }
      })}
    </div>
  );
}
