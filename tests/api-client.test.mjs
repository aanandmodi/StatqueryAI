import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';

import ts from 'typescript';

const source = await readFile(new URL('../lib/api-client.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
const { jsonRequest } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);

void test('network failure explains the connection problem without blaming the TIFF', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(jsonRequest('/api/satquery/assets'), /connection to the local website was interrupted/);
});

void test('a plain-text framework 413 has an actionable size-limit explanation', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('Payload Too Large', { status: 413 }));
  await assert.rejects(jsonRequest('/api/satquery/assets'), /50 MB.*restart the frontend/);
});

void test('preserves detailed backend validation errors', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ error: { message: 'Asset was stored but failed raster validation', details: { errors: ['TIFF is missing a CRS'] } } }, { status: 422 }));
  await assert.rejects(jsonRequest('/api/satquery/assets'), /failed raster validation: TIFF is missing a CRS/);
});

void test('does not pretend successful non-JSON responses are valid API data', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('<html>wrong route</html>', { status: 200 }));
  await assert.rejects(jsonRequest('/api/satquery/analyses'), /unreadable response instead of JSON/);
});

void test('successful JSON requests preserve multipart bodies and disable caching', async (t) => {
  const form = new FormData();
  form.append('file', new Blob(['raster'], { type: 'image/tiff' }), 'sample.tif');
  const mock = t.mock.method(globalThis, 'fetch', async () => Response.json({ id: 'ast_ab' }, { status: 201 }));
  assert.deepEqual(await jsonRequest('/api/satquery/assets', { method: 'POST', body: form }), { id: 'ast_ab' });
  assert.equal(mock.mock.calls[0].arguments[1].body, form);
  assert.equal(mock.mock.calls[0].arguments[1].cache, 'no-store');
});

void test('an intentional cancellation is distinguished from a network failure', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => { throw new DOMException('Aborted', 'AbortError'); });
  await assert.rejects(jsonRequest('/api/satquery/assets', { signal: AbortSignal.abort() }), /request was cancelled/);
});
