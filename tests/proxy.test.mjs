import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { after, before, test } from 'node:test';

import ts from 'typescript';

// Compile the actual route without a web-framework test dependency. Its fetch
// calls below use Node's real HTTP transport, including streamed POST bodies.
const source = await readFile(new URL('../app/api/satquery/[...path]/route.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
const { GET, POST } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const received = [];
const originalBackend = process.env.SATQUERY_BACKEND_URL;
const originalKey = process.env.SATQUERY_BACKEND_API_KEY;
const server = createServer(async (request, response) => {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  const body = Buffer.concat(chunks).toString('utf8');
  received.push({ method: request.method, url: request.url, headers: request.headers, body });
  response.writeHead(request.method === 'POST' ? 202 : 200, { 'content-type': 'application/json' });
  response.end(JSON.stringify({ received: true }));
});

before(async () => {
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  process.env.SATQUERY_BACKEND_URL = `http://127.0.0.1:${address.port}`;
  process.env.SATQUERY_BACKEND_API_KEY = 'test-server-only-key';
});

after(async () => {
  if (originalBackend === undefined) delete process.env.SATQUERY_BACKEND_URL;
  else process.env.SATQUERY_BACKEND_URL = originalBackend;
  if (originalKey === undefined) delete process.env.SATQUERY_BACKEND_API_KEY;
  else process.env.SATQUERY_BACKEND_API_KEY = originalKey;
  server.closeAllConnections();
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
});

const context = (...path) => ({ params: Promise.resolve({ path }) });

void test('case pagination is bounded and extra query fields are dropped', async () => {
  const response = await GET(new Request('http://local.test/api/satquery/analyses?limit=20&offset=40&secret=drop'), context('analyses'));
  assert.equal(response.status, 200);
  await response.arrayBuffer();
  assert.equal(received.at(-1).url, '/v1/analyses?limit=20&offset=40');
  for (const query of ['limit=0', 'limit=99999', 'offset=-1', 'limit=2&limit=3']) {
    const before = received.length;
    const rejected = await GET(new Request('http://local.test/api/satquery/analyses?' + query), context('analyses'));
    assert.equal(rejected.status, 400);
    assert.equal(received.length, before);
  }
});

void test('only registered historical/weather context paths are forwarded', async () => {
  for (const name of ['history', 'weather']) {
    const response = await POST(new Request('http://local.test/api/satquery/evidence/' + name + '/search', {
      method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({test: true}),
    }), context('evidence', name, 'search'));
    assert.equal(response.status, 202);
    await response.arrayBuffer();
    assert.equal(received.at(-1).url, '/v1/evidence/' + name + '/search');
  }
  const before = received.length;
  const rejected = await POST(new Request('http://local.test/api/satquery/evidence/fetch/search', {
    method: 'POST', body: '{}',
  }), context('evidence', 'fetch', 'search'));
  assert.equal(rejected.status, 404);
  assert.equal(received.length, before);
});

void test('only forwards registered mask IDs and the boolean color switch', async () => {
  const response = await GET(new Request('http://local.test/api/satquery/analyses/anl_ab/masks/ev_mask_1?colored=true&url=https://evil.test'), context('analyses', 'anl_ab', 'masks', 'ev_mask_1'));
  assert.equal(response.status, 200);
  assert.equal(received.at(-1).url, '/v1/analyses/anl_ab/masks/ev_mask_1?colored=true');
  const count = received.length;
  const denied = await GET(new Request('http://local.test/api/satquery/analyses/anl_ab/masks/secret'), context('analyses', 'anl_ab', 'masks', 'secret'));
  assert.equal(denied.status, 404);
  assert.equal(received.length, count);
});

void test('streams a multipart image upload through real Node fetch', async () => {
  const form = new FormData();
  form.append('file', new Blob(['raster-test-bytes'], { type: 'image/tiff' }), 'scene.tif');
  form.append('modality', 'optical');
  const request = new Request('http://local.test/api/satquery/assets', { method: 'POST', body: form });
  const response = await POST(request, context('assets'));
  assert.equal(response.status, 202);
  assert.deepEqual(await response.json(), { received: true });
  const forwarded = received.at(-1);
  assert.equal(forwarded.method, 'POST');
  assert.equal(forwarded.url, '/v1/assets');
  assert.match(forwarded.headers['content-type'], /^multipart\/form-data; boundary=/);
  assert.match(forwarded.body, /filename="scene.tif"/);
  assert.match(forwarded.body, /raster-test-bytes/);
  assert.equal(forwarded.headers['x-api-key'], 'test-server-only-key');
  assert.equal(response.headers.get('x-api-key'), null);
});

void test('streams JSON analysis requests and preserves the idempotency key', async () => {
  const request = new Request('http://local.test/api/satquery/analyses', {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'idempotency-key': 'analysis-test-1' },
    body: JSON.stringify({ query: 'Describe this scene', asset_ids: ['ast_ab'] }),
  });
  const response = await POST(request, context('analyses'));
  assert.equal(response.status, 202);
  await response.arrayBuffer();
  const forwarded = received.at(-1);
  assert.equal(forwarded.headers['idempotency-key'], 'analysis-test-1');
  assert.deepEqual(JSON.parse(forwarded.body).asset_ids, ['ast_ab']);
});

void test('forwards only a validated overlay asset selector', async () => {
  const response = await GET(
    new Request('http://local.test/api/satquery/analyses/anl_ab/overlay?asset_id=ast_cd&secret=drop'),
    context('analyses', 'anl_ab', 'overlay'),
  );
  assert.equal(response.status, 200);
  await response.arrayBuffer();
  assert.equal(received.at(-1).url, '/v1/analyses/anl_ab/overlay?asset_id=ast_cd');
  const normal = await GET(
    new Request('http://local.test/api/satquery/capabilities?asset_id=ast_cd'), context('capabilities'),
  );
  await normal.arrayBuffer();
  assert.equal(received.at(-1).url, '/v1/capabilities');
});

void test('rejects invalid or duplicate overlay selectors without forwarding', async () => {
  const beforeCount = received.length;
  for (const query of ['asset_id=../bad', 'asset_id=ast_ab&asset_id=ast_cd']) {
    const response = await GET(
      new Request(`http://local.test/api/satquery/analyses/anl_ab/overlay?${query}`),
      context('analyses', 'anl_ab', 'overlay'),
    );
    assert.equal(response.status, 400);
  }
  assert.equal(received.length, beforeCount);
});

void test('malformed percent encoding returns 400 rather than throwing', async () => {
  const beforeCount = received.length;
  const response = await GET(new Request('http://local.test/api/satquery/%'), context('%'));
  assert.equal(response.status, 400);
  assert.equal((await response.json()).error.code, 'invalid_proxy_path');
  assert.equal(received.length, beforeCount);
});

void test('aborts a timed-out upstream request with an explicit 504', async (t) => {
  const durations = [];
  t.mock.method(AbortSignal, 'timeout', (duration) => {
    durations.push(duration);
    return AbortSignal.abort(new DOMException('Timed out', 'TimeoutError'));
  });
  const response = await GET(new Request('http://local.test/api/satquery/capabilities'), context('capabilities'));
  assert.equal(response.status, 504);
  assert.equal((await response.json()).error.code, 'backend_timeout');
  assert.deepEqual(durations, [30_000]);
});

void test('propagates client cancellation instead of starting a new upstream request', async () => {
  const request = new Request('http://local.test/api/satquery/capabilities', { signal: AbortSignal.abort() });
  const response = await GET(request, context('capabilities'));
  assert.equal(response.status, 499);
  assert.equal((await response.json()).error.code, 'request_cancelled');
});
