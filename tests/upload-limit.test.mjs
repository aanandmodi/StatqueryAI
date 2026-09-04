import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';

import ts from 'typescript';

// Exercise the actual framework preflight: testing POST() alone misses its earlier 1 MiB gate.
import { parseBodySizeLimit } from '../node_modules/vinext/dist/config/next-config.js';
import {
  handleProgressiveServerActionRequest,
  readActionFormDataWithLimit,
} from '../node_modules/vinext/dist/server/app-server-action-execution.js';

const source = await readFile(new URL('../next.config.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
const { default: config } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const configuredLimit = parseBodySizeLimit(config.experimental?.serverActions?.bodySizeLimit);
const mib = 1024 * 1024;

function multipartRequest(sizeBytes, declaredLength) {
  const form = new FormData();
  form.append('file', new Blob([new Uint8Array(sizeBytes)], { type: 'image/tiff' }), 'regression.tif');
  form.append('modality', 'optical');
  return new Request('http://localhost:3000/api/satquery/assets', {
    method: 'POST', body: form,
    headers: declaredLength === undefined ? {} : { 'content-length': String(declaredLength) },
  });
}

async function preflight(request, maxActionBodySize) {
  return handleProgressiveServerActionRequest({
    request,
    contentType: request.headers.get('content-type'),
    actionId: null,
    allowedOrigins: [],
    cleanPathname: '/api/satquery/assets',
    hasPageRoute: false,
    maxActionBodySize,
    clearRequestContext() {},
    getAndClearPendingCookies() { return []; },
    readFormDataWithLimit: readActionFormDataWithLimit,
    async decodeAction() { return null; },
    reportRequestError(error) { throw error; },
  });
}

void test('multipart preflight allows 50 MiB files plus bounded envelope overhead', () => {
  assert.equal(configuredLimit, 51 * mib);
});

void test('reproduces the old 1 MiB rejection before our upload route', async () => {
  const response = await preflight(multipartRequest(1, 6 * mib), parseBodySizeLimit(undefined));
  assert.equal(response.status, 413);
});

void test('6 MiB streamed multipart uploads now reach the route with their bytes intact', async () => {
  const request = multipartRequest(6 * mib);
  assert.equal(await preflight(request, configuredLimit), null);
  const body = await request.formData();
  assert.equal(body.get('file').size, 6 * mib);
  assert.equal(body.get('modality'), 'optical');
});

void test('over-limit request bodies are still rejected', async () => {
  const response = await preflight(multipartRequest(1, configuredLimit + 1), configuredLimit);
  assert.equal(response.status, 413);
});
