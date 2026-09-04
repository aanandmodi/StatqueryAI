type RouteContext = { params: Promise<{ path: string[] }> };
type Method = 'GET' | 'POST';
type StreamingRequestInit = RequestInit & { duplex?: 'half' };

const READ_PATHS = [
  /^analyses$/,
  /^health\/ready$/,
  /^capabilities$/,
  /^assets\/ast_[a-f0-9]+$/,
  /^assets\/ast_[a-f0-9]+\/preview$/,
  /^analyses\/anl_[a-f0-9]+$/,
  /^analyses\/anl_[a-f0-9]+\/report$/,
  /^analyses\/anl_[a-f0-9]+\/overlay$/,
  /^analyses\/anl_[a-f0-9]+\/masks\/ev_mask_[1-8]$/,
];
const WRITE_PATHS = [/^assets$/, /^analyses$/, /^evidence\/(history|weather)\/search$/];

function binding(name: string): string {
  return process.env[name]?.trim() ?? '';
}

function permitted(method: Method, path: string): boolean {
  const patterns = method === 'GET' ? READ_PATHS : WRITE_PATHS;
  return patterns.some((pattern) => pattern.test(path));
}

async function proxy(request: Request, context: RouteContext, method: Method) {
  const { path: segments } = await context.params;
  let path: string;
  try {
    path = segments.map((segment) => decodeURIComponent(segment)).join('/');
  } catch {
    return Response.json({ error: { code: 'invalid_proxy_path', message: 'The route contains invalid URL encoding.' } }, { status: 400 });
  }
  if (!permitted(method, path)) {
    return Response.json({ error: { code: 'proxy_route_denied', message: 'Route is not permitted.' } }, { status: 404 });
  }

  const configuredUrl = binding('SATQUERY_BACKEND_URL');
  const backendUrl = configuredUrl || (process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : '');
  if (!backendUrl) {
    return Response.json(
      { error: { code: 'backend_not_configured', message: 'The SatQuery API URL is not configured.' } },
      { status: 503 },
    );
  }

  let target: URL;
  try {
    target = new URL(`/v1/${path}`, backendUrl);
    if (!['http:', 'https:'].includes(target.protocol)) throw new Error('invalid protocol');
  } catch {
    return Response.json({ error: { code: 'backend_url_invalid', message: 'The SatQuery API URL is invalid.' } }, { status: 503 });
  }

  // No arbitrary query forwarding: selected asset or a bounded mask color switch only.
  if (method === 'GET' && path === 'analyses') {
    const params = new URL(request.url).searchParams;
    for (const [key, max] of [['limit', 100], ['offset', 10000]] as const) {
      const value = params.get(key);
      if (value !== null) {
        if (params.getAll(key).length !== 1 || !/^\d+$/.test(value) || Number(value) > max || (key === 'limit' && Number(value) < 1)) {
          return Response.json({error: {message: 'Invalid casebook pagination.'}}, {status: 400});
        }
        target.searchParams.set(key, value);
      }
    }
  }
  if (method === 'GET' && /^analyses\/anl_[a-f0-9]+\/masks\/ev_mask_[1-8]$/.test(path)) {
    if (new URL(request.url).searchParams.get('colored') === 'true') target.searchParams.set('colored', 'true');
  }
  if (method === 'GET' && /^analyses\/anl_[a-f0-9]+\/overlay$/.test(path)) {
    const sourceQuery = new URL(request.url).searchParams;
    const assetIds = sourceQuery.getAll('asset_id');
    if (assetIds.length > 1 || (assetIds.length === 1 && !/^ast_[a-f0-9]+$/.test(assetIds[0]))) {
      return Response.json({ error: { code: 'invalid_asset_id', message: 'The overlay asset selector is invalid.' } }, { status: 400 });
    }
    if (assetIds.length === 1) target.searchParams.set('asset_id', assetIds[0]);
  }

  const headers = new Headers();
  const contentType = request.headers.get('content-type');
  const idempotencyKey = request.headers.get('idempotency-key');
  const apiKey = binding('SATQUERY_BACKEND_API_KEY');
  if (contentType) headers.set('content-type', contentType);
  if (idempotencyKey) headers.set('idempotency-key', idempotencyKey.slice(0, 160));
  if (apiKey) headers.set('x-api-key', apiKey);

  // These routes only upload or create/poll jobs; inference runs asynchronously in
  // the controller. Keep streams bounded, including response-body consumption.
  const timeoutSignal = AbortSignal.timeout(method === 'POST' ? 120_000 : 30_000);
  const signal = AbortSignal.any([request.signal, timeoutSignal]);
  try {
    const init: StreamingRequestInit = {
      method,
      headers,
      body: method === 'POST' ? request.body : undefined,
      // Node fetch requires this when forwarding a ReadableStream request body.
      ...(method === 'POST' && request.body ? { duplex: 'half' as const } : {}),
      redirect: 'manual',
      signal,
    };
    const upstream = await fetch(target, init);
    const responseHeaders = new Headers();
    for (const name of ['content-type', 'content-disposition', 'cache-control']) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    responseHeaders.set('x-content-type-options', 'nosniff');
    return new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    });
  } catch {
    if (timeoutSignal.aborted) {
      return Response.json(
        { error: { code: 'backend_timeout', message: 'The local controller request timed out. Check its terminal before retrying.' } },
        { status: 504 },
      );
    }
    if (request.signal.aborted) {
      return Response.json(
        { error: { code: 'request_cancelled', message: 'The client disconnected or cancelled the request.' } },
        { status: 499 },
      );
    }
    return Response.json(
      {
        error: {
          code: 'backend_unreachable',
          message: 'The local SatQuery backend is not reachable. Start it on http://127.0.0.1:8000.',
        },
      },
      { status: 503 },
    );
  }
}

export async function GET(request: Request, context: RouteContext) {
  return proxy(request, context, 'GET');
}

export async function POST(request: Request, context: RouteContext) {
  return proxy(request, context, 'POST');
}
