import { env } from 'cloudflare:workers';

type RouteContext = { params: Promise<{ path: string[] }> };
type Method = 'GET' | 'POST';

const READ_PATHS = [
  /^health\/ready$/,
  /^capabilities$/,
  /^assets\/ast_[a-f0-9]+$/,
  /^assets\/ast_[a-f0-9]+\/preview$/,
  /^analyses\/anl_[a-f0-9]+$/,
  /^analyses\/anl_[a-f0-9]+\/report$/,
];
const WRITE_PATHS = [/^assets$/, /^analyses$/];

function binding(name: string): string {
  const runtimeValue = (env as unknown as Record<string, unknown>)[name];
  if (typeof runtimeValue === 'string' && runtimeValue.trim()) {
    return runtimeValue.trim();
  }
  return process.env[name]?.trim() ?? '';
}

function permitted(method: Method, path: string): boolean {
  const patterns = method === 'GET' ? READ_PATHS : WRITE_PATHS;
  return patterns.some((pattern) => pattern.test(path));
}

async function proxy(request: Request, context: RouteContext, method: Method) {
  const { path: segments } = await context.params;
  const path = segments.map((segment) => decodeURIComponent(segment)).join('/');
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

  const headers = new Headers();
  const contentType = request.headers.get('content-type');
  const idempotencyKey = request.headers.get('idempotency-key');
  const apiKey = binding('SATQUERY_BACKEND_API_KEY');
  if (contentType) headers.set('content-type', contentType);
  if (idempotencyKey) headers.set('idempotency-key', idempotencyKey.slice(0, 160));
  if (apiKey) headers.set('x-api-key', apiKey);

  try {
    const upstream = await fetch(target, {
      method,
      headers,
      body: method === 'POST' ? request.body : undefined,
      redirect: 'manual',
    });
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
    return Response.json(
      {
        error: {
          code: 'backend_unreachable',
          message: 'The free API Space is sleeping or unavailable. Wake it and retry shortly.',
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

