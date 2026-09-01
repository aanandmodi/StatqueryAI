import { env } from 'cloudflare:workers';

const ALLOWED_TASKS = new Set(['single_vqa', 'caption', 'grounding']);
const DEFAULT_SPACE_URL = 'https://aanandmodi-satquery-qwen3vl-space.hf.space';
const MAX_ENCODED_CHARS = 14_000_000;

function binding(name: string): string {
  const runtimeValue = (env as unknown as Record<string, unknown>)[name];
  if (typeof runtimeValue === 'string' && runtimeValue.trim()) return runtimeValue.trim();
  return process.env[name]?.trim() ?? '';
}

function modelSpaceUrl(): string {
  const configured = binding('SATQUERY_MODEL_SPACE_URL') || DEFAULT_SPACE_URL;
  const url = new URL(configured);
  if (url.protocol !== 'https:' || !/^[a-z0-9-]+\.hf\.space$/i.test(url.hostname)) {
    throw new Error('The model Space URL is not an allowed Hugging Face host.');
  }
  return url.origin;
}

function completePayload(body: string): Record<string, unknown> {
  let event = '';
  for (const line of body.split(/\r?\n/)) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    if (!line.startsWith('data:')) continue;
    const data = line.slice(5).trim();
    if (event === 'error') throw new Error(`ZeroGPU returned an error: ${data.slice(0, 300)}`);
    if (!['complete', 'completed'].includes(event)) continue;
    let parsed: unknown = JSON.parse(data);
    if (Array.isArray(parsed) && parsed.length === 1) parsed = parsed[0];
    if (typeof parsed === 'string') parsed = JSON.parse(parsed);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('ZeroGPU returned an unexpected response shape.');
    }
    return parsed as Record<string, unknown>;
  }
  throw new Error('ZeroGPU did not return a complete queue event.');
}

function unavailable(message: string) {
  return Response.json(
    { error: { code: 'model_space_unavailable', message } },
    { status: 503, headers: { 'cache-control': 'no-store' } },
  );
}

export async function GET() {
  try {
    const response = await fetch(`${modelSpaceUrl()}/gradio_api/info`, {
      headers: { accept: 'application/json' },
      signal: AbortSignal.timeout(12_000),
    });
    if (!response.ok) return unavailable('The free ZeroGPU Space is building, sleeping, or awaiting its grant.');
    return Response.json({ status: 'ok', tasks: [...ALLOWED_TASKS], backend: 'sites-edge' });
  } catch {
    return unavailable('The free ZeroGPU Space is building, sleeping, or awaiting its grant.');
  }
}

export async function POST(request: Request) {
  let payload: Record<string, unknown>;
  try {
    payload = (await request.json()) as Record<string, unknown>;
  } catch {
    return Response.json({ error: { code: 'invalid_json', message: 'A JSON request body is required.' } }, { status: 400 });
  }

  const imageBase64 = typeof payload.image_base64 === 'string' ? payload.image_base64.trim() : '';
  const task = typeof payload.task === 'string' ? payload.task.trim().toLowerCase() : '';
  const question = typeof payload.question === 'string' ? payload.question.trim() : '';
  const maxNewTokens = Math.max(1, Math.min(256, Number(payload.max_new_tokens) || 128));
  if (!imageBase64 || imageBase64.length > MAX_ENCODED_CHARS) {
    return Response.json({ error: { code: 'invalid_image', message: 'The preview image is missing or exceeds 10 MB.' } }, { status: 422 });
  }
  if (!ALLOWED_TASKS.has(task) || question.length < 2 || question.length > 2000) {
    return Response.json({ error: { code: 'invalid_query', message: 'Choose a released task and a 2–2000 character question.' } }, { status: 422 });
  }

  try {
    const endpoint = `${modelSpaceUrl()}/gradio_api/call/analyze`;
    const submitted = await fetch(endpoint, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ data: [imageBase64, task, question, maxNewTokens] }),
      signal: AbortSignal.timeout(30_000),
    });
    if (!submitted.ok) throw new Error(`Space submission returned ${submitted.status}`);
    const eventId = String(((await submitted.json()) as { event_id?: string }).event_id ?? '').trim();
    if (!eventId) throw new Error('Space submission did not return an event ID.');

    const completed = await fetch(`${endpoint}/${encodeURIComponent(eventId)}`, {
      headers: { accept: 'text/event-stream' },
      signal: AbortSignal.timeout(150_000),
    });
    if (!completed.ok) throw new Error(`Space queue returned ${completed.status}`);
    const result = completePayload(await completed.text());
    return Response.json(result, { headers: { 'cache-control': 'no-store' } });
  } catch (error) {
    const detail = error instanceof Error ? error.message : 'unknown Space error';
    return unavailable(`Free ZeroGPU inference is unavailable or its quota is exhausted. ${detail}`);
  }
}

