function statusMessage(status: number): string {
  if (status === 413) {
    return 'The upload exceeded a server request-size limit. Each file may be up to 50 MB. If your file is smaller, restart the frontend to load its upload-limit configuration.';
  }
  if (status === 502 || status === 503 || status === 504) {
    return 'The local backend or model connection is unavailable. Check the backend terminal and that the Kaggle/ngrok session is still running.';
  }
  return `Request failed with status ${status}`;
}

export async function jsonRequest<T>(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(input, { ...init, cache: 'no-store' });
  } catch {
    if (init?.signal?.aborted) throw new Error('The request was cancelled.');
    throw new Error(
      'The connection to the local website was interrupted before a response arrived. Refresh the page and check that the frontend and backend are running. Your file has not been confirmed as uploaded.',
    );
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    // Framework limits can return plain-text 413, not our backend's JSON error contract.
    if (!response.ok) throw new Error(statusMessage(response.status));
    throw new Error(
      'The local API returned an unreadable response instead of JSON. Check the frontend/backend logs and refresh the page.',
    );
  }
  if (!response.ok) {
    const error =
      payload && typeof payload === 'object' && 'error' in payload
        ? payload.error
        : null;
    const message =
      error && typeof error === 'object' && 'message' in error
        ? error.message
        : null;
    const details =
      error && typeof error === 'object' && 'details' in error
        ? error.details
        : null;
    const validationErrors =
      details &&
      typeof details === 'object' &&
      'errors' in details &&
      Array.isArray(details.errors)
        ? details.errors.filter(
            (item): item is string =>
              typeof item === 'string' && item.trim().length > 0,
          )
        : [];
    const apiDetails =
      payload &&
      typeof payload === 'object' &&
      'detail' in payload &&
      Array.isArray(payload.detail)
        ? payload.detail
            .map((item: { msg?: string }) => item.msg)
            .filter(Boolean)
            .join('; ')
        : '';
    const base =
      typeof message === 'string' && message.trim()
        ? message
        : apiDetails || statusMessage(response.status);
    throw new Error(
      validationErrors.length
        ? `${base}: ${validationErrors.join('; ')}`
        : base,
    );
  }
  return payload as T;
}
