interface Env {
  STUDIO_API_ORIGIN: string;
  STUDIO_ORIGIN_VERIFY_SECRET: string;
}

const API_HOST = 'studio-api.dajoongbim.com';
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

function isPublicAssetRequest(request: Request, url: URL) {
  return request.method === 'GET'
    && url.pathname.startsWith('/api/assets/v1/');
}

function validatedOrigin(value: string) {
  const origin = new URL(value);
  if (origin.protocol !== 'https:' || !origin.hostname.endsWith('.awsapprunner.com')) {
    throw new Error('STUDIO_API_ORIGIN must be an HTTPS App Runner service URL');
  }
  return origin;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const incoming = new URL(request.url);
    if (incoming.hostname !== API_HOST) return new Response('Not Found', { status: 404 });

    const publicAsset = isPublicAssetRequest(request, incoming);
    if (publicAsset) {
      const cached = await caches.default.match(request);
      if (cached) return cached;
    }

    let origin: URL;
    try {
      origin = validatedOrigin(env.STUDIO_API_ORIGIN);
    } catch {
      return new Response('Service configuration unavailable', { status: 503 });
    }
    if (!env.STUDIO_ORIGIN_VERIFY_SECRET || env.STUDIO_ORIGIN_VERIFY_SECRET.length < 32) {
      return new Response('Service configuration unavailable', { status: 503 });
    }

    const target = new URL(`${incoming.pathname}${incoming.search}`, origin);
    const headers = new Headers(request.headers);
    const suppliedRequestId = headers.get('X-Request-ID') || '';
    const requestId = /^[A-Za-z0-9]{8,64}$/.test(suppliedRequestId)
      ? suppliedRequestId
      : crypto.randomUUID().replaceAll('-', '');
    const contentLength = Number(headers.get('Content-Length') || '0');
    if (Number.isFinite(contentLength) && contentLength > MAX_UPLOAD_BYTES) {
      return Response.json(
        { detail: 'drawing exceeds the 100 MB upload limit', request_id: requestId },
        { status: 413, headers: { 'Cache-Control': 'no-store', 'X-Request-ID': requestId } },
      );
    }
    headers.set('X-Dajoong-Origin-Verify', env.STUDIO_ORIGIN_VERIFY_SECRET);
    headers.set('X-Dajoong-Country', request.cf?.country || 'US');
    headers.set('X-Request-ID', requestId);
    headers.set('X-Forwarded-Host', incoming.host);
    headers.delete('Host');

    const upstreamRequest = new Request(target, {
      method: request.method,
      headers,
      body: request.method === 'GET' || request.method === 'HEAD' ? undefined : request.body,
      redirect: 'manual',
    });
    let upstream: Response;
    try {
      upstream = await fetch(upstreamRequest);
    } catch {
      return Response.json(
        { detail: 'The Dajoong service is temporarily unavailable.', request_id: requestId },
        { status: 502, headers: { 'Cache-Control': 'no-store', 'X-Request-ID': requestId } },
      );
    }
    const responseHeaders = new Headers(upstream.headers);
    responseHeaders.set('Strict-Transport-Security', 'max-age=63072000; includeSubDomains; preload');
    responseHeaders.set('X-Content-Type-Options', 'nosniff');
    responseHeaders.set('Referrer-Policy', 'no-referrer');
    if (!publicAsset) responseHeaders.set('Cache-Control', 'no-store');
    responseHeaders.set('X-Request-ID', upstream.headers.get('X-Request-ID') || requestId);
    const response = new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
    if (publicAsset && upstream.ok && request.method === 'GET') {
      await caches.default.put(request, response.clone());
    }
    return response;
  },
} satisfies ExportedHandler<Env>;
