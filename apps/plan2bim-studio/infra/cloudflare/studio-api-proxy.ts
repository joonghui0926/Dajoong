interface Env {
  STUDIO_API_ORIGIN: string;
  STUDIO_ORIGIN_VERIFY_SECRET: string;
}

const API_HOST = 'studio-api.builiconstruction.com';

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
    headers.set('X-Dajoong-Origin-Verify', env.STUDIO_ORIGIN_VERIFY_SECRET);
    headers.set('X-Forwarded-Host', incoming.host);
    headers.delete('Host');

    const upstreamRequest = new Request(target, {
      method: request.method,
      headers,
      body: request.method === 'GET' || request.method === 'HEAD' ? undefined : request.body,
      redirect: 'manual',
    });
    const upstream = await fetch(upstreamRequest);
    const responseHeaders = new Headers(upstream.headers);
    responseHeaders.set('Strict-Transport-Security', 'max-age=63072000; includeSubDomains; preload');
    responseHeaders.set('X-Content-Type-Options', 'nosniff');
    responseHeaders.set('Referrer-Policy', 'no-referrer');
    responseHeaders.set('Cache-Control', 'no-store');
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  },
} satisfies ExportedHandler<Env>;
