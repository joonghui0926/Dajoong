interface Env {
  ASSETS: Fetcher;
}

const MARKETING_HOST = 'dajoongbim.com';
const WWW_HOST = 'www.dajoongbim.com';
const STUDIO_HOST = 'studio.dajoongbim.com';
const APP_LINK_HOST = 'app.dajoongbim.com';
const ALLOWED_HOSTS = new Set([MARKETING_HOST, WWW_HOST, STUDIO_HOST, APP_LINK_HOST]);
const ASSOCIATION_PATHS = new Set(['/.well-known/apple-app-site-association', '/.well-known/assetlinks.json']);

function secure(response: Response, requestUrl: URL): Response {
  const headers = new Headers(response.headers);
  const isEmbeddableStudio = requestUrl.hostname === STUDIO_HOST && requestUrl.pathname.startsWith('/studio');
  headers.set('Content-Security-Policy', [
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    "connect-src 'self' https://studio-api.dajoongbim.com https://*.amazoncognito.com https://*.amazonaws.com",
    "worker-src 'self' blob:",
    "frame-src 'self' https://studio.dajoongbim.com",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    isEmbeddableStudio
      ? "frame-ancestors 'self' https://dajoongbim.com https://www.dajoongbim.com"
      : "frame-ancestors 'none'",
  ].join('; '));
  headers.set('Strict-Transport-Security', 'max-age=63072000; includeSubDomains; preload');
  headers.set('X-Content-Type-Options', 'nosniff');
  if (isEmbeddableStudio) headers.delete('X-Frame-Options');
  else headers.set('X-Frame-Options', 'DENY');
  headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  headers.set('Cross-Origin-Opener-Policy', 'same-origin-allow-popups');
  headers.set('Cross-Origin-Resource-Policy', 'same-site');
  if (requestUrl.pathname.startsWith('/.well-known/')) {
    headers.set('Content-Type', 'application/json; charset=utf-8');
    headers.set('Cache-Control', 'public, max-age=300');
  }
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function permanentRedirect(url: URL, hostname: string): Response {
  const target = new URL(url);
  target.protocol = 'https:';
  target.hostname = hostname;
  target.port = '';
  return Response.redirect(target.toString(), 308);
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (!ALLOWED_HOSTS.has(url.hostname)) return secure(new Response('Not Found', { status: 404 }), url);

    // Association files must be served directly from every native-app hostname;
    // redirecting these verification requests can make Universal/App Links fail.
    if (ASSOCIATION_PATHS.has(url.pathname)) {
      return secure(await env.ASSETS.fetch(request), url);
    }

    if (url.hostname === WWW_HOST) return secure(permanentRedirect(url, MARKETING_HOST), url);
    if (url.hostname === APP_LINK_HOST) return secure(permanentRedirect(url, STUDIO_HOST), url);
    if (url.hostname === MARKETING_HOST && (url.pathname === '/studio' || url.pathname.startsWith('/studio/'))) {
      return secure(permanentRedirect(url, STUDIO_HOST), url);
    }

    let response = await env.ASSETS.fetch(request);
    if (response.status === 404 && request.method === 'GET') {
      response = await env.ASSETS.fetch(new Request(new URL('/index.html', url), request));
    }
    return secure(response, url);
  },
} satisfies ExportedHandler<Env>;
