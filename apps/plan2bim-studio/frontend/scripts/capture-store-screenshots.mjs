import { cp, mkdir, readFile, rm } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const root = fileURLToPath(new URL('..', import.meta.url));
const viteCli = join(root, 'node_modules', 'vite', 'bin', 'vite.js');
const server = spawn(process.execPath, [viteCli, 'preview', '--host', '127.0.0.1', '--port', '4178'], {
  cwd: root,
  env: { ...process.env, VITE_COGNITO_AUTHORITY: '', VITE_COGNITO_CLIENT_ID: '' },
  stdio: ['ignore', 'pipe', 'pipe'],
});

const baseUrl = 'http://127.0.0.1:4178';
for (let attempt = 0; attempt < 80; attempt += 1) {
  try {
    const response = await fetch(baseUrl);
    if (response.ok) break;
  } catch {
    if (attempt === 79) throw new Error('Studio screenshot server did not start');
  }
  await new Promise((resolve) => setTimeout(resolve, 250));
}

const iosDirectory = join(root, 'store', 'app-store', 'screenshots', 'en-US');
const iosKoreanDirectory = join(root, 'store', 'app-store', 'screenshots', 'ko');
const androidImages = join(root, 'store', 'google-play', 'metadata', 'en-US', 'images');
const androidScreenshots = join(androidImages, 'phoneScreenshots');
const androidKoreanImages = join(root, 'store', 'google-play', 'metadata', 'ko-KR', 'images');
await rm(join(root, 'store', 'app-store', 'screenshots'), { recursive: true, force: true });
await rm(androidImages, { recursive: true, force: true });
await mkdir(iosDirectory, { recursive: true });
await mkdir(androidScreenshots, { recursive: true });

const browser = await chromium.launch();
const captures = [
  ['01-convert-drawings', '/'],
  ['02-linked-review', '/studio'],
];

async function captureSet(directory, viewport, deviceScaleFactor) {
  const context = await browser.newContext({ viewport, deviceScaleFactor, colorScheme: 'light' });
  await context.addInitScript(() => {
    localStorage.setItem('dajoong-cookie-consent-v2', JSON.stringify({ essential: true, analytics: false, policyVersion: '2026-08-09', recordedAt: '2026-08-09T00:00:00.000Z' }));
  });
  const page = await context.newPage();
  for (const [name, path] of captures) {
    await page.goto(`${baseUrl}${path}`, { waitUntil: 'commit', timeout: 30_000 });
    await page.waitForTimeout(path === '/studio' ? 1_800 : 800);
    await page.screenshot({ path: join(directory, `${name}.png`), fullPage: false });
  }
  await context.close();
}

try {
  await captureSet(iosDirectory, { width: 440, height: 956 }, 3);
  await captureSet(androidScreenshots, { width: 360, height: 640 }, 3);

  const logo = await readFile(join(root, 'public', 'brand', 'dajoong-logo-mark-512.png'));
  const page = await browser.newPage({ viewport: { width: 1024, height: 500 }, deviceScaleFactor: 1 });
  await page.setContent(`<!doctype html><style>*{box-sizing:border-box}body{margin:0;width:1024px;height:500px;display:flex;align-items:center;padding:72px;background:linear-gradient(128deg,#f8f3ea 0%,#eef4f1 58%,#dcebe5 100%);font-family:Arial,sans-serif;color:#123044}.mark{width:260px;height:260px;object-fit:contain;filter:drop-shadow(0 20px 30px rgba(18,48,68,.16))}.copy{margin-left:66px}.copy small{font-size:21px;letter-spacing:.16em;color:#557168}.copy h1{margin:14px 0 12px;font-size:58px;letter-spacing:-.05em}.copy p{margin:0;max-width:520px;font-size:26px;line-height:1.35;color:#45616b}</style><img class="mark" src="data:image/png;base64,${logo.toString('base64')}"><div class="copy"><small>DAJOONG</small><h1>Plan2BIM Studio</h1><p>Drawings to editable, source-linked BIM.</p></div>`);
  await page.screenshot({ path: join(androidImages, 'featureGraphic.png') });
  await page.close();
  await cp(join(root, 'public', 'brand', 'dajoong-logo-mark-512.png'), join(androidImages, 'icon.png'));
  await cp(iosDirectory, iosKoreanDirectory, { recursive: true });
  await cp(androidImages, androidKoreanImages, { recursive: true });
} finally {
  await browser.close();
  server.kill();
}

console.log('Created App Store and Google Play screenshots from the production client.');
