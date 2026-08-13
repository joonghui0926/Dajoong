import { mkdir, rm } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const root = fileURLToPath(new URL('..', import.meta.url));
const repositoryRoot = fileURLToPath(new URL('../../../..', import.meta.url));
const viteCli = join(root, 'node_modules', 'vite', 'bin', 'vite.js');
const outputRoot = join(repositoryRoot, 'artifacts', 'screenshots');
const webOutput = join(outputRoot, 'web');
const appOutput = join(outputRoot, 'app');
const authOnly = process.argv.includes('--auth-only');

const routes = [
  ['01-landing', '/'],
  ['03-studio', '/studio'],
  ['05-privacy', '/privacy'],
  ['06-cookies', '/cookies'],
  ['07-terms', '/terms'],
  ['08-support', '/support'],
  ['09-account-deletion', '/account-deletion'],
];

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitForServer(baseUrl) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(baseUrl);
      if (response.ok) return;
    } catch {
      // The dev server is still starting.
    }
    await delay(250);
  }
  throw new Error(`Screenshot server did not start at ${baseUrl}`);
}

async function withServer({ port, mode }, callback) {
  const args = [viteCli, '--host', '127.0.0.1', '--port', String(port), '--strictPort'];
  if (mode) args.push('--mode', mode);
  const server = spawn(process.execPath, args, {
    cwd: root,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const baseUrl = `http://127.0.0.1:${port}`;
  try {
    await waitForServer(baseUrl);
    await callback(baseUrl);
  } finally {
    server.kill();
  }
}

async function createContext(browser, viewport, deviceScaleFactor = 1) {
  const context = await browser.newContext({ viewport, deviceScaleFactor, colorScheme: 'light' });
  await context.addInitScript(() => {
    localStorage.setItem('dajoong-cookie-consent-v2', JSON.stringify({
      essential: true,
      analytics: false,
      policyVersion: '2026-08-09',
      recordedAt: '2026-08-09T00:00:00.000Z',
    }));
  });
  return context;
}

async function captureProductPages(browser, baseUrl, directory, viewport, { includeLanding }) {
  const context = await createContext(browser, viewport);
  const page = await context.newPage();
  const selectedRoutes = includeLanding ? routes : routes.filter(([name]) => name !== '01-landing');
  for (const [name, path] of selectedRoutes) {
    await page.goto(`${baseUrl}${path}`, { waitUntil: 'domcontentloaded', timeout: 30_000 });
    await delay(path === '/studio' ? 2_500 : 500);
    await page.screenshot({ path: join(directory, `${name}.png`), fullPage: false });
    if (path === '/') {
      await page.screenshot({ path: join(directory, '01-landing-full.png'), fullPage: true });
      const scrollMetrics = await page.evaluate(() => {
        const scroller = document.querySelector('.landing-page');
        return scroller
          ? { scrollHeight: scroller.scrollHeight, clientHeight: scroller.clientHeight }
          : { scrollHeight: document.documentElement.scrollHeight, clientHeight: window.innerHeight };
      });
      const maxScroll = Math.max(0, scrollMetrics.scrollHeight - scrollMetrics.clientHeight);
      const step = Math.max(1, Math.round(scrollMetrics.clientHeight * 0.82));
      let section = 2;
      for (let y = step; y < maxScroll; y += step) {
        await page.evaluate((top) => {
          const scroller = document.querySelector('.landing-page');
          if (scroller) scroller.scrollTo({ top, behavior: 'instant' });
          else window.scrollTo({ top, behavior: 'instant' });
        }, y);
        await delay(150);
        await page.screenshot({ path: join(directory, `01-landing-section-${String(section).padStart(2, '0')}.png`), fullPage: false });
        section += 1;
      }
      if (maxScroll > 0) {
        await page.evaluate((top) => {
          const scroller = document.querySelector('.landing-page');
          if (scroller) scroller.scrollTo({ top, behavior: 'instant' });
          else window.scrollTo({ top, behavior: 'instant' });
        }, maxScroll);
        await delay(150);
        await page.screenshot({ path: join(directory, `01-landing-section-${String(section).padStart(2, '0')}.png`), fullPage: false });
      }
      await page.evaluate(() => {
        const scroller = document.querySelector('.landing-page');
        if (scroller) scroller.scrollTo({ top: 0, behavior: 'instant' });
        else window.scrollTo({ top: 0, behavior: 'instant' });
      });
    }
    if (path === '/studio') {
      await page.getByRole('button', { name: 'Convert', exact: true }).click();
      await delay(250);
      await page.screenshot({ path: join(directory, '04-conversion-dialog.png'), fullPage: false });
    }
  }
  await context.close();
}

async function captureSignIn(browser, baseUrl, directory, viewport) {
  const context = await createContext(browser, viewport);
  const page = await context.newPage();
  await page.goto(`${baseUrl}/studio`, { waitUntil: 'domcontentloaded', timeout: 30_000 });
  await page.getByRole('button', { name: 'Continue with email' }).waitFor();
  await page.screenshot({ path: join(directory, '02-sign-in.png'), fullPage: false });
  await context.close();
}

if (!authOnly) await rm(outputRoot, { recursive: true, force: true });
await Promise.all([mkdir(webOutput, { recursive: true }), mkdir(appOutput, { recursive: true })]);

const browser = await chromium.launch();
try {
  if (!authOnly) {
    await withServer({ port: 4180 }, async (baseUrl) => {
      await captureProductPages(browser, baseUrl, webOutput, { width: 1440, height: 1000 }, { includeLanding: true });
      await captureProductPages(browser, baseUrl, appOutput, { width: 440, height: 956 }, { includeLanding: true });
    });
  }
  await withServer({ port: 4181, mode: 'capture' }, async (baseUrl) => {
    await captureSignIn(browser, baseUrl, webOutput, { width: 1440, height: 1000 });
    await captureSignIn(browser, baseUrl, appOutput, { width: 440, height: 956 });
  });
} finally {
  await browser.close();
}

console.log(`Created product review screenshots in ${outputRoot}`);
