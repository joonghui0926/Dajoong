import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const root = fileURLToPath(new URL('..', import.meta.url));
const repositoryRoot = fileURLToPath(new URL('../../../..', import.meta.url));
const viteCli = join(root, 'node_modules', 'vite', 'bin', 'vite.js');
const server = spawn(process.execPath, [viteCli, 'preview', '--host', '127.0.0.1', '--port', '4178', '--strictPort'], {
  cwd: root,
  env: { ...process.env, VITE_COGNITO_AUTHORITY: '', VITE_COGNITO_CLIENT_ID: '' },
  stdio: ['ignore', 'pipe', 'pipe'],
});

const baseUrl = 'http://127.0.0.1:4178';
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitForServer() {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(baseUrl);
      if (response.ok) return;
    } catch {
      // Vite preview is still starting.
    }
    await delay(250);
  }
  throw new Error('Studio screenshot server did not start');
}

await waitForServer();

const iosRoot = join(root, 'store', 'app-store', 'screenshots');
const androidRoot = join(root, 'store', 'google-play', 'metadata');
const webRoot = join(root, 'store', 'web-showcase');
const artifactRoot = join(repositoryRoot, 'artifacts', 'store-showcase');
const rawMobile = join(artifactRoot, 'raw', 'mobile');
const rawWeb = join(artifactRoot, 'raw', 'web');

await rm(iosRoot, { recursive: true, force: true });
await rm(webRoot, { recursive: true, force: true });
await rm(artifactRoot, { recursive: true, force: true });
for (const locale of ['en-US', 'ko-KR']) {
  await rm(join(androidRoot, locale, 'images'), { recursive: true, force: true });
}
await Promise.all([
  mkdir(join(iosRoot, 'en-US'), { recursive: true }),
  mkdir(join(iosRoot, 'ko'), { recursive: true }),
  mkdir(join(androidRoot, 'en-US', 'images', 'phoneScreenshots'), { recursive: true }),
  mkdir(join(androidRoot, 'ko-KR', 'images', 'phoneScreenshots'), { recursive: true }),
  mkdir(join(webRoot, 'en-US'), { recursive: true }),
  mkdir(join(webRoot, 'ko-KR'), { recursive: true }),
  mkdir(rawMobile, { recursive: true }),
  mkdir(rawWeb, { recursive: true }),
]);

const stories = [
  {
    id: '01-landing',
    action: 'landing',
    copy: {
      'en-US': ['DAJOONG PLAN2BIM', 'From drawing to building data.', 'Open the source-linked BIM workspace from one focused starting point.'],
      'ko-KR': ['DAJOONG PLAN2BIM', '도면에서 건물 데이터까지.', '하나의 명확한 시작점에서 도면과 연결된 BIM 작업공간을 여세요.'],
    },
    alt: {
      'en-US': 'Dajoong landing page introducing the source-linked drawing-to-BIM workspace.',
      'ko-KR': '도면과 연결된 BIM 작업공간을 소개하는 Dajoong 랜딩 페이지.',
    },
  },
  {
    id: '02-linked-bim',
    action: 'model',
    copy: {
      'en-US': ['DRAWING + MODEL', 'See the building, not just the sheet.', 'A live 3D model stays connected to the source drawing.'],
      'ko-KR': ['도면 + 모델', '도면을 넘어 건물을 확인하세요.', '3D 모델이 원본 도면과 계속 연결됩니다.'],
    },
    alt: {
      'en-US': 'Dajoong Studio showing a color 3D building model generated from a floor plan.',
      'ko-KR': '평면도에서 생성한 컬러 3D 건물 모델을 보여주는 Dajoong Studio.',
    },
  },
  {
    id: '03-convert-drawing',
    action: 'convert',
    copy: {
      'en-US': ['LIGHTWEIGHT CONVERSION', 'Turn a drawing into editable BIM.', 'Upload one floor or a complete building set.'],
      'ko-KR': ['경량 변환', '도면을 편집 가능한 BIM으로.', '한 층부터 건물 전체 도면까지 변환합니다.'],
    },
    alt: {
      'en-US': 'Dajoong conversion dialog for uploading a floor plan or multi-level drawing set.',
      'ko-KR': '평면도 또는 여러 층 도면 세트를 올리는 Dajoong 변환 창.',
    },
  },
  {
    id: '04-source-plan',
    action: 'plan',
    copy: {
      'en-US': ['SOURCE-LINKED', 'Review every element against the plan.', 'Walls, openings, rooms, and equipment remain selectable.'],
      'ko-KR': ['원본 연결', '모든 요소를 도면과 함께 검토하세요.', '벽과 문, 공간과 설비를 각각 선택할 수 있습니다.'],
    },
    alt: {
      'en-US': 'Dajoong plan view with individually selectable walls, doors, rooms, and equipment.',
      'ko-KR': '벽과 문, 공간과 설비를 개별 선택할 수 있는 Dajoong 도면 화면.',
    },
  },
  {
    id: '05-model-assurance',
    action: 'quality',
    copy: {
      'en-US': ['MODEL ASSURANCE', 'Review risk before release.', 'Evidence, topology checks, and priorities stay in one queue.'],
      'ko-KR': ['모델 검수', '배포 전에 위험을 먼저 확인하세요.', '근거와 정합성 검사, 우선순위를 한곳에서 관리합니다.'],
    },
    alt: {
      'en-US': 'Dajoong model assurance screen with guided review priorities and integrity checks.',
      'ko-KR': '검토 우선순위와 정합성 검사를 보여주는 Dajoong 모델 검수 화면.',
    },
  },
  {
    id: '06-secure-checkout',
    action: 'checkout',
    copy: {
      'en-US': ['SIMPLE CHECKOUT', 'Choose one drawing or unlimited.', 'The first drawing is free. Continue for $3.99 or use unlimited for $79 a month.'],
      'ko-KR': ['간편 결제', '도면 한 장 또는 무제한.', '첫 도면은 무료입니다. 이후 장당 $3.99 또는 월 $79 무제한을 선택하세요.'],
    },
    alt: {
      'en-US': 'Dajoong secure checkout with the first drawing free and per-drawing credits.',
      'ko-KR': '첫 도면 무료와 도면별 크레딧을 보여주는 Dajoong 보안 결제 화면.',
    },
  },
];

const appStories = stories.filter((story) => story.action !== 'quality');
const appAssetId = (story) => story.action === 'checkout' ? '05-secure-checkout' : story.id;

const checkoutContext = {
  country: 'US',
  currency: 'USD',
  unit_amount: 399,
  unit_label: '$3.99 / drawing',
  monthly_amount: 7900,
  monthly_label: '$79 / month',
  unlimited_active: false,
  unlimited_until: 0,
  free_units_remaining: 1,
  paid_units: 0,
  billing_enforced: true,
  configured_providers: ['stripe'],
  native_provider: '',
  comparison_multiple: 5.01,
  comparison_basis: 'T-company $20 minimum 3D order',
  monthly_comparison_multiple: 4.81,
  monthly_comparison_basis: 'Autodesk Revit standard monthly subscription ($380/month)',
  monthly_comparison_source_url: 'https://www.autodesk.com/solutions/revit-subscription-faq',
  comparison_source_url: 'https://support.twindo.com/article/716-how-does-scan-to-cad-pricing-work',
  speed_median_seconds: 2.720126,
  speed_p95_seconds: 12.972123,
  speed_runs: 7,
  speed_comparison_multiple: 21175,
  speed_benchmark_url: '/benchmarks/plan2bim-speed-2026-08-09.json',
  speed_turnaround_source_url: 'https://support.twindo.com/article/721-how-long-will-it-take-to-receive-my-order',
};

async function createCaptureContext(browser, viewport) {
  const context = await browser.newContext({ viewport, deviceScaleFactor: 1, colorScheme: 'light' });
  await context.addInitScript(() => {
    localStorage.setItem('dajoong-cookie-consent-v2', JSON.stringify({
      essential: true,
      analytics: false,
      policyVersion: '2026-08-09',
      recordedAt: '2026-08-09T00:00:00.000Z',
    }));
    localStorage.removeItem('dajoong-plan2bim-studio-session-v1');
  });
  await context.route('**/api/billing/context**', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(checkoutContext) });
  });
  return context;
}

async function openStory(page, story, surface) {
  if (story.action === 'landing') {
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 30_000 });
    await page.locator('.landing-page').waitFor({ state: 'visible', timeout: 30_000 });
    await delay(1_800);
    return;
  }
  await page.goto(`${baseUrl}/studio`, { waitUntil: 'domcontentloaded', timeout: 30_000 });
  await page.locator('.studio-shell').waitFor({ state: 'visible', timeout: 30_000 });
  await delay(1_800);
  if (story.action === 'model') {
    if (surface === 'mobile') await page.getByTitle('3D only').click();
  } else if (story.action === 'convert') {
    await page.getByRole('button', { name: 'Convert', exact: true }).click();
    await page.locator('.conversion-dialog').waitFor({ state: 'visible' });
  } else if (story.action === 'plan') {
    await page.getByTitle('Plan only').click();
  } else if (story.action === 'quality') {
    await page.locator('.quality-pill').evaluate((element) => element.click());
    await page.locator('.quality-review').waitFor({ state: 'visible' });
  } else if (story.action === 'checkout') {
    await page.locator('.header-actions .header-button').filter({ hasText: 'Credits' }).evaluate((element) => element.click());
    await page.locator('.checkout-shell').waitFor({ state: 'visible' });
    await page.locator('.checkout-pipeline-image').evaluate(async (image) => {
      if (image.complete && image.naturalWidth > 0) return;
      await new Promise((resolve, reject) => {
        image.addEventListener('load', resolve, { once: true });
        image.addEventListener('error', reject, { once: true });
      });
    });
    if (surface === 'mobile') {
      await page.locator('.checkout-backdrop').evaluate((element) => element.scrollTo({ top: element.scrollHeight, behavior: 'instant' }));
    }
  }
  await delay(350);
}

async function captureRawSet(browser, viewport, directory, surface) {
  const context = await createCaptureContext(browser, viewport);
  for (const story of stories) {
    const page = await context.newPage();
    await openStory(page, story, surface);
    await page.screenshot({ path: join(directory, `${story.id}.png`), fullPage: false });
    await page.close();
  }
  await context.close();
}

function escapeHtml(value) {
  return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}

async function dataUrl(path) {
  return `data:image/png;base64,${(await readFile(path)).toString('base64')}`;
}

const palettes = [
  ['#f0f3ee', '#102f3a', '#348070'],
  ['#edf4f1', '#0e2f3a', '#4f8d78'],
  ['#f7f1e8', '#132f3a', '#d5a957'],
  ['#e8f1f4', '#102d3a', '#5b93a8'],
  ['#f3eee7', '#112d38', '#a86b56'],
  ['#e9f0ed', '#112e39', '#37765e'],
];

async function renderPhoneAsset(browser, source, target, size, copy, palette) {
  const [width, height] = size;
  const [, ink, accent] = palette;
  const sourceUrl = await dataUrl(source);
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1, colorScheme: 'light' });
  const page = await context.newPage();
  await page.setContent(`<!doctype html><html><head><meta charset="utf-8"><style>
    *{box-sizing:border-box}html,body{margin:0;width:${width}px;height:${height}px;overflow:hidden}
    body{font-family:Arial,Helvetica,sans-serif;color:${ink};background:#fff}
    .copy{position:absolute;z-index:3;top:4.1%;left:7.8%;right:7.8%}
    .copy small{display:block;margin-bottom:1.1%;color:${accent};font-size:${Math.round(width * .019)}px;font-weight:800;letter-spacing:.16em}
    .copy h1{max-width:92%;margin:0;color:${ink};font-size:${Math.round(width * .059)}px;line-height:1.02;letter-spacing:-.055em;font-weight:650}
    .copy p{max-width:82%;margin:1.6% 0 0;color:${ink}a8;font-size:${Math.round(width * .022)}px;line-height:1.35}
    .stage{position:absolute;inset:17.1% 8.5% 3.8%;display:grid;place-items:center}
    .device{position:relative;height:100%;aspect-ratio:412/915;padding:2.5%;border-radius:${Math.round(width * .074)}px;background:linear-gradient(145deg,#38454b,#071319 46%,#223239);box-shadow:0 ${Math.round(height * .025)}px ${Math.round(height * .06)}px #071b2450,0 0 0 1px #ffffff80 inset}
    .device:before{content:"";position:absolute;inset:.65%;border:1px solid #8aa0a853;border-radius:inherit;pointer-events:none}
    .screen{width:100%;height:100%;display:block;object-fit:cover;object-position:top;border-radius:${Math.round(width * .056)}px;background:#f7f2ea;box-shadow:0 0 0 2px #000 inset}
    .camera{position:absolute;z-index:5;top:1.8%;left:50%;width:24%;height:1.8%;transform:translateX(-50%);border-radius:99px;background:#05090b;box-shadow:0 1px 0 #54636a55}
  </style></head><body>
    <section class="copy"><small>${escapeHtml(copy[0])}</small><h1>${escapeHtml(copy[1])}</h1><p>${escapeHtml(copy[2])}</p></section>
    <div class="stage"><div class="device"><i class="camera"></i><img class="screen" src="${sourceUrl}"></div></div>
  </body></html>`);
  await page.screenshot({ path: target, type: 'png' });
  await context.close();
}

async function renderWebAsset(browser, source, target, copy, palette) {
  const [, ink, accent] = palette;
  const sourceUrl = await dataUrl(source);
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1, colorScheme: 'light' });
  const page = await context.newPage();
  await page.setContent(`<!doctype html><html><head><meta charset="utf-8"><style>
    *{box-sizing:border-box}html,body{margin:0;width:1920px;height:1080px;overflow:hidden}
    body{font-family:Arial,Helvetica,sans-serif;color:${ink};background:#fff}
    .copy{position:absolute;top:54px;left:112px;right:112px;display:grid;grid-template-columns:1fr .8fr;align-items:end;gap:80px}
    .copy small{display:block;margin-bottom:12px;color:${accent};font-size:15px;font-weight:800;letter-spacing:.17em}
    .copy h1{max-width:1000px;margin:0;font-size:61px;line-height:.98;letter-spacing:-.055em;font-weight:650}
    .copy p{justify-self:end;max-width:520px;margin:0 0 5px;color:${ink}a6;font-size:20px;line-height:1.45;text-align:right}
    .browser{position:absolute;left:92px;right:92px;top:224px;bottom:58px;overflow:hidden;border:1px solid #82979c70;border-radius:23px;background:#f8f4ed;box-shadow:0 34px 85px #0e2d3840,0 1px 0 #fff inset}
    .chrome{height:48px;display:flex;align-items:center;gap:9px;padding:0 18px;border-bottom:1px solid #cad3d1;background:#fbf8f2}
    .chrome i{width:11px;height:11px;border-radius:50%;background:#c3ccc9}.chrome i:first-child{background:${accent}}
    .url{width:42%;height:24px;margin-left:10px;border:1px solid #d6ddda;border-radius:6px;background:#eef2ef;color:#799096;font-size:10px;line-height:22px;text-align:center;letter-spacing:.04em}
    .screen{width:100%;height:calc(100% - 48px);display:block;object-fit:cover;object-position:top;background:#f8f4ed}
  </style></head><body>
    <section class="copy"><div><small>${escapeHtml(copy[0])}</small><h1>${escapeHtml(copy[1])}</h1></div><p>${escapeHtml(copy[2])}</p></section>
    <div class="browser"><div class="chrome"><i></i><i></i><i></i><span class="url">studio.dajoongbim.com</span></div><img class="screen" src="${sourceUrl}"></div>
  </body></html>`);
  await page.screenshot({ path: target, type: 'png' });
  await context.close();
}

async function renderContactSheet(browser, paths, target, title, portrait) {
  const width = 2400;
  const height = portrait ? 1120 : 2140;
  const cards = await Promise.all(paths.map(async (path, index) => `<figure><img src="${await dataUrl(path)}"><figcaption>${String(index + 1).padStart(2, '0')}</figcaption></figure>`));
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  await page.setContent(`<!doctype html><style>
    *{box-sizing:border-box}html,body{margin:0;width:${width}px;height:${height}px;overflow:hidden}body{padding:68px;background:#0e2d38;color:#f7f3ec;font-family:Arial,sans-serif}
    h1{margin:0 0 42px;font-size:42px;letter-spacing:-.04em}.grid{display:grid;grid-template-columns:${portrait ? 'repeat(5,1fr)' : 'repeat(2,1fr)'};gap:${portrait ? '28px' : '32px'};align-items:start}
    figure{position:relative;margin:0;overflow:hidden;border:1px solid #718990;border-radius:18px;background:#f7f1e8;box-shadow:0 18px 45px #020b0f66}img{width:100%;display:block}figcaption{position:absolute;right:14px;bottom:12px;padding:8px 10px;border-radius:8px;background:#0d2630d9;font-size:15px;font-weight:700}
    ${portrait ? 'figure{height:800px}img{height:100%;object-fit:contain}' : 'figure{height:596px}img{height:100%;object-fit:cover}'}
  </style><h1>${escapeHtml(title)}</h1><div class="grid">${cards.join('')}</div>`);
  await page.screenshot({ path: target, type: 'jpeg', quality: 92 });
  await context.close();
}

const browser = await chromium.launch();
try {
  await captureRawSet(browser, { width: 412, height: 915 }, rawMobile, 'mobile');
  await captureRawSet(browser, { width: 1600, height: 760 }, rawWeb, 'web');

  for (const [index, story] of stories.entries()) {
    const palette = palettes[index];
    const mobileSource = join(rawMobile, `${story.id}.png`);
    const webSource = join(rawWeb, `${story.id}.png`);
    for (const locale of ['en-US', 'ko-KR']) {
      if (story.action !== 'quality') {
        const playTarget = join(androidRoot, locale, 'images', 'phoneScreenshots', `${appAssetId(story)}.png`);
        await renderPhoneAsset(browser, mobileSource, playTarget, [1080, 1920], story.copy[locale], palette);
      }
      const webTarget = join(webRoot, locale, `${story.id}.png`);
      await renderWebAsset(browser, webSource, webTarget, story.copy[locale], palette);
    }
    if (story.action !== 'quality') {
      await renderPhoneAsset(browser, mobileSource, join(iosRoot, 'en-US', `${appAssetId(story)}.png`), [1320, 2868], story.copy['en-US'], palette);
      await renderPhoneAsset(browser, mobileSource, join(iosRoot, 'ko', `${appAssetId(story)}.png`), [1320, 2868], story.copy['ko-KR'], palette);
    }
  }

  for (const locale of ['en-US', 'ko-KR']) {
    const imageRoot = join(androidRoot, locale, 'images');
    await writeFile(join(imageRoot, 'alt-text.json'), `${JSON.stringify(Object.fromEntries(appStories.map((story) => [`${appAssetId(story)}.png`, story.alt[locale]])), null, 2)}\n`);
  }

  const logo = await readFile(join(root, 'public', 'brand', 'dajoong-logo-mark-512.png'));
  for (const locale of ['en-US', 'ko-KR']) {
    const imageRoot = join(androidRoot, locale, 'images');
    const page = await browser.newPage({ viewport: { width: 1024, height: 500 }, deviceScaleFactor: 1 });
    const localized = locale === 'ko-KR'
      ? ['도면을 편집 가능한 BIM으로.', '원본과 연결된 2D·3D 모델']
      : ['Drawings to editable BIM.', 'Source-linked 2D and 3D models'];
    await page.setContent(`<!doctype html><style>*{box-sizing:border-box}body{margin:0;width:1024px;height:500px;display:flex;align-items:center;padding:72px;background:linear-gradient(128deg,#f8f3ea 0%,#eef4f1 58%,#dcebe5 100%);font-family:Arial,sans-serif;color:#123044}.mark{width:260px;height:260px;object-fit:contain;filter:drop-shadow(0 20px 30px rgba(18,48,68,.16))}.copy{margin-left:66px}.copy small{font-size:21px;letter-spacing:.16em;color:#557168}.copy h1{margin:14px 0 12px;font-size:54px;line-height:1.04;letter-spacing:-.05em}.copy p{margin:0;max-width:520px;font-size:24px;line-height:1.35;color:#45616b}</style><img class="mark" src="data:image/png;base64,${logo.toString('base64')}"><div class="copy"><small>DAJOONG PLAN2BIM</small><h1>${localized[0]}</h1><p>${localized[1]}</p></div>`);
    await page.screenshot({ path: join(imageRoot, 'featureGraphic.png') });
    await page.close();
    await cp(join(root, 'public', 'brand', 'dajoong-logo-mark-512.png'), join(imageRoot, 'icon.png'));
  }

  await renderContactSheet(
    browser,
    appStories.map((story) => join(androidRoot, 'en-US', 'images', 'phoneScreenshots', `${appAssetId(story)}.png`)),
    join(artifactRoot, 'dajoong-play-store-contact-sheet.jpg'),
    'DAJOONG · GOOGLE PLAY PHONE SCREENS',
    true,
  );
  await renderContactSheet(
    browser,
    stories.map((story) => join(webRoot, 'en-US', `${story.id}.png`)),
    join(artifactRoot, 'dajoong-web-contact-sheet.jpg'),
    'DAJOONG · WEB PRODUCT SCREENS',
    false,
  );
} finally {
  await browser.close();
  server.kill();
}

console.log(`Created ${appStories.length} framed phone screens per locale and ${stories.length} web showcase screens.`);
