import { readFile, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const read = (path) => readFile(join(root, path), 'utf8');
const assert = (condition, message) => { if (!condition) throw new Error(message); };

const declaration = JSON.parse(await read('store/release-declaration.json'));
assert(declaration.app.bundleId === 'com.dajoong.plan2bim', 'Store bundle ID drifted');
assert(declaration.app.supportEmail === 'jjoonghui@gmail.com', 'Store support email drifted');
const requiredPublicUrls = {
  supportUrl: 'https://studio.dajoongbim.com/support',
  privacyUrl: 'https://studio.dajoongbim.com/privacy',
  accountDeletionUrl: 'https://studio.dajoongbim.com/account-deletion',
};
for (const [name, expected] of Object.entries(requiredPublicUrls)) {
  assert(declaration.app[name] === expected, `${name} must stay on the deployed Dajoong .com domain`);
}
assert(declaration.privacy.tracking === false && declaration.privacy.sold === false, 'Privacy declaration must fail closed');

const limits = [
  ['store/app-store/metadata/en-US/name.txt', 30],
  ['store/app-store/metadata/en-US/subtitle.txt', 30],
  ['store/app-store/metadata/en-US/keywords.txt', 100],
  ['store/app-store/metadata/ko/name.txt', 30],
  ['store/app-store/metadata/ko/subtitle.txt', 30],
  ['store/app-store/metadata/ko/keywords.txt', 100],
  ['store/google-play/metadata/en-US/title.txt', 30],
  ['store/google-play/metadata/en-US/short_description.txt', 80],
  ['store/google-play/metadata/en-US/full_description.txt', 4000],
  ['store/google-play/metadata/ko-KR/title.txt', 30],
  ['store/google-play/metadata/ko-KR/short_description.txt', 80],
  ['store/google-play/metadata/ko-KR/full_description.txt', 4000],
];
for (const [path, limit] of limits) {
  const value = (await read(path)).trim();
  assert(value.length > 0 && value.length <= limit, `${path} must contain 1-${limit} characters (found ${value.length})`);
}

for (const locale of ['en-US', 'ko']) {
  assert((await read(`store/app-store/metadata/${locale}/privacy_url.txt`)).trim() === declaration.app.privacyUrl, `App Store ${locale} privacy URL drifted`);
  assert((await read(`store/app-store/metadata/${locale}/support_url.txt`)).trim() === declaration.app.supportUrl, `App Store ${locale} support URL drifted`);
}

function pngDimensions(buffer) {
  assert(buffer.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])), 'Invalid PNG signature');
  return [buffer.readUInt32BE(16), buffer.readUInt32BE(20)];
}
async function validateScreenshots(directory, expected) {
  const names = (await readdir(join(root, directory))).filter((name) => name.endsWith('.png')).sort();
  assert(names.length >= 2, `${directory} requires at least two screenshots`);
  for (const name of names) {
    const dimensions = pngDimensions(await readFile(join(root, directory, name)));
    assert(dimensions[0] === expected[0] && dimensions[1] === expected[1], `${directory}/${name} has invalid dimensions ${dimensions.join('x')}`);
  }
}
await validateScreenshots('store/app-store/screenshots/en-US', [1320, 2868]);
await validateScreenshots('store/app-store/screenshots/ko', [1320, 2868]);
await validateScreenshots('store/google-play/metadata/en-US/images/phoneScreenshots', [1080, 1920]);
await validateScreenshots('store/google-play/metadata/ko-KR/images/phoneScreenshots', [1080, 1920]);

for (const locale of ['en-US', 'ko-KR']) {
  const imageRoot = `store/google-play/metadata/${locale}/images`;
  assert(pngDimensions(await readFile(join(root, imageRoot, 'featureGraphic.png'))).join('x') === '1024x500', `${locale} feature graphic must be 1024x500`);
  assert(pngDimensions(await readFile(join(root, imageRoot, 'icon.png'))).join('x') === '512x512', `${locale} icon must be 512x512`);
}

const androidVariables = await read('android/variables.gradle');
assert(/targetSdkVersion\s*=\s*36\b/.test(androidVariables), 'Android targetSdkVersion must be 36');
const manifest = await read('android/app/src/main/AndroidManifest.xml');
assert(manifest.includes('android:usesCleartextTraffic="false"'), 'Android cleartext traffic must stay disabled');
assert(manifest.includes('android:allowBackup="false"'), 'Android backup must stay disabled');
assert(manifest.includes('android:autoVerify="true"'), 'Android App Links must stay verified');
assert(manifest.includes('android:host="app.dajoongbim.com"'), 'Android app link host drifted');
assert(manifest.includes('android:host="studio.dajoongbim.com"'), 'Android Studio link host drifted');
assert(manifest.includes('android:scheme="com.dajoong.plan2bim"'), 'Android OAuth callback scheme drifted');
const privacyManifest = await read('ios/App/App/PrivacyInfo.xcprivacy');
assert(privacyManifest.includes('<key>NSPrivacyTracking</key>') && privacyManifest.includes('<false/>'), 'iOS privacy manifest must declare no tracking');
assert(privacyManifest.includes('<key>NSPrivacyAccessedAPITypes</key>'), 'iOS privacy manifest must explicitly declare required-reason API use');
const entitlements = await read('ios/App/App/App.entitlements');
assert(entitlements.includes('com.apple.developer.applesignin') && entitlements.includes('<string>Default</string>'), 'iOS Sign in with Apple entitlement is required');
assert(entitlements.includes('applinks:app.dajoongbim.com') && entitlements.includes('applinks:studio.dajoongbim.com'), 'iOS associated domains drifted');
const pbx = await read('ios/App/App.xcodeproj/project.pbxproj');
assert(pbx.includes('PRODUCT_BUNDLE_IDENTIFIER = com.dajoong.plan2bim;'), 'iOS bundle ID drifted');
assert(pbx.includes('com.apple.SignInWithApple') && pbx.includes('com.apple.AssociatedDomains'), 'iOS target capabilities drifted');

const webWorkerConfig = await read('../infra/cloudflare/wrangler.web.jsonc');
for (const host of ['dajoongbim.com', 'www.dajoongbim.com', 'studio.dajoongbim.com', 'app.dajoongbim.com']) {
  assert(webWorkerConfig.includes(`"pattern": "${host}"`), `Cloudflare custom domain missing: ${host}`);
}
const webWorker = await read('../infra/cloudflare/studio-web-worker.ts');
assert(webWorker.includes("const MARKETING_HOST = 'dajoongbim.com'"), 'Marketing apex host guard drifted');
assert(webWorker.includes("const STUDIO_HOST = 'studio.dajoongbim.com'"), 'Studio host guard drifted');
assert(webWorker.includes('ASSOCIATION_PATHS'), 'Native association files must bypass hostname redirects');

const protectedPattern = /\.(?:onnx|pt|pth|ckpt|safetensors|pem|p12|jks|keystore)$/i;
for (const directory of ['dist', 'android/app/src/main/assets/public']) {
  try {
    const stack = [join(root, directory)];
    while (stack.length) {
      const current = stack.pop();
      for (const entry of await readdir(current, { withFileTypes: true })) {
        const path = join(current, entry.name);
        if (entry.isDirectory()) stack.push(path);
        else assert(!protectedPattern.test(path), `Protected model or signing artifact entered a client: ${path}`);
      }
    }
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
}

const bytes = (await stat(join(root, 'store/release-declaration.json'))).size;
console.log(JSON.stringify({ schema: 'dajoong.store-validation.v1', metadataChecks: limits.length, declarationBytes: bytes, status: 'ready' }, null, 2));
