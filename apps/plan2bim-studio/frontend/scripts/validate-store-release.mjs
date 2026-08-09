import { readFile, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const read = (path) => readFile(join(root, path), 'utf8');
const assert = (condition, message) => { if (!condition) throw new Error(message); };

const declaration = JSON.parse(await read('store/release-declaration.json'));
assert(declaration.app.bundleId === 'com.dajoong.plan2bim', 'Store bundle ID drifted');
assert(declaration.app.supportEmail === 'jjoonghui@gmail.com', 'Store support email drifted');
for (const url of ['supportUrl', 'privacyUrl', 'accountDeletionUrl']) {
  assert(new URL(declaration.app[url]).protocol === 'https:', `${url} must use HTTPS`);
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
const privacyManifest = await read('ios/App/App/PrivacyInfo.xcprivacy');
assert(privacyManifest.includes('<key>NSPrivacyTracking</key>') && privacyManifest.includes('<false/>'), 'iOS privacy manifest must declare no tracking');
const pbx = await read('ios/App/App.xcodeproj/project.pbxproj');
assert(pbx.includes('PRODUCT_BUNDLE_IDENTIFIER = com.dajoong.plan2bim;'), 'iOS bundle ID drifted');

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
