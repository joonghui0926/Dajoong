import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
const output = join(root, 'public', '.well-known');
const teamId = (process.env.APPLE_TEAM_ID || '').trim();
const fingerprint = (process.env.ANDROID_APP_LINK_SHA256 || '').trim().toUpperCase();
const bundleId = 'com.dajoong.plan2bim';

if (!/^[A-Z0-9]{10}$/.test(teamId)) {
  throw new Error('APPLE_TEAM_ID must contain the 10-character Apple team identifier');
}
if (!/^(?:[0-9A-F]{2}:){31}[0-9A-F]{2}$/.test(fingerprint)) {
  throw new Error('ANDROID_APP_LINK_SHA256 must contain the 32-byte colon-delimited signing fingerprint');
}

await mkdir(output, { recursive: true });
await writeFile(join(output, 'apple-app-site-association'), `${JSON.stringify({
  applinks: {
    apps: [],
    details: [{
      appIDs: [`${teamId}.${bundleId}`],
      components: [{ '/': '/studio*', comment: 'Open the Dajoong Studio application' }],
    }],
  },
}, null, 2)}\n`);
await writeFile(join(output, 'assetlinks.json'), `${JSON.stringify([{
  relation: ['delegate_permission/common.handle_all_urls'],
  target: {
    namespace: 'android_app',
    package_name: bundleId,
    sha256_cert_fingerprints: [fingerprint],
  },
}], null, 2)}\n`);
console.log('Prepared verified iOS and Android association files for app.builiconstruction.com and studio.builiconstruction.com');
