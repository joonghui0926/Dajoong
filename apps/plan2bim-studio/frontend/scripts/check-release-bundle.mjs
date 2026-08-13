import { readFile, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const dist = join(root, 'dist');
const files = [];
async function walk(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) await walk(path);
    else files.push(path);
  }
}
await walk(dist);

const protectedExtensions = /\.(?:onnx|pt|pth|ckpt|safetensors|pem|p12|jks|keystore)$/i;
const protectedFiles = files.filter((path) => protectedExtensions.test(path));
if (protectedFiles.length) throw new Error(`Protected artifacts entered the client bundle: ${protectedFiles.join(', ')}`);

const textFiles = files.filter((path) => /\.(?:js|mjs|html|json|css)$/i.test(path));
const bundleText = (await Promise.all(textFiles.map((path) => readFile(path, 'utf8')))).join('\n');
const forbiddenRuntimeMarkers = [
  'onnxruntime',
  'api.openai.com',
  'api.anthropic.com',
  'generativelanguage.googleapis.com',
  'api.replicate.com',
  'api-inference.huggingface.co',
];
for (const marker of forbiddenRuntimeMarkers) {
  if (bundleText.toLowerCase().includes(marker.toLowerCase())) {
    throw new Error(`Client bundle contains a forbidden inference runtime or endpoint: ${marker}`);
  }
}
if (!bundleText.includes('studio-api.dajoongbim.com')) {
  throw new Error('Client bundle does not contain the canonical Dajoong conversion API boundary');
}
if (/\b(?:AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,})\b/.test(bundleText)) {
  throw new Error('Client bundle appears to contain a server credential');
}

const scripts = await Promise.all(files.filter((path) => path.endsWith('.js')).map(async (path) => ({ path, bytes: (await stat(path)).size })));
const totalBytes = scripts.reduce((sum, file) => sum + file.bytes, 0);
const largestBytes = Math.max(0, ...scripts.map((file) => file.bytes));
const distBytes = (await Promise.all(files.map(async (path) => (await stat(path)).size))).reduce((sum, bytes) => sum + bytes, 0);
const indexHtml = await readFile(join(dist, 'index.html'), 'utf8');
const entryMatch = indexHtml.match(/<script[^>]+type="module"[^>]+src="([^"]+\.js)"/i);
if (!entryMatch) throw new Error('Could not identify the initial JavaScript entry in dist/index.html.');
const entryPath = join(dist, entryMatch[1].replace(/^\/+/, ''));
const entryBytes = (await stat(entryPath)).size;
if (largestBytes > 600_000) throw new Error(`Largest JavaScript chunk is ${largestBytes} bytes; budget is 600000.`);
if (totalBytes > 2_500_000) throw new Error(`Total JavaScript is ${totalBytes} bytes; budget is 2500000.`);
if (entryBytes > 250_000) throw new Error(`Initial JavaScript entry is ${entryBytes} bytes; budget is 250000.`);
if (distBytes > 4_000_000) throw new Error(`Production bundle is ${distBytes} bytes; budget is 4000000.`);
console.log(JSON.stringify({
  schema: 'dajoong.release-budget.v3',
  serverOnly: true,
  canonicalApi: 'https://studio-api.dajoongbim.com',
  scripts: scripts.length,
  entryBytes,
  largestBytes,
  totalBytes,
  distBytes,
}, null, 2));
