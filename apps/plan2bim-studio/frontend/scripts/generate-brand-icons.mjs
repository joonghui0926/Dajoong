import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "@playwright/test";

const root = fileURLToPath(new URL("..", import.meta.url));
const publicRoot = join(root, "public");
const svgPath = join(publicRoot, "brand", "dajoong-logo-mark.svg");
const svg = await readFile(svgPath, "utf8");
const svgUrl = `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`;

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

async function render(size, mimeType = "image/png") {
  const dataUrl = await page.evaluate(
    async ({ source, size: outputSize, mimeType: outputType }) => {
      const image = new Image();
      image.src = source;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = outputSize;
      canvas.height = outputSize;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("2D canvas is unavailable");
      context.clearRect(0, 0, outputSize, outputSize);
      const height = outputSize * 0.94;
      const width = height * (580 / 651);
      context.drawImage(image, (outputSize - width) / 2, (outputSize - height) / 2, width, height);
      return canvas.toDataURL(outputType, 1);
    },
    { source: svgUrl, size, mimeType },
  );
  return Buffer.from(dataUrl.slice(dataUrl.indexOf(",") + 1), "base64");
}

function makeIco(entries) {
  const directoryBytes = 6 + entries.length * 16;
  const header = Buffer.alloc(directoryBytes);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(entries.length, 4);
  let offset = directoryBytes;
  entries.forEach(({ size, png }, index) => {
    const cursor = 6 + index * 16;
    header.writeUInt8(size === 256 ? 0 : size, cursor);
    header.writeUInt8(size === 256 ? 0 : size, cursor + 1);
    header.writeUInt8(0, cursor + 2);
    header.writeUInt8(0, cursor + 3);
    header.writeUInt16LE(1, cursor + 4);
    header.writeUInt16LE(32, cursor + 6);
    header.writeUInt32LE(png.length, cursor + 8);
    header.writeUInt32LE(offset, cursor + 12);
    offset += png.length;
  });
  return Buffer.concat([header, ...entries.map(({ png }) => png)]);
}

const pngs = new Map();
for (const size of [16, 32, 48, 64, 180, 192, 256, 512]) {
  pngs.set(size, await render(size));
}

const assets = [
  ["pwa-64x64.png", pngs.get(64)],
  ["pwa-192x192.png", pngs.get(192)],
  ["pwa-512x512.png", pngs.get(512)],
  ["apple-touch-icon-180x180.png", pngs.get(180)],
  ["brand/dajoong-logo-mark-512.png", pngs.get(512)],
  ["brand/dajoong-logo-mark-512.webp", await render(512, "image/webp")],
  ["favicon.ico", makeIco([16, 32, 48, 256].map((size) => ({ size, png: pngs.get(size) })))],
];

for (const [relativePath, bytes] of assets) {
  const outputPath = join(publicRoot, relativePath);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, bytes);
}

await browser.close();
console.log(JSON.stringify({ source: svgPath, generated: assets.map(([path]) => path) }, null, 2));
