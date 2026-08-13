import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";

const graphPath = new URL("../public/sample/03-plan-graph.json", import.meta.url);
const manifestPath = new URL("../public/sample/sample-manifest.json", import.meta.url);

const graph = JSON.parse(await readFile(graphPath, "utf8"));
graph.provenance ??= {};

if (graph.provenance.fixture_axis_contract !== "local_size_plus_yaw_v1") {
  for (const fixture of graph.fixtures ?? []) {
    const yaw = Math.abs(Math.round(Number(fixture.yaw_deg ?? 0) / 90)) % 2;
    const generatedFromAxisAlignedEvidence = /^svg-fixture-/.test(
      String(fixture.source_entity_id ?? ""),
    );
    if (!generatedFromAxisAlignedEvidence || yaw !== 1) continue;
    for (const property of ["size_m", "geometry_scale_xyz"]) {
      const dimensions = fixture[property];
      if (!Array.isArray(dimensions) || dimensions.length < 3) continue;
      [dimensions[0], dimensions[1]] = [dimensions[1], dimensions[0]];
    }
  }
  graph.provenance.fixture_axis_contract = "local_size_plus_yaw_v1";
}

const graphText = `${JSON.stringify(graph, null, 2)}\n`;
await writeFile(graphPath, graphText, "utf8");

const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
manifest.graph.sha256 = createHash("sha256").update(graphText).digest("hex");
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

console.log(JSON.stringify({
  fixtureAxisContract: graph.provenance.fixture_axis_contract,
  graphSha256: manifest.graph.sha256,
  fixtures: graph.fixtures?.length ?? 0,
}));
