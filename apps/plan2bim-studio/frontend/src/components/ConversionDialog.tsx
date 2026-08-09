import { Building2, FileImage, LoaderCircle, Plus, Trash2, X } from "lucide-react";
import { useState } from "react";
import type { FormEvent } from "react";

import type { PlanGraph } from "../types";
import { authFetch } from "../auth";
import { studioApiUrl } from "../serverApi";

interface BuildingLevelDraft {
  id: string;
  name: string;
  page: string;
  elevation: string;
  pixelsPerMeter: string;
}

interface ConversionDialogProps {
  open: boolean;
  onClose: () => void;
  onStatus: (message: string) => void;
  onComplete: (graph: PlanGraph, sourceUrl: string, jobId: string) => void;
}

export function ConversionDialog({ open, onClose, onStatus, onComplete }: ConversionDialogProps) {
  const [file, setFile] = useState<File | null>(null);
  const [projectId, setProjectId] = useState("dajoong-project");
  const [levelId, setLevelId] = useState("L1");
  const [levelName, setLevelName] = useState("Level 1");
  const [pixelsPerMeter, setPixelsPerMeter] = useState("100");
  const [height, setHeight] = useState("3.0");
  const [thickness, setThickness] = useState("0.12");
  const [pageNumber, setPageNumber] = useState("1");
  const [pdfDpi, setPdfDpi] = useState("300");
  const [buildingMode, setBuildingMode] = useState(false);
  const [buildingLevels, setBuildingLevels] = useState<BuildingLevelDraft[]>([
    { id: "L1", name: "Ground floor", page: "1", elevation: "0", pixelsPerMeter: "100" },
    { id: "L2", name: "Second floor", page: "2", elevation: "3", pixelsPerMeter: "100" },
  ]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  if (!open) return null;

  const isPdf = file?.type === "application/pdf" || file?.name.toLowerCase().endsWith(".pdf");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) return setError("Choose a drawing or PDF first.");
    if (buildingMode && !isPdf) return setError("Building set mode requires a multi-page PDF.");
    setSubmitting(true);
    setError("");
    try {
      const body = new FormData();
      body.set("drawing", file);
      let endpoint = "/api/jobs";
      if (buildingMode) {
        endpoint = "/api/building-jobs";
        body.set(
          "building_config",
          JSON.stringify({
            project_id: projectId,
            pdf_dpi: Number(pdfDpi),
            levels: buildingLevels.map((level) => ({
              source_path: file.name,
              page_number: Number(level.page),
              level_id: level.id,
              name: level.name,
              elevation_m: Number(level.elevation),
              nominal_height_m: Number(height),
              pixels_per_meter: Number(level.pixelsPerMeter),
              wall_thickness_m: Number(thickness),
            })),
            vertical_connections: [],
          }),
        );
      } else {
        body.set("pixels_per_meter", pixelsPerMeter);
        body.set("project_id", projectId);
        body.set("level_id", levelId);
        body.set("level_name", levelName);
        body.set("nominal_height_m", height);
        body.set("wall_thickness_m", thickness);
        body.set("page_number", pageNumber);
        body.set("pdf_dpi", pdfDpi);
      }
      const response = await authFetch(studioApiUrl(endpoint), { method: "POST", body });
      if (!response.ok) throw new Error(await response.text());
      const job = (await response.json()) as { id: string; status: string; error?: string };
      onStatus(`Conversion ${job.id.slice(0, 8)} queued`);
      const finished = await pollJob(job.id, (status) =>
        onStatus(`Conversion ${status.replaceAll("_", " ")}`),
      );
      if (finished.status === "failed") throw new Error(finished.error || "conversion failed");
      const [graphResponse, renderResponse] = await Promise.all([
        authFetch(studioApiUrl(`/api/jobs/${job.id}/artifacts/graph`)),
        authFetch(studioApiUrl(`/api/jobs/${job.id}/artifacts/render`)),
      ]);
      if (!graphResponse.ok) throw new Error("conversion completed without a PlanGraph");
      const graph = (await graphResponse.json()) as PlanGraph;
      const sourceUrl = renderResponse.ok ? URL.createObjectURL(await renderResponse.blob()) : "";
      onComplete(graph, sourceUrl, job.id);
      onStatus(`Conversion ready · ${graph.walls.length} walls · ${graph.rooms.length} rooms`);
      onClose();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not start conversion");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => event.target === event.currentTarget && !submitting && onClose()}
    >
      <form className="conversion-dialog" onSubmit={submit}>
        <div className="dialog-header">
          <div><span className="eyebrow">NEW CONVERSION</span><h2>{buildingMode ? "Convert a building set" : "Convert a floor plan"}</h2></div>
          <button type="button" onClick={onClose} disabled={submitting} aria-label="Close conversion dialog"><X size={18} /></button>
        </div>
        <label className={file ? "drawing-drop selected" : "drawing-drop"}>
          <FileImage size={24} />
          <strong>{file ? file.name : "Choose a drawing or PDF"}</strong>
          <span>PDF, PNG, JPEG, or TIFF · original sheets work best</span>
          <input
            type="file"
            accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,application/pdf,image/*"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
        </label>
        <div className="conversion-mode" role="group" aria-label="Conversion scope">
          <button type="button" className={!buildingMode ? "active" : ""} onClick={() => setBuildingMode(false)}><FileImage size={15} /> One level</button>
          <button type="button" className={buildingMode ? "active" : ""} onClick={() => setBuildingMode(true)}><Building2 size={15} /> Building set</button>
        </div>
        <div className="dialog-grid">
          <DialogField label="Project ID" value={projectId} onChange={setProjectId} />
          {!buildingMode ? <DialogField label="Level ID" value={levelId} onChange={setLevelId} /> : <div />}
          {!buildingMode ? <DialogField label="Level name" value={levelName} onChange={setLevelName} /> : null}
          {!buildingMode ? <DialogField label="Pixels per meter" value={pixelsPerMeter} onChange={setPixelsPerMeter} type="number" step="0.01" /> : null}
          <DialogField label="Wall height (m)" value={height} onChange={setHeight} type="number" step="0.05" />
          <DialogField label="Wall thickness (m)" value={thickness} onChange={setThickness} type="number" step="0.01" />
          {isPdf && !buildingMode ? <DialogField label="PDF page" value={pageNumber} onChange={setPageNumber} type="number" step="1" min="1" /> : null}
          {isPdf ? <DialogField label="PDF render DPI" value={pdfDpi} onChange={setPdfDpi} type="number" step="1" min="72" /> : null}
        </div>
        {buildingMode ? (
          <BuildingLevelEditor levels={buildingLevels} onChange={setBuildingLevels} />
        ) : null}
        <p className="scale-help">{buildingMode ? "Each page keeps its own scale and elevation. Vertical connections stay empty until confirmed, so the converter does not invent stairs." : "Use a known dimension or sheet metadata for scale. Physical size is kept explicit and auditable."}</p>
        {error ? <div className="dialog-error">{error}</div> : null}
        <div className="dialog-actions">
          <button type="button" className="secondary-button" onClick={onClose} disabled={submitting}>Cancel</button>
          <button type="submit" className="primary-button" disabled={submitting || !file}>
            {submitting ? <LoaderCircle className="spin" size={16} /> : null}
            {submitting ? "Converting…" : "Start conversion"}
          </button>
        </div>
      </form>
    </div>
  );
}

function BuildingLevelEditor({
  levels,
  onChange,
}: {
  levels: BuildingLevelDraft[];
  onChange: (levels: BuildingLevelDraft[]) => void;
}) {
  const update = (index: number, changes: Partial<BuildingLevelDraft>) =>
    onChange(levels.map((level, itemIndex) => itemIndex === index ? { ...level, ...changes } : level));
  return (
    <section className="building-level-editor">
      <div className="building-level-heading"><div><span>PDF LEVEL MAP</span><b>{levels.length} levels</b></div><button type="button" onClick={() => onChange([...levels, { id: `L${levels.length + 1}`, name: `Level ${levels.length + 1}`, page: String(levels.length + 1), elevation: String(levels.length * 3), pixelsPerMeter: levels.at(-1)?.pixelsPerMeter ?? "100" }])}><Plus size={14} /> Add level</button></div>
      <div className="building-level-labels"><span>ID</span><span>Name</span><span>Page</span><span>Elevation</span><span>Scale</span><i /></div>
      {levels.map((level, index) => (
        <div className="building-level-row" key={`${level.id}-${index}`}>
          <input required aria-label={`Level ${index + 1} ID`} value={level.id} onChange={(event) => update(index, { id: event.target.value })} />
          <input required aria-label={`Level ${index + 1} name`} value={level.name} onChange={(event) => update(index, { name: event.target.value })} />
          <input required aria-label={`Level ${index + 1} PDF page`} type="number" min="1" step="1" value={level.page} onChange={(event) => update(index, { page: event.target.value })} />
          <input required aria-label={`Level ${index + 1} elevation`} type="number" step="0.05" value={level.elevation} onChange={(event) => update(index, { elevation: event.target.value })} />
          <input required aria-label={`Level ${index + 1} scale`} type="number" min="0.001" step="0.01" value={level.pixelsPerMeter} onChange={(event) => update(index, { pixelsPerMeter: event.target.value })} />
          <button type="button" aria-label={`Remove ${level.name}`} disabled={levels.length <= 1} onClick={() => onChange(levels.filter((_, itemIndex) => itemIndex !== index))}><Trash2 size={14} /></button>
        </div>
      ))}
    </section>
  );
}

function DialogField({
  label,
  value,
  onChange,
  type = "text",
  step,
  min,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: string;
  step?: string;
  min?: string;
}) {
  return (
    <label className="dialog-field">
      <span>{label}</span>
      <input
        required
        type={type}
        step={step}
        min={min ?? (type === "number" ? "0.001" : undefined)}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

async function pollJob(jobId: string, onStatus: (status: string) => void) {
  const deadline = Date.now() + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, 700));
    const response = await authFetch(studioApiUrl(`/api/jobs/${jobId}`));
    if (!response.ok) throw new Error("could not read conversion status");
    const job = (await response.json()) as { status: string; error?: string };
    onStatus(job.status);
    if (["complete", "review_required", "failed"].includes(job.status)) return job;
  }
  throw new Error("conversion timed out after 10 minutes");
}
