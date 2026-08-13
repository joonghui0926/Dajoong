import { Check, ClipboardCheck, Gauge, History, Layers3, Lock, MousePointer2, ScanSearch, ShieldAlert, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

import { confidenceLabel } from "../graph";
import { distanceMeters, setSegmentLength, setWallLength } from "../editorGeometry";
import type { ReviewPriority } from "../reviewPlanner";
import type { BaseEntity, CollectionName, Selection, WallEntity } from "../types";

interface PropertyPanelProps {
  entity: BaseEntity | null;
  selection: Selection | null;
  entities: BaseEntity[];
  selections: Selection[];
  changeCount: number;
  reviewPriority: ReviewPriority | null;
  locked: boolean;
  onEdit: (changes: Record<string, unknown>) => void;
  onBatchEdit: (changes: Record<string, unknown>) => void;
  onAccept: () => void;
  onBatchAccept: () => void;
  onDelete: () => void;
  onBatchDelete: () => void;
  onBrowseFamily: () => void;
  onRequestOpeningRehost: () => void;
}

export function PropertyPanel({
  entity,
  selection,
  entities,
  selections,
  changeCount,
  reviewPriority,
  locked,
  onEdit,
  onBatchEdit,
  onAccept,
  onBatchAccept,
  onDelete,
  onBatchDelete,
  onBrowseFamily,
  onRequestOpeningRehost,
}: PropertyPanelProps) {
  if (selections.length > 1) {
    return (
      <BatchPropertyPanel
        entities={entities}
        selections={selections}
        locked={locked}
        onEdit={onBatchEdit}
        onAccept={onBatchAccept}
        onDelete={onBatchDelete}
        onBrowseFamily={onBrowseFamily}
      />
    );
  }
  if (!entity || !selection) {
    return (
      <aside className="property-panel panel empty-properties">
        <ClipboardCheck size={28} />
        <h2>Select an element</h2>
        <p>Selection stays linked between the plan, the 3D view, and this property palette.</p>
      </aside>
    );
  }
  return (
    <aside className={locked ? "property-panel panel is-locked" : "property-panel panel"}>
      <div className="panel-title-row">
        <div>
          <span className="eyebrow">PROPERTIES</span>
          <h2>{entityTitle(selection.collection, entity)}</h2>
        </div>
        <span className={`review-badge ${locked ? "locked" : entity.review_state === "accepted" ? "accepted" : "review"}`}>
          {locked ? <><Lock size={11} /> Locked</> : entity.review_state === "accepted" ? "Verified" : "Review"}
        </span>
      </div>
      <div className="property-meta">
        <code>{entity.id}</code>
        <span>{confidenceLabel(entity)} confidence</span>
      </div>
      <div className="property-scroll">
        {locked ? <div className="property-lock-note"><Lock size={13} /> Unlock this element or its category in the Model Browser to edit it.</div> : null}
        <PropertyFields collection={selection.collection} entity={entity} onEdit={onEdit} />
        {reviewPriority ? <ReviewRiskSection priority={reviewPriority} /> : null}
        {selection.collection === "fixtures" ? <AssetMatchSection entity={entity} /> : null}
        {selection.collection === "fixtures" ? (
          <section className="property-section family-property-action">
            <h3>Component family</h3>
            <button className="secondary-button" disabled={locked} onClick={onBrowseFamily}><Layers3 size={15} /> Browse and replace</button>
          </section>
        ) : null}
        {selection.collection === "openings" ? (
          <section className="property-section family-property-action">
            <h3>Host relationship</h3>
            <div className="audit-row"><span>Wall</span><b>{String(entity.wall_id ?? "Missing host")}</b></div>
            <button className="secondary-button" disabled={locked} onClick={onRequestOpeningRehost}>
              <MousePointer2 size={15} /> Pick new host in plan
            </button>
          </section>
        ) : null}
        <section className="property-section">
          <h3>Audit</h3>
          <div className="audit-row"><span>Source</span><b>{String(entity.model_version ?? "Unknown")}</b></div>
          <div className="audit-row"><span>Uncertainty</span><b>{Math.round(Number(entity.uncertainty ?? 0) * 100)}%</b></div>
          <div className="audit-row"><span>Session changes</span><b>{changeCount}</b></div>
        </section>
      </div>
      <div className="property-actions">
        <button className="secondary-button danger" disabled={locked} onClick={onDelete}><Trash2 size={15} /> Delete</button>
        <button className="primary-button" disabled={locked} onClick={onAccept}><Check size={15} /> Accept</button>
      </div>
      <div className="autosave-note"><History size={13} /> Patch history is saved locally</div>
    </aside>
  );
}

function AssetMatchSection({ entity }: { entity: BaseEntity }) {
  const score = finiteNumber(entity.asset_selection_score);
  if (score === null) return null;
  const margin = finiteNumber(entity.asset_selection_margin);
  const elapsedUs = finiteNumber(entity.asset_selection_elapsed_us);
  const context = isRecord(entity.asset_selection_context) ? entity.asset_selection_context : {};
  const components = isRecord(entity.asset_selection_components) ? entity.asset_selection_components : {};
  const alternates = Array.isArray(entity.asset_selection_alternates)
    ? entity.asset_selection_alternates.filter(isRecord).slice(0, 3)
    : [];
  const selectedName = String(entity.asset_name ?? entity.native_variant ?? entity.family_id ?? "Matched family");
  const nearby = Array.isArray(context.nearby_families)
    ? context.nearby_families.map(String).slice(0, 4)
    : [];
  const componentRows = [
    ["Envelope", finiteNumber(components.shape)],
    ["Footprint", finiteNumber(components.footprint)],
    ["Room context", finiteNumber(components.context)],
    ["Mesh detail", finiteNumber(components.detail)],
  ] as const;
  return (
    <section className="property-section asset-match-section">
      <div className="asset-match-heading">
        <div><ScanSearch size={14} /><h3>Context match</h3></div>
        <span className={entity.asset_selection_review_required ? "review" : "matched"}>
          {Math.round(score * 100)}% {entity.asset_selection_review_required ? "check" : "match"}
        </span>
      </div>
      <strong className="asset-match-name">{selectedName}</strong>
      <div className="asset-match-context">
        <span>{String(context.installation ?? "placement unknown").replaceAll("_", " ")}</span>
        {context.room_label ? <span>{String(context.room_label)}</span> : null}
        {finiteNumber(context.nearest_wall_m) !== null ? <span>{finiteNumber(context.nearest_wall_m)!.toFixed(2)} m to wall</span> : null}
      </div>
      <div className="asset-match-metrics">
        {componentRows.map(([label, value]) => value === null ? null : (
          <div key={label} className="asset-match-metric">
            <span>{label}</span><i><b style={{ width: `${Math.round(value * 100)}%` }} /></i><code>{Math.round(value * 100)}</code>
          </div>
        ))}
      </div>
      <div className="asset-match-speed">
        <Gauge size={13} />
        <span>{elapsedUs === null ? "Local decision" : `${(elapsedUs / 1000).toFixed(3)} ms decision`}</span>
        {margin !== null ? <span>{Math.round(margin * 1000) / 10}% lead</span> : null}
        <span>{Number(entity.asset_candidate_count ?? 0)} candidates</span>
      </div>
      {nearby.length ? <p className="asset-nearby">Nearby: {nearby.join(" · ")}</p> : null}
      {alternates.length ? (
        <div className="asset-alternates">
          <span>Next best</span>
          {alternates.map((alternate, index) => (
            <div key={String(alternate.asset_uid ?? alternate.native_variant ?? index)}>
              <b>{String(alternate.asset_name ?? alternate.native_variant ?? "Alternative")}</b>
              <code>{Math.round((finiteNumber(alternate.score) ?? 0) * 100)}%</code>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function finiteNumber(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function ReviewRiskSection({ priority }: { priority: ReviewPriority }) {
  return (
    <section className="property-section review-risk-section">
      <div className="review-risk-heading">
        <div><ShieldAlert size={14} /><h3>Review priority</h3></div>
        <span className={priority.band}>{priority.percent} / 100 · {priority.band}</span>
      </div>
      <p>Guides review order using model evidence and BIM consequences.</p>
      <ul>
        {priority.reasons.slice(0, 4).map((reason) => (
          <li key={reason.code}>
            <span>{reason.label}</span>
            <b>+{Math.round(reason.contribution * 100)}</b>
          </li>
        ))}
      </ul>
    </section>
  );
}

function BatchPropertyPanel({
  entities,
  selections,
  locked,
  onEdit,
  onAccept,
  onDelete,
  onBrowseFamily,
}: {
  entities: BaseEntity[];
  selections: Selection[];
  locked: boolean;
  onEdit: (changes: Record<string, unknown>) => void;
  onAccept: () => void;
  onDelete: () => void;
  onBrowseFamily: () => void;
}) {
  const collections = [...new Set(selections.map((item) => item.collection))];
  const collection = collections.length === 1 ? collections[0] : null;
  const reviewCount = entities.filter((item) => item.review_state !== "accepted").length;
  return (
    <aside className={locked ? "property-panel panel batch-properties is-locked" : "property-panel panel batch-properties"}>
      <div className="panel-title-row">
        <div>
          <span className="eyebrow">MULTI SELECTION</span>
          <h2>{selections.length} elements</h2>
        </div>
        <Layers3 size={18} />
      </div>
      <div className="property-meta">
        <code>{collection ? collection.replaceAll("_", " ") : `${collections.length} categories`}</code>
        <span>{reviewCount} still require review</span>
      </div>
      <div className="property-scroll">
        {locked ? <div className="property-lock-note"><Lock size={13} /> Unlock every selected element and category to edit this batch.</div> : null}
        <section className="property-section">
          <h3>Batch properties</h3>
          {collection === "walls" ? (
            <>
              <NumberField label="Height (m)" value={sharedNumber(entities, "height_m")} step={0.05} onCommit={(value) => onEdit({ height_m: value })} />
              <NumberField label="Thickness (m)" value={sharedNumber(entities, "thickness_m")} step={0.01} onCommit={(value) => onEdit({ thickness_m: value })} />
              <TextField label="Material" value={sharedText(entities, "material")} onCommit={(value) => onEdit({ material: value })} />
            </>
          ) : null}
          {collection === "openings" ? (
            <>
              <NumberField label="Width (m)" value={sharedNumber(entities, "width_m")} step={0.05} onCommit={(value) => onEdit({ width_m: value })} />
              <NumberField label="Height (m)" value={sharedNumber(entities, "height_m")} step={0.05} onCommit={(value) => onEdit({ height_m: value })} />
              <ChoiceField label="Type" value={sharedText(entities, "type")} options={openingTypeOptions} onCommit={(value) => onEdit({ type: value })} />
              {entities.every((item) => item.type === "door") ? (
                <>
                  <ChoiceField label="Operation" value={sharedText(entities, "operation_type")} options={doorOperationOptions} onCommit={(value) => onEdit({ operation_type: value })} />
                  <ChoiceField label="Hinge" value={sharedText(entities, "handing")} options={doorHandingOptions} onCommit={(value) => onEdit({ handing: value })} />
                  <ChoiceField label="Swing side" value={sharedText(entities, "swing_side")} options={doorSwingOptions} onCommit={(value) => onEdit({ swing_side: value })} />
                </>
              ) : null}
            </>
          ) : null}
          {collection === "fixtures" ? (
            <>
              <NumberField label="Base elevation (m)" value={sharedNumber(entities, "base_elevation_m")} step={0.05} onCommit={(value) => onEdit({ base_elevation_m: value })} />
              <NumberField label="Rotation (°)" value={sharedNumber(entities, "yaw_deg")} step={1} onCommit={(value) => onEdit({ yaw_deg: value })} />
              <TextField label="Family" value={sharedText(entities, "family_id")} onCommit={(value) => onEdit({ family_id: value })} />
              <button className="secondary-button batch-family-button" disabled={locked} onClick={onBrowseFamily}><Layers3 size={15} /> Replace from library</button>
            </>
          ) : null}
          {!collection || !["walls", "openings", "fixtures"].includes(collection) ? (
            <p className="batch-help">Choose elements from one editable category to expose shared dimensions. Ctrl click keeps the current selection.</p>
          ) : null}
        </section>
        <section className="property-section">
          <h3>Selection summary</h3>
          {collections.map((item) => (
            <div className="audit-row" key={item}>
              <span>{item.replaceAll("_", " ")}</span>
              <b>{selections.filter((selection) => selection.collection === item).length}</b>
            </div>
          ))}
        </section>
      </div>
      <div className="property-actions">
        <button className="secondary-button danger" disabled={locked} onClick={onDelete}><Trash2 size={15} /> Delete selected</button>
        <button className="primary-button" disabled={locked} onClick={onAccept}><Check size={15} /> Accept selected</button>
      </div>
      <div className="autosave-note"><History size={13} /> One undo restores the full batch</div>
    </aside>
  );
}

function sharedNumber(entities: BaseEntity[], key: string): number | null {
  const values = entities.map((entity) => Number(entity[key])).filter(Number.isFinite);
  if (!values.length || values.length !== entities.length) return null;
  return values.every((value) => Math.abs(value - values[0]) < 1e-9) ? values[0] : null;
}

function sharedText(entities: BaseEntity[], key: string): string | null {
  const values = entities.map((entity) => String(entity[key] ?? ""));
  return values.length && values.every((value) => value === values[0]) ? values[0] : null;
}

function entityTitle(collection: CollectionName, entity: BaseEntity): string {
  const type = entity.name ?? entity.type ?? entity.family_id;
  if (type) return String(type);
  return collection.slice(0, -1).replace("_", " ");
}

function PropertyFields({
  collection,
  entity,
  onEdit,
}: {
  collection: CollectionName;
  entity: BaseEntity;
  onEdit: (changes: Record<string, unknown>) => void;
}) {
  return (
    <>
      <section className="property-section">
        <h3>Identity</h3>
        {"name" in entity ? <TextField label="Name" value={String(entity.name ?? "")} onCommit={(value) => onEdit({ name: value })} /> : null}
        {"type" in entity && !["openings", "constraints"].includes(collection) ? <TextField label="Type" value={String(entity.type ?? "")} onCommit={(value) => onEdit({ type: value })} /> : null}
        {collection === "openings" ? <ChoiceField label="Type" value={String(entity.type ?? "opening")} options={openingTypeOptions} onCommit={(value) => onEdit({ type: value })} /> : null}
        {"kind" in entity ? <TextField label="Kind" value={String(entity.kind ?? "")} onCommit={(value) => onEdit({ kind: value })} /> : null}
        {"family_id" in entity ? <TextField label="Family" value={String(entity.family_id ?? "")} onCommit={(value) => onEdit({ family_id: value })} /> : null}
        {"discipline" in entity ? <TextField label="Discipline" value={String(entity.discipline ?? "")} onCommit={(value) => onEdit({ discipline: value })} /> : null}
        {"occupancy" in entity ? <TextField label="Occupancy" value={String(entity.occupancy ?? "")} onCommit={(value) => onEdit({ occupancy: value })} /> : null}
        {"material" in entity ? <TextField label="Material" value={String(entity.material ?? "")} onCommit={(value) => onEdit({ material: value })} /> : null}
      </section>
      <section className="property-section">
        <h3>Dimensions & position</h3>
        {collection === "walls" ? (
          <>
            <NumberField
              label="Length (m)"
              value={distanceMeters(entity.from as [number, number], entity.to as [number, number])}
              step={0.01}
              onCommit={(value) => {
                const changes = setWallLength(entity as WallEntity, value);
                if (changes) {
                  onEdit(changes);
                  return;
                }
                // Route an invalid driving value through the shared BIM planner so
                // the field reverts and the user receives the same explicit reason
                // as a rejected handle, Exact Move, or 3D gizmo edit.
                onEdit({ to: [...(entity.from as [number, number])] });
              }}
            />
            <PointField label="Start" value={entity.from as [number, number]} onCommit={(value) => onEdit({ from: value })} />
            <PointField label="End" value={entity.to as [number, number]} onCommit={(value) => onEdit({ to: value })} />
          </>
        ) : null}
        {collection === "dimensions" ? (
          <>
            <NumberField
              label="Measured length (m)"
              value={distanceMeters(entity.from as [number, number], entity.to as [number, number])}
              step={0.001}
              onCommit={(value) => {
                const changes = setSegmentLength(
                  entity.from as [number, number],
                  entity.to as [number, number],
                  value,
                );
                if (changes) onEdit(changes);
              }}
            />
            <PointField label="Start" value={entity.from as [number, number]} onCommit={(value) => onEdit({ from: value })} />
            <PointField label="End" value={entity.to as [number, number]} onCommit={(value) => onEdit({ to: value })} />
          </>
        ) : null}
        {collection === "fixtures" || collection === "openings" || collection === "vertical_connections" ? (
          <PointField label="Center" value={entity.center_m as [number, number]} onCommit={(value) => onEdit({ center_m: value })} />
        ) : null}
        {numericFields(collection).map(([key, label, step]) =>
          key in entity ? (
            <NumberField key={key} label={label} value={Number(entity[key])} step={step} onCommit={(value) => onEdit({ [key]: value })} />
          ) : null,
        )}
        {collection === "fixtures" && Array.isArray(entity.size_m) ? (
          <SizeField value={entity.size_m as [number, number, number]} onCommit={(value) => onEdit({ size_m: value })} />
        ) : null}
        {collection === "routes" && Array.isArray(entity.section_m) ? (
          <PairField label="Section (m)" axisA="W" axisB="H" value={entity.section_m as [number, number]} onCommit={(value) => onEdit({ section_m: value })} />
        ) : null}
        {collection === "vertical_connections" && Array.isArray(entity.footprint_m) ? (
          <PairField label="Footprint (m)" axisA="W" axisB="D" value={entity.footprint_m as [number, number]} onCommit={(value) => onEdit({ footprint_m: value })} />
        ) : null}
      </section>
      {collection === "openings" ? (
        <section className="property-section">
          <h3>Operation</h3>
          {entity.type === "door" ? (
            <>
              <ChoiceField label="Operation" value={String(entity.operation_type ?? "single_swing")} options={doorOperationOptions} onCommit={(value) => onEdit({ operation_type: value })} />
              <ChoiceField label="Hinge" value={String(entity.handing ?? "start")} options={doorHandingOptions} onCommit={(value) => onEdit({ handing: value })} />
              <ChoiceField label="Swing side" value={String(entity.swing_side ?? "positive")} options={doorSwingOptions} onCommit={(value) => onEdit({ swing_side: value })} />
            </>
          ) : null}
        </section>
      ) : null}
      {collection === "fixtures" ? (
        <section className="property-section">
          <h3>Placement relationship</h3>
          <div className="audit-row"><span>Host type</span><b>{String(entity.mounting ?? "floor")}</b></div>
          <div className="audit-row"><span>Assigned room</span><b>{String(entity.room_id ?? "Unassigned")}</b></div>
          {entity.host_wall_id ? <div className="audit-row"><span>Host wall</span><b>{String(entity.host_wall_id)}</b></div> : null}
          <p className="property-relationship-note">Room and host references update automatically after a verified move.</p>
        </section>
      ) : null}
      {collection === "vertical_connections" ? (
        <section className="property-section">
          <h3>Level connection</h3>
          <TextField label="From level" value={String(entity.from_level_id ?? "")} onCommit={(value) => onEdit({ from_level_id: value })} />
          <TextField label="To level" value={String(entity.to_level_id ?? "")} onCommit={(value) => onEdit({ to_level_id: value })} />
        </section>
      ) : null}
      {collection === "constraints" ? (
        <section className="property-section">
          <h3>Constraint references</h3>
          {(Array.isArray(entity.references) ? entity.references : []).map((reference, index) => {
            const item = reference as { entity_id?: string; handle?: string };
            return (
              <div className="audit-row" key={`${String(item.entity_id)}:${String(item.handle)}:${index}`}>
                <span>{String(item.handle ?? "endpoint")}</span>
                <b>{String(item.entity_id ?? "Missing reference")}</b>
              </div>
            );
          })}
        </section>
      ) : null}
    </>
  );
}

function numericFields(collection: CollectionName): Array<[string, string, number]> {
  if (collection === "walls") return [["height_m", "Height (m)", 0.05], ["thickness_m", "Thickness (m)", 0.01]];
  if (collection === "openings") return [["width_m", "Width (m)", 0.05], ["height_m", "Height (m)", 0.05], ["sill_height_m", "Sill (m)", 0.05]];
  if (collection === "fixtures") return [["base_elevation_m", "Base elevation (m)", 0.05], ["yaw_deg", "Rotation (°)", 1]];
  if (collection === "levels") return [["elevation_m", "Elevation (m)", 0.05], ["nominal_height_m", "Level height (m)", 0.05]];
  if (collection === "vertical_connections") return [["yaw_deg", "Rotation (°)", 1]];
  return [];
}

function TextField({ label, value, onCommit }: { label: string; value: string | null; onCommit: (value: string) => void }) {
  const [draft, setDraft] = useState(value ?? "");
  const [dirty, setDirty] = useState(false);
  useEffect(() => {
    setDraft(value ?? "");
    setDirty(false);
  }, [value]);
  return (
    <label className="property-field"><span>{label}</span><input value={draft} placeholder={value === null ? "Mixed values" : undefined} onChange={(event) => { setDraft(event.target.value); setDirty(true); }} onBlur={() => { if (dirty) onCommit(draft); setDraft(value ?? ""); setDirty(false); }} onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()} /></label>
  );
}

const openingTypeOptions = [
  ["door", "Door"],
  ["window", "Window"],
  ["opening", "Open passage"],
] as const;

const doorOperationOptions = [
  ["unknown", "Needs review"],
  ["single_swing", "Single swing"],
  ["double_swing", "Double swing"],
  ["sliding", "Sliding"],
  ["folding", "Folding"],
] as const;

const doorHandingOptions = [
  ["unknown", "Needs review"],
  ["start", "Wall start"],
  ["end", "Wall end"],
] as const;

const doorSwingOptions = [
  ["unknown", "Needs review"],
  ["positive", "Positive side"],
  ["negative", "Negative side"],
] as const;

function ChoiceField({
  label,
  value,
  options,
  onCommit,
}: {
  label: string;
  value: string | null;
  options: ReadonlyArray<readonly [string, string]>;
  onCommit: (value: string) => void;
}) {
  return (
    <label className="property-field">
      <span>{label}</span>
      <select
        value={value ?? ""}
        aria-label={label}
        onChange={(event) => onCommit(event.target.value)}
      >
        {value === null ? <option value="" disabled>Mixed values</option> : null}
        {options.map(([optionValue, optionLabel]) => (
          <option key={optionValue} value={optionValue}>{optionLabel}</option>
        ))}
      </select>
    </label>
  );
}

function NumberField({ label, value, step, onCommit }: { label: string; value: number | null; step: number; onCommit: (value: number) => void }) {
  const [draft, setDraft] = useState(formatNumber(value, step));
  const [dirty, setDirty] = useState(false);
  useEffect(() => {
    setDraft(formatNumber(value, step));
    setDirty(false);
  }, [value, step]);
  return (
    <label className="property-field"><span>{label}</span><input type="number" step={step} value={draft} placeholder={value === null ? "Mixed values" : undefined} onChange={(event) => { setDraft(event.target.value); setDirty(true); }} onBlur={() => { const next = Number(draft); if (dirty && draft.trim() && Number.isFinite(next)) onCommit(next); setDraft(formatNumber(value, step)); setDirty(false); }} onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()} /></label>
  );
}

function formatNumber(value: number | null, step: number): string {
  if (value === null || !Number.isFinite(value)) return "";
  const decimal = String(step).split(".")[1];
  const precision = decimal?.length ?? 0;
  return Number(value.toFixed(precision)).toString();
}

function PointField({ label, value, onCommit }: { label: string; value?: [number, number]; onCommit: (value: [number, number]) => void }) {
  const point = value ?? [0, 0];
  return <div className="property-vector"><span>{label}</span><NumberField label="X" value={point[0]} step={0.05} onCommit={(x) => onCommit([x, point[1]])} /><NumberField label="Y" value={point[1]} step={0.05} onCommit={(y) => onCommit([point[0], y])} /></div>;
}

function SizeField({ value, onCommit }: { value: [number, number, number]; onCommit: (value: [number, number, number]) => void }) {
  return <div className="property-vector size"><span>Size (m)</span><NumberField label="W" value={value[0]} step={0.05} onCommit={(x) => onCommit([x, value[1], value[2]])} /><NumberField label="D" value={value[1]} step={0.05} onCommit={(y) => onCommit([value[0], y, value[2]])} /><NumberField label="H" value={value[2]} step={0.05} onCommit={(z) => onCommit([value[0], value[1], z])} /></div>;
}

function PairField({ label, axisA, axisB, value, onCommit }: { label: string; axisA: string; axisB: string; value: [number, number]; onCommit: (value: [number, number]) => void }) {
  return <div className="property-vector"><span>{label}</span><NumberField label={axisA} value={value[0]} step={0.01} onCommit={(first) => onCommit([first, value[1]])} /><NumberField label={axisB} value={value[1]} step={0.01} onCommit={(second) => onCommit([value[0], second])} /></div>;
}
