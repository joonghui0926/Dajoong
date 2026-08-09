import { CopyPlus, FlipHorizontal2, Rows3, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { parseOffsetInput } from "../precisionInput";
import type { MirrorAxis } from "../repetitionCommands";

export type PatternMode = "mirror" | "array";

interface PatternDialogProps {
  open: PatternMode | null;
  selectionCount: number;
  defaultCenter: [number, number] | null;
  onClose: () => void;
  onMirror: (axis: MirrorAxis, coordinateM: number, keepOriginal: boolean) => string | null;
  onArray: (count: number, step: [number, number]) => string | null;
}

export function PatternDialog({
  open,
  selectionCount,
  defaultCenter,
  onClose,
  onMirror,
  onArray,
}: PatternDialogProps) {
  const [mode, setMode] = useState<PatternMode>("mirror");
  const [axis, setAxis] = useState<MirrorAxis>("vertical");
  const [axisCoordinate, setAxisCoordinate] = useState("0 m");
  const [keepOriginal, setKeepOriginal] = useState(true);
  const [count, setCount] = useState("3");
  const [stepX, setStepX] = useState("1 m");
  const [stepY, setStepY] = useState("0 m");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setMode(open);
    setAxis("vertical");
    setAxisCoordinate(`${(defaultCenter?.[0] ?? 0).toFixed(3)} m`);
    setKeepOriginal(true);
    setCount("3");
    setStepX("1 m");
    setStepY("0 m");
    setError("");
  }, [defaultCenter, open]);

  useEffect(() => {
    if (!open || !defaultCenter) return;
    setAxisCoordinate(`${(axis === "vertical" ? defaultCenter[0] : defaultCenter[1]).toFixed(3)} m`);
  }, [axis, defaultCenter, open]);

  const parsedCoordinate = useMemo(() => parseOffsetInput(axisCoordinate), [axisCoordinate]);
  const parsedStep = useMemo(() => {
    const x = parseOffsetInput(stepX);
    const y = parseOffsetInput(stepY);
    return x === null || y === null ? null : [x, y] as [number, number];
  }, [stepX, stepY]);
  const parsedCount = Number(count);
  if (!open) return null;

  return (
    <div
      className="dialog-backdrop exact-move-backdrop"
      role="presentation"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <form
        className="exact-move-dialog pattern-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="pattern-dialog-title"
        onSubmit={(event) => {
          event.preventDefault();
          const applyError = mode === "mirror"
            ? parsedCoordinate === null
              ? "Enter a valid mirror-line coordinate."
              : onMirror(axis, parsedCoordinate, keepOriginal)
            : !parsedStep
              ? "Enter valid metric or imperial spacing."
              : onArray(parsedCount, parsedStep);
          setError(applyError ?? "");
        }}
      >
        <header>
          <div className="exact-move-icon">{mode === "mirror" ? <FlipHorizontal2 size={17} /> : <Rows3 size={17} />}</div>
          <div>
            <span>REPEAT GEOMETRY</span>
            <h2 id="pattern-dialog-title">Pattern {selectionCount} selected {selectionCount === 1 ? "component" : "components"}</h2>
          </div>
          <button type="button" onClick={onClose} aria-label="Close pattern"><X size={17} /></button>
        </header>
        <div className="pattern-tabs" role="tablist" aria-label="Pattern type">
          <button type="button" role="tab" aria-selected={mode === "mirror"} onClick={() => { setMode("mirror"); setError(""); }}><FlipHorizontal2 size={14} /> Mirror</button>
          <button type="button" role="tab" aria-selected={mode === "array"} onClick={() => { setMode("array"); setError(""); }}><Rows3 size={14} /> Linear array</button>
        </div>
        {mode === "mirror" ? (
          <>
            <p>Reflect around an exact plan coordinate. Keeping the original creates an audited mirrored copy.</p>
            <div className="exact-move-fields pattern-fields">
              <label>
                <span>Mirror line</span>
                <select aria-label="Mirror line" value={axis} onChange={(event) => setAxis(event.target.value as MirrorAxis)}>
                  <option value="vertical">Vertical · X =</option>
                  <option value="horizontal">Horizontal · Y =</option>
                </select>
              </label>
              <label>
                <span>Coordinate</span>
                <input autoFocus aria-label="Mirror coordinate" value={axisCoordinate} onChange={(event) => setAxisCoordinate(event.target.value)} aria-invalid={parsedCoordinate === null} />
              </label>
            </div>
            <label className="pattern-keep-original"><input type="checkbox" checked={keepOriginal} onChange={(event) => setKeepOriginal(event.target.checked)} /><span><b>Keep original</b><small>Create mirrored copies instead of moving the selected components.</small></span></label>
          </>
        ) : (
          <>
            <p>Create an exact row of copies. Total instances includes the selected source set.</p>
            <div className="exact-move-fields pattern-array-fields">
              <label>
                <span>Total instances</span>
                <input autoFocus type="number" min="2" max="100" step="1" aria-label="Total instances" value={count} onChange={(event) => setCount(event.target.value)} />
              </label>
              <label>
                <span>Step X</span>
                <input aria-label="Array step X" value={stepX} onChange={(event) => setStepX(event.target.value)} aria-invalid={Boolean(stepX.trim() && parseOffsetInput(stepX) === null)} />
              </label>
              <label>
                <span>Step Y</span>
                <input aria-label="Array step Y" value={stepY} onChange={(event) => setStepY(event.target.value)} aria-invalid={Boolean(stepY.trim() && parseOffsetInput(stepY) === null)} />
              </label>
            </div>
          </>
        )}
        <small>Units: mm, cm, m, in, ft · each command is one Undo step</small>
        {error ? <div className="exact-move-error" role="alert">{error}</div> : null}
        <footer>
          <button type="button" onClick={onClose}>Cancel</button>
          <button
            className="primary-button"
            type="submit"
            disabled={mode === "mirror"
              ? parsedCoordinate === null
              : !parsedStep || !Number.isInteger(parsedCount) || parsedCount < 2 || parsedCount > 100 || Math.hypot(...parsedStep) < 0.000001}
          >
            {mode === "mirror" ? <><CopyPlus size={14} /> Apply mirror</> : <><Rows3 size={14} /> Create array</>}
          </button>
        </footer>
      </form>
    </div>
  );
}
