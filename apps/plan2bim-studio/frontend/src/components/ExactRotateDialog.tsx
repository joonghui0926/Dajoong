import { RotateCw, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { parseOffsetInput, parseSignedAngleInput } from "../precisionInput";

interface ExactRotateDialogProps {
  open: boolean;
  count: number;
  defaultPivot: [number, number] | null;
  onClose: () => void;
  onApply: (deltaDegrees: number, pivot: [number, number]) => string | null;
}

export function ExactRotateDialog({
  open,
  count,
  defaultPivot,
  onClose,
  onApply,
}: ExactRotateDialogProps) {
  const [angle, setAngle] = useState("90 deg");
  const [pivotX, setPivotX] = useState("0 m");
  const [pivotY, setPivotY] = useState("0 m");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setAngle("90 deg");
    setPivotX(`${(defaultPivot?.[0] ?? 0).toFixed(3)} m`);
    setPivotY(`${(defaultPivot?.[1] ?? 0).toFixed(3)} m`);
    setError("");
  }, [defaultPivot, open]);

  const parsed = useMemo(() => {
    const degrees = parseSignedAngleInput(angle);
    const x = parseOffsetInput(pivotX);
    const y = parseOffsetInput(pivotY);
    return degrees === null || x === null || y === null
      ? null
      : { degrees, pivot: [x, y] as [number, number] };
  }, [angle, pivotX, pivotY]);
  if (!open) return null;

  return (
    <div
      className="dialog-backdrop exact-move-backdrop"
      role="presentation"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <form
        className="exact-move-dialog exact-rotate-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="exact-rotate-title"
        onSubmit={(event) => {
          event.preventDefault();
          if (!parsed) {
            setError("Enter a valid signed angle and finite pivot coordinates.");
            return;
          }
          setError(onApply(parsed.degrees, parsed.pivot) ?? "");
        }}
      >
        <header>
          <div className="exact-move-icon"><RotateCw size={17} /></div>
          <div>
            <span>PRECISE TRANSFORM</span>
            <h2 id="exact-rotate-title">Rotate {count} selected {count === 1 ? "component" : "components"}</h2>
          </div>
          <button type="button" onClick={onClose} aria-label="Close exact rotate"><X size={17} /></button>
        </header>
        <p>Enter a signed relative angle. Every selected component rotates around the same editable plan pivot.</p>
        <div className="rotate-angle-row">
          <label>
            <span>Relative angle</span>
            <input autoFocus aria-label="Relative angle" value={angle} onChange={(event) => setAngle(event.target.value)} aria-invalid={parseSignedAngleInput(angle) === null} />
          </label>
          <div className="rotate-angle-presets" aria-label="Angle presets">
            {[-90, -45, 15, 45, 90, 180].map((degrees) => (
              <button type="button" key={degrees} onClick={() => setAngle(`${degrees} deg`)}>{degrees > 0 ? "+" : ""}{degrees}°</button>
            ))}
          </div>
        </div>
        <div className="exact-move-fields rotate-pivot-fields">
          <label>
            <span>Pivot X</span>
            <input aria-label="Rotation pivot X" value={pivotX} onChange={(event) => setPivotX(event.target.value)} aria-invalid={parseOffsetInput(pivotX) === null} />
          </label>
          <label>
            <span>Pivot Y</span>
            <input aria-label="Rotation pivot Y" value={pivotY} onChange={(event) => setPivotY(event.target.value)} aria-invalid={parseOffsetInput(pivotY) === null} />
          </label>
        </div>
        <small>Positive angles rotate counterclockwise in plan · units: mm, cm, m, in, ft</small>
        {error ? <div className="exact-move-error" role="alert">{error}</div> : null}
        <footer>
          <button type="button" onClick={onClose}>Cancel</button>
          <button className="primary-button" type="submit" disabled={!parsed || Math.abs(parsed.degrees) < 0.000001}>
            Apply rotation
          </button>
        </footer>
      </form>
    </div>
  );
}
