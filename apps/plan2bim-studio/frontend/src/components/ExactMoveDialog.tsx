import { Move, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { parseOffsetInput } from "../precisionInput";

interface ExactMoveDialogProps {
  open: boolean;
  count: number;
  onClose: () => void;
  onApply: (delta: [number, number]) => string | null;
}

export function ExactMoveDialog({
  open,
  count,
  onClose,
  onApply,
}: ExactMoveDialogProps) {
  const [deltaX, setDeltaX] = useState("0 m");
  const [deltaY, setDeltaY] = useState("0 m");
  const [error, setError] = useState("");
  useEffect(() => {
    if (!open) return;
    setDeltaX("0 m");
    setDeltaY("0 m");
    setError("");
  }, [open]);
  const parsed = useMemo(() => {
    const x = parseOffsetInput(deltaX);
    const y = parseOffsetInput(deltaY);
    return x === null || y === null ? null : [x, y] as [number, number];
  }, [deltaX, deltaY]);
  if (!open) return null;

  return (
    <div
      className="dialog-backdrop exact-move-backdrop"
      role="presentation"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <form
        className="exact-move-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="exact-move-title"
        onSubmit={(event) => {
          event.preventDefault();
          if (!parsed) {
            setError("Enter a valid metric or imperial offset in both fields.");
            return;
          }
          const applyError = onApply(parsed);
          setError(applyError ?? "");
        }}
      >
        <header>
          <div className="exact-move-icon"><Move size={17} /></div>
          <div>
            <span>PRECISE TRANSFORM</span>
            <h2 id="exact-move-title">Move {count} selected {count === 1 ? "element" : "elements"}</h2>
          </div>
          <button type="button" onClick={onClose} aria-label="Close exact move"><X size={17} /></button>
        </header>
        <p>
          Enter signed offsets. Hosted openings and wall components stay attached. Room boundaries, walls, and object clearances are verified before commit.
        </p>
        <div className="exact-move-fields">
          <label>
            <span>Delta X</span>
            <input
              autoFocus
              aria-label="Delta X"
              value={deltaX}
              onChange={(event) => setDeltaX(event.target.value)}
              aria-invalid={Boolean(deltaX.trim() && parseOffsetInput(deltaX) === null)}
            />
          </label>
          <label>
            <span>Delta Y</span>
            <input
              aria-label="Delta Y"
              value={deltaY}
              onChange={(event) => setDeltaY(event.target.value)}
              aria-invalid={Boolean(deltaY.trim() && parseOffsetInput(deltaY) === null)}
            />
          </label>
        </div>
        <small>Examples: 250 mm, -0.5 m, 8 in, 1 ft 6 in</small>
        {error ? <div className="exact-move-error" role="alert">{error}</div> : null}
        <footer>
          <button type="button" onClick={onClose}>Cancel</button>
          <button
            className="primary-button"
            type="submit"
            disabled={!parsed || (parsed[0] === 0 && parsed[1] === 0)}
          >
            Apply move
          </button>
        </footer>
      </form>
    </div>
  );
}
