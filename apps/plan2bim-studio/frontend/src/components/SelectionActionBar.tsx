import {
  AlignHorizontalDistributeCenter,
  Check,
  ClipboardCopy,
  ClipboardPaste,
  Copy,
  FlipHorizontal2,
  Link2,
  Layers2,
  Lock,
  Move,
  MousePointer2,
  Route,
  Rows3,
  Scan,
  RefreshCcw,
  RotateCw,
  Trash2,
} from "lucide-react";
import type { AlignmentMode, DistributionAxis } from "../editorCommands";
import type { LevelEntity } from "../types";

interface SelectionActionBarProps {
  count: number;
  canDuplicate: boolean;
  canCopy: boolean;
  canPaste: boolean;
  canMoveExact: boolean;
  canRotateExact: boolean;
  canPattern: boolean;
  canAlign: boolean;
  canDistribute: boolean;
  keyObjectLabel: string;
  canFlipDoor: boolean;
  canRehostOpening: boolean;
  canJoinWalls: boolean;
  copyTargets: LevelEntity[];
  locked: boolean;
  onDuplicate: () => void;
  onCopy: () => void;
  onPaste: () => void;
  onMoveExact: () => void;
  onRotateExact: () => void;
  onMirror: () => void;
  onArray: () => void;
  onIsolate: () => void;
  onAlign: (mode: AlignmentMode) => void;
  onDistribute: (axis: DistributionAxis) => void;
  onFlipDoorHanding: () => void;
  onRehostOpening: () => void;
  onReverseDoorSwing: () => void;
  onJoinWalls: () => void;
  onCornerWalls: () => void;
  onCopyToLevel: (levelId: string) => void;
  onAccept: () => void;
  onDelete: () => void;
}

export function SelectionActionBar({
  count,
  canDuplicate,
  canCopy,
  canPaste,
  canMoveExact,
  canRotateExact,
  canPattern,
  canAlign,
  canDistribute,
  keyObjectLabel,
  canFlipDoor,
  canRehostOpening,
  canJoinWalls,
  copyTargets,
  locked,
  onDuplicate,
  onCopy,
  onPaste,
  onMoveExact,
  onRotateExact,
  onMirror,
  onArray,
  onIsolate,
  onAlign,
  onDistribute,
  onFlipDoorHanding,
  onRehostOpening,
  onReverseDoorSwing,
  onJoinWalls,
  onCornerWalls,
  onCopyToLevel,
  onAccept,
  onDelete,
}: SelectionActionBarProps) {
  if (!count) return null;
  return (
    <div className="selection-action-bar" role="toolbar" aria-label="Selection actions">
      <span>{count === 1 ? "1 selected" : `${count} selected`}</span>
      {locked ? <span className="selection-locked"><Lock size={12} /> Locked</span> : null}
      <button onClick={onIsolate} title="Isolate selected in 2D and 3D (I)">
        <Scan size={14} /><b>Isolate</b>
      </button>
      <button onClick={onCopy} disabled={!canCopy} title="Copy BIM selection with dependencies (Ctrl+C)">
        <ClipboardCopy size={14} /><b>Copy</b>
      </button>
      <button onClick={onPaste} disabled={!canPaste} title="Paste BIM selection at nearest clear location (Ctrl+V)">
        <ClipboardPaste size={14} /><b>Paste</b>
      </button>
      <button onClick={onDuplicate} disabled={!canDuplicate || locked} title="Duplicate (Ctrl+D)">
        <Copy size={14} /><b>Duplicate</b>
      </button>
      <button onClick={onMoveExact} disabled={!canMoveExact || locked} title="Move by exact offset (M)">
        <Move size={14} /><b>Move exact</b>
      </button>
      {canRehostOpening ? (
        <button onClick={onRehostOpening} disabled={locked} title="Pick a new host wall (Shift+H)">
          <MousePointer2 size={14} /><b>Pick host</b>
        </button>
      ) : null}
      <button onClick={onRotateExact} disabled={!canRotateExact || locked} title="Rotate components by an exact angle and pivot (R)">
        <RotateCw size={14} /><b>Rotate exact</b>
      </button>
      <button onClick={onMirror} disabled={!canPattern || locked} title="Mirror placed components about an exact plan line (Shift+M)">
        <FlipHorizontal2 size={14} /><b>Mirror</b>
      </button>
      <button onClick={onArray} disabled={!canPattern || locked} title="Create an exact linear array of placed components (Shift+A)">
        <Rows3 size={14} /><b>Array</b>
      </button>
      {copyTargets.length ? (
        <label className="selection-copy-level" title="Copy selection with hosted BIM dependencies">
          <Layers2 size={14} />
          <select
            aria-label="Copy selected to level"
            disabled={locked}
            value=""
            onChange={(event) => {
              if (event.target.value) onCopyToLevel(event.target.value);
            }}
          >
            <option value="">Copy to…</option>
            {copyTargets.map((level) => <option key={level.id} value={level.id}>{level.name}</option>)}
          </select>
        </label>
      ) : null}
      {count >= 2 ? (
        <>
          <i />
          <label
            className="selection-arrange"
            title={`Last selected is the fixed key object: ${keyObjectLabel}. Alt+X/Y aligns centers. Alt+Shift+X/Y spaces equally.`}
          >
            <AlignHorizontalDistributeCenter size={14} />
            <select
              aria-label={`Arrange selected components. Key object ${keyObjectLabel}`}
              disabled={!canAlign || locked}
              value=""
              onChange={(event) => {
                const [kind, value] = event.target.value.split(":") as ["align" | "distribute", AlignmentMode | DistributionAxis];
                if (kind === "align") onAlign(value as AlignmentMode);
                if (kind === "distribute") onDistribute(value as DistributionAxis);
              }}
            >
              <option value="">Arrange…</option>
              <optgroup label={`Align to key · ${keyObjectLabel}`}>
                <option value="align:left">Left edges</option>
                <option value="align:center-x">Horizontal centers</option>
                <option value="align:right">Right edges</option>
                <option value="align:top">Top edges</option>
                <option value="align:center-y">Vertical centers</option>
                <option value="align:bottom">Bottom edges</option>
              </optgroup>
              <optgroup label="Equal clear spacing">
                <option value="distribute:horizontal" disabled={!canDistribute}>Horizontal · outer anchors fixed</option>
                <option value="distribute:vertical" disabled={!canDistribute}>Vertical · outer anchors fixed</option>
              </optgroup>
            </select>
          </label>
        </>
      ) : null}
      {canFlipDoor ? (
        <>
          <i />
          <button onClick={onFlipDoorHanding} title="Flip door hinge (H)">
            <FlipHorizontal2 size={14} /><b>Flip hinge</b>
          </button>
          <button onClick={onReverseDoorSwing} title="Reverse door swing side (S)">
            <RefreshCcw size={14} /><b>Reverse swing</b>
          </button>
        </>
      ) : null}
      {canJoinWalls ? (
        <>
          <i />
          <button onClick={onJoinWalls} title="Join nearest wall endpoints">
            <Link2 size={14} /><b>Join</b>
          </button>
          <button onClick={onCornerWalls} title="Trim or extend walls to corner">
            <Route size={14} /><b>Corner</b>
          </button>
        </>
      ) : null}
      <i />
      <button onClick={onAccept} disabled={locked} title="Accept selected"><Check size={14} /><b>Accept</b></button>
      <button className="danger" onClick={onDelete} disabled={locked} title="Delete selected">
        <Trash2 size={14} /><b>Delete</b>
      </button>
    </div>
  );
}
