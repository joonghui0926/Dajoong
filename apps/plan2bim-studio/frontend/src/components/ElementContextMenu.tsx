import {
  AlignHorizontalDistributeCenter,
  AlignVerticalDistributeCenter,
  Check,
  ClipboardCopy,
  ClipboardPaste,
  Copy,
  Eye,
  EyeOff,
  Lock,
  LockOpen,
  MousePointer2,
  Scan,
  Trash2,
  Waypoints,
} from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";

import type { ElementContextMenuRequest } from "../types";
import type { RelatedSelectionGroup } from "../bimRelations";
import type { AlignmentMode, DistributionAxis } from "../editorCommands";

interface ElementContextMenuProps {
  request: ElementContextMenuRequest;
  title: string;
  hidden: boolean;
  locked: boolean;
  allLocked: boolean;
  inheritedLock: boolean;
  accepted: boolean;
  canDuplicate: boolean;
  canCopy: boolean;
  canPaste: boolean;
  canRehostOpening: boolean;
  canArrange: boolean;
  canDistribute: boolean;
  keyObjectLabel: string;
  relatedActions: RelatedSelectionGroup[];
  onClose: () => void;
  onIsolate: () => void;
  onToggleVisibility: () => void;
  onToggleLock: () => void;
  onDuplicate: () => void;
  onCopy: () => void;
  onPaste: () => void;
  onRehostOpening: () => void;
  onAlign: (mode: AlignmentMode) => void;
  onDistribute: (axis: DistributionAxis) => void;
  onAccept: () => void;
  onDelete: () => void;
  onSelectRelated: (group: RelatedSelectionGroup) => void;
}

export function ElementContextMenu({
  request,
  title,
  hidden,
  locked,
  allLocked,
  inheritedLock,
  accepted,
  canDuplicate,
  canCopy,
  canPaste,
  canRehostOpening,
  canArrange,
  canDistribute,
  keyObjectLabel,
  relatedActions,
  onClose,
  onIsolate,
  onToggleVisibility,
  onToggleLock,
  onDuplicate,
  onCopy,
  onPaste,
  onRehostOpening,
  onAlign,
  onDistribute,
  onAccept,
  onDelete,
  onSelectRelated,
}: ElementContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);
  const position = useMemo(() => {
    const viewportWidth = typeof window === "undefined" ? 1280 : window.innerWidth;
    const viewportHeight = typeof window === "undefined" ? 800 : window.innerHeight;
    return {
      left: Math.max(8, Math.min(request.clientX, viewportWidth - 232)),
      top: Math.max(8, Math.min(request.clientY, viewportHeight - 420)),
    };
  }, [request.clientX, request.clientY]);

  useEffect(() => {
    menuRef.current?.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus();
  }, [request]);

  const run = (action: () => void) => {
    action();
    onClose();
  };

  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      onClose();
      return;
    }
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    const buttons = [...(menuRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? [])];
    if (!buttons.length) return;
    event.preventDefault();
    const activeIndex = buttons.indexOf(document.activeElement as HTMLButtonElement);
    if (event.key === "Home") buttons[0].focus();
    else if (event.key === "End") buttons.at(-1)?.focus();
    else {
      const direction = event.key === "ArrowDown" ? 1 : -1;
      buttons[(activeIndex + direction + buttons.length) % buttons.length].focus();
    }
  };

  return (
    <div className="element-menu-layer" onPointerDown={onClose} onContextMenu={(event) => event.preventDefault()}>
      <div
        ref={menuRef}
        className="element-context-menu"
        role="menu"
        aria-label="Element actions"
        style={position}
        onPointerDown={(event) => event.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <div className="element-menu-heading">
          <strong>{request.targets.length > 1 ? `${request.targets.length} elements` : title}</strong>
          <small>{request.targets.length > 1 ? "Current selection" : request.anchor.id}</small>
        </div>
        <button role="menuitem" onClick={() => run(onIsolate)}>
          <Scan size={14} /><span>Isolate in 2D and 3D</span><kbd>I</kbd>
        </button>
        <button role="menuitem" onClick={() => run(onToggleVisibility)}>
          {hidden ? <Eye size={14} /> : <EyeOff size={14} />}
          <span>{hidden ? "Show selection" : "Hide selection"}</span>
        </button>
        <button
          role="menuitem"
          disabled={inheritedLock}
          title={inheritedLock ? "Unlock the parent category first" : undefined}
          onClick={() => run(onToggleLock)}
        >
          {allLocked ? <LockOpen size={14} /> : <Lock size={14} />}
          <span>{inheritedLock ? "Locked by category" : allLocked ? "Unlock selection" : "Lock selection"}</span>
        </button>
        {relatedActions.length ? <i /> : null}
        {relatedActions.map((group) => (
          <button key={group.id} role="menuitem" onClick={() => run(() => onSelectRelated(group))}>
            <Waypoints size={14} /><span>{group.label}</span>
          </button>
        ))}
        <i />
        <button role="menuitem" disabled={!canCopy} onClick={() => run(onCopy)}>
          <ClipboardCopy size={14} /><span>Copy BIM selection</span><kbd>Ctrl C</kbd>
        </button>
        <button role="menuitem" disabled={!canPaste} onClick={() => run(onPaste)}>
          <ClipboardPaste size={14} /><span>Paste BIM selection</span><kbd>Ctrl V</kbd>
        </button>
        <button role="menuitem" disabled={!canDuplicate || locked} onClick={() => run(onDuplicate)}>
          <Copy size={14} /><span>Duplicate</span><kbd>Ctrl D</kbd>
        </button>
        {canRehostOpening ? (
          <button role="menuitem" disabled={locked} onClick={() => run(onRehostOpening)}>
            <MousePointer2 size={14} /><span>Pick new host wall</span><kbd>Shift H</kbd>
          </button>
        ) : null}
        {request.targets.length >= 2 ? (
          <>
            <i />
            <div className="element-menu-subheading">Arrange · key {keyObjectLabel}</div>
            <button role="menuitem" disabled={!canArrange || locked} onClick={() => run(() => onAlign("center-x"))}>
              <AlignHorizontalDistributeCenter size={14} /><span>Align centers horizontally</span>
            </button>
            <button role="menuitem" disabled={!canArrange || locked} onClick={() => run(() => onAlign("center-y"))}>
              <AlignVerticalDistributeCenter size={14} /><span>Align centers vertically</span>
            </button>
            {request.targets.length >= 3 ? (
              <>
                <button role="menuitem" disabled={!canDistribute || locked} onClick={() => run(() => onDistribute("horizontal"))}>
                  <AlignHorizontalDistributeCenter size={14} /><span>Equal horizontal spacing</span>
                </button>
                <button role="menuitem" disabled={!canDistribute || locked} onClick={() => run(() => onDistribute("vertical"))}>
                  <AlignVerticalDistributeCenter size={14} /><span>Equal vertical spacing</span>
                </button>
              </>
            ) : null}
          </>
        ) : null}
        <button role="menuitem" disabled={accepted || locked} onClick={() => run(onAccept)}>
          <Check size={14} /><span>{accepted ? "Review accepted" : "Accept review"}</span>
        </button>
        <button className="danger" role="menuitem" disabled={locked} onClick={() => run(onDelete)}>
          <Trash2 size={14} /><span>Delete</span><kbd>Del</kbd>
        </button>
      </div>
    </div>
  );
}
