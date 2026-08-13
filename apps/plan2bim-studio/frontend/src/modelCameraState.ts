export type Vector3Tuple = [number, number, number];

export interface ModelCameraSnapshot {
  levelId: string;
  viewRevision: number;
  position: Vector3Tuple;
  target: Vector3Tuple;
}

export function canRestoreModelCamera(
  snapshot: ModelCameraSnapshot | null,
  levelId: string,
  viewRevision: number,
): snapshot is ModelCameraSnapshot {
  return snapshot?.levelId === levelId && snapshot.viewRevision === viewRevision;
}
