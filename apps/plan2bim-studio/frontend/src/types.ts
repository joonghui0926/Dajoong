export type CollectionName =
  | "levels"
  | "walls"
  | "rooms"
  | "openings"
  | "fixtures"
  | "routes"
  | "vertical_connections"
  | "constraints"
  | "dimensions";

export type ReviewState = "accepted" | "review_required" | "rejected" | string;

export interface BaseEntity {
  id: string;
  level_id?: string;
  confidence?: number;
  uncertainty?: number;
  review_state?: ReviewState;
  model_version?: string;
  [key: string]: unknown;
}

export interface LevelEntity extends BaseEntity {
  name: string;
  elevation_m?: number;
  nominal_height_m?: number;
}

export interface WallEntity extends BaseEntity {
  level_id: string;
  from: [number, number];
  to: [number, number];
  thickness_m: number;
  height_m: number;
  wall_type?: string;
  material?: string;
}

export interface RoomEntity extends BaseEntity {
  level_id: string;
  name: string;
  polygon: [number, number][];
  occupancy?: string;
}

export interface OpeningEntity extends BaseEntity {
  level_id: string;
  type: string;
  wall_id: string;
  center_m: [number, number];
  width_m: number;
  height_m: number;
  sill_height_m?: number;
  x_m?: number;
  family_id?: string;
  operation_type?: "single_swing" | "double_swing" | "sliding" | "folding" | "fixed" | "unknown" | string;
  handing?: "start" | "end" | "double" | "unknown" | string;
  swing_side?: "positive" | "negative" | "both" | "none" | "unknown" | string;
}

export interface FixtureEntity extends BaseEntity {
  level_id: string;
  type: string;
  family_id?: string;
  discipline?: string;
  room_id?: string;
  center_m: [number, number];
  base_elevation_m?: number;
  size_m: [number, number, number];
  yaw_deg?: number;
  material?: string;
}

export interface RouteEntity extends BaseEntity {
  level_id: string;
  system_id?: string;
  discipline?: "electrical" | "mechanical" | "plumbing" | "fire" | string;
  kind?: string;
  points_m: [number, number, number][];
  section_m?: [number, number];
  material?: string;
}

export interface VerticalConnectionEntity extends BaseEntity {
  type?: "stair" | "ramp" | "escalator" | "elevator" | "riser" | string;
  kind?: "stair" | "ramp" | "escalator" | "elevator" | "riser" | string;
  from_level_id: string;
  to_level_id: string;
  shaft_id?: string;
  center_m: [number, number];
  footprint_m: [number, number];
  yaw_deg?: number;
}

export interface ConstraintReference {
  collection: "walls";
  entity_id: string;
  handle: "from" | "to";
}

export interface GeometricConstraintEntity extends BaseEntity {
  level_id: string;
  type: "coincident" | "distance" | string;
  references: ConstraintReference[];
  value_m?: number;
}

export interface PlanGraph {
  schema_version: string;
  project_id?: string;
  sheet_id?: string;
  levels: LevelEntity[];
  walls: WallEntity[];
  rooms: RoomEntity[];
  openings: OpeningEntity[];
  fixtures: FixtureEntity[];
  routes: RouteEntity[];
  vertical_connections?: VerticalConnectionEntity[];
  constraints?: GeometricConstraintEntity[];
  dimensions?: Measurement[];
  pipeline?: Record<string, unknown>;
  drawing_profile?: DrawingComplexityProfile;
  qualification?: ModelQualification;
  verification?: PlanGraphVerification;
  [key: string]: unknown;
}

export interface PlanGraphViolation {
  code: string;
  severity: "info" | "warning" | "error";
  message: string;
  entity_ids: string[];
  source_ref_ids?: string[];
  remediation?: string;
}

export interface PlanGraphVerification {
  schema_version: string;
  release_allowed: boolean;
  review_required: boolean;
  checked_invariants: number;
  passed_invariants: number;
  violations: PlanGraphViolation[];
  content_sha256?: string;
}

export interface DrawingComplexityProfile {
  schema_version: string;
  difficulty_class: "simple" | "moderate" | "difficult" | "extreme";
  complexity_score: number;
  reasons: string[];
  profiling_ms?: number;
  content_sha256?: string;
}

export interface QualificationClaim {
  claim: string;
  status: "measured" | "unmeasured" | "model_mismatch" | "insufficient_sample";
  metric?: string;
  estimate?: number | null;
  conservative_floor?: number | null;
  sample_count?: number;
  note?: string;
}

export interface ModelQualification {
  schema_version: string;
  exact_model_match?: boolean;
  difficulty_class?: string;
  benchmark_cohort?: string;
  benchmark_sample_count?: number;
  claims?: QualificationClaim[];
  production_release_eligible: boolean;
  review_required: boolean;
  review_reasons?: string[];
  manifest_sha256?: string;
}

export interface Selection {
  collection: CollectionName;
  id: string;
}

export interface ElementContextMenuRequest {
  anchor: Selection;
  targets: Selection[];
  clientX: number;
  clientY: number;
}

export interface Measurement extends BaseEntity {
  id: string;
  level_id: string;
  from: [number, number];
  to: [number, number];
  type?: "aligned" | "horizontal" | "vertical" | string;
  name?: string;
}

export interface CorrectionOperation {
  id: string;
  action: "add" | "update" | "delete" | "accept";
  collection: CollectionName;
  entity_id: string;
  changes: Record<string, unknown>;
  reason: string;
}

export type ViewMode = "split" | "plan" | "model";
export type EditorTool = "select" | "wall" | "opening" | "room" | "object" | "measure";
