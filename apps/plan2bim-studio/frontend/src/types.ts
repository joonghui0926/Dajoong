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
  geometry_status?: "evidence_sized" | "approved_family" | "semantic_marker" | "licensed_api_asset" | "native_bim_parametric" | string;
  asset_uid?: string;
  asset_name?: string;
  asset_author?: string;
  asset_license?: string;
  asset_source_uri?: string;
  asset_sha256?: string;
  asset_mesh_sha256?: string;
  geometry_ref?: string;
  geometry_scale_xyz?: [number, number, number];
  asset_candidate_count?: number;
  asset_selection_score?: number;
  asset_selection_margin?: number;
  asset_selection_review_required?: boolean;
  asset_selection_elapsed_us?: number;
  asset_selection_policy?: string;
  asset_selection_context?: {
    room_id?: string;
    room_label?: string;
    installation?: string;
    nearest_wall_m?: number | null;
    nearest_wall_angle_deg?: number | null;
    nearby_families?: string[];
  };
  asset_selection_components?: {
    score?: number;
    footprint?: number;
    shape?: number;
    context?: number;
    detail?: number;
  };
  asset_selection_alternates?: Array<{
    asset_uid?: string;
    asset_name?: string;
    native_variant?: string;
    score?: number;
  }>;
  mesh_vertices?: [number, number, number][];
  mesh_faces?: [number, number, number][];
  mesh_face_colors?: [number, number, number][];
}

export interface FamilyAssetDefinition {
  schema_version: "dajoong.family-asset.v1" | string;
  geometry_status: string;
  asset_uid?: string;
  family_id?: string;
  asset_mesh_sha256: string;
  normalized_to_unit_envelope?: boolean;
  mesh_vertices: [number, number, number][];
  mesh_faces: [number, number, number][];
  mesh_face_colors?: [number, number, number][];
}

export interface AssetDelivery {
  schema_version: "dajoong.asset-delivery.v1" | string;
  catalog_url: string;
  mesh_url_template: string;
  format: "dajoong.mesh.v1" | string;
  content_addressed: boolean;
  lazy_by_visible_level: boolean;
}

export interface DetectionReviewCandidate {
  id: string;
  level_id: string;
  source_candidate_id: string;
  source_bbox_px: [number, number, number, number];
  bbox_m: [number, number, number, number];
  proposed_type: string;
  confidence: number;
  reason: string;
  review_state: "review_required" | string;
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
  family_assets?: Record<string, FamilyAssetDefinition>;
  asset_delivery?: AssetDelivery;
  detection_review_candidates?: DetectionReviewCandidate[];
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
