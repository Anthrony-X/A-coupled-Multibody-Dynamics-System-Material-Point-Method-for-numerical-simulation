from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


VEC3 = tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a tracked vehicle OBJ model, identify named parts, and export a "
            "multibody-oriented part inventory and kinematic tree."
        )
    )
    parser.add_argument(
        "--obj",
        type=Path,
        default=Path("ZTZ_96/source/ZTZ96/ZTZ96.obj"),
        help="Path to the source OBJ model",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ZTZ_96/multibody"),
        help="Directory for the generated JSON/CSV/Markdown outputs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ZTZ96",
        help="Vehicle/model name to store in the generated metadata",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file prefix. Default is derived from --model-name.",
    )
    parser.add_argument("--mass-kg", type=float, default=None, help="Open-source/reference vehicle mass, kg")
    parser.add_argument("--nominal-length-m", type=float, default=None, help="Open-source/reference overall length, m")
    parser.add_argument("--nominal-width-m", type=float, default=None, help="Open-source/reference overall width, m")
    parser.add_argument("--nominal-height-m", type=float, default=None, help="Open-source/reference overall height, m")
    parser.add_argument("--engine-power-kw", type=float, default=None, help="Open-source/reference engine power, kW")
    parser.add_argument("--max-road-speed-mps", type=float, default=None, help="Open-source/reference road speed, m/s")
    parser.add_argument("--range-km", type=float, default=None, help="Open-source/reference road range, km")
    parser.add_argument("--crew", type=int, default=None, help="Open-source/reference crew count")
    parser.add_argument("--passengers", type=int, default=None, help="Open-source/reference passenger/dismount count")
    parser.add_argument("--track-width-m", type=float, default=None, help="Manual calibrated track width override, m")
    parser.add_argument("--track-pitch-m", type=float, default=None, help="Manual calibrated track pitch override, m")
    parser.add_argument("--shoe-length-m", type=float, default=None, help="Manual calibrated track shoe length override, m")
    parser.add_argument(
        "--spec-source",
        action="append",
        default=[],
        help="Reference source used for the open-source vehicle parameters. May be repeated.",
    )
    return parser.parse_args()


def output_prefix_from_model_name(model_name: str, output_prefix: str | None) -> str:
    if output_prefix:
        return output_prefix
    prefix = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return prefix or "tracked_vehicle"


def parse_obj_index(token: str, vertex_count: int) -> int:
    head = token.split("/", 1)[0]
    raw = int(head)
    return vertex_count + raw if raw < 0 else raw - 1


def load_material_semantics(mtl_path: Path) -> dict[str, str]:
    materials: dict[str, str] = {}
    current: str | None = None

    if not mtl_path.exists():
        return materials

    with mtl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("newmtl "):
                current = line.split(maxsplit=1)[1].strip()
            elif current and line.startswith("map_Kd "):
                texture = Path(line.split(maxsplit=1)[1].strip()).name
                materials[current] = texture

    return materials


def load_obj_parts(obj_path: Path, material_semantics: dict[str, str]) -> list[dict[str, Any]]:
    vertices: list[VEC3] = []
    parts: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                _, xs, ys, zs = line.split()[:4]
                vertices.append((float(xs), float(ys), float(zs)))
                continue

            if line.startswith("o "):
                current = {
                    "object_name": line[2:].strip(),
                    "vertex_indices": set(),
                    "materials": [],
                    "face_count": 0,
                }
                parts.append(current)
                continue

            if current is None:
                continue

            if line.startswith("usemtl "):
                material_id = line.split(maxsplit=1)[1].strip()
                current["materials"].append(material_id)
                continue

            if line.startswith("f "):
                current["face_count"] += 1
                for token in line.split()[1:]:
                    current["vertex_indices"].add(parse_obj_index(token, len(vertices)))

    for part in parts:
        pts = [vertices[i] for i in sorted(part["vertex_indices"])]
        mins = [min(p[k] for p in pts) for k in range(3)]
        maxs = [max(p[k] for p in pts) for k in range(3)]
        center = [(mins[k] + maxs[k]) * 0.5 for k in range(3)]
        size = [maxs[k] - mins[k] for k in range(3)]

        material_ids = sorted(set(part["materials"]))
        material_labels = sorted({material_semantics.get(mid, mid) for mid in material_ids})

        part["bbox_min_xyz"] = mins
        part["bbox_max_xyz"] = maxs
        part["center_xyz"] = center
        part["size_xyz"] = size
        part["material_ids"] = material_ids
        part["material_labels"] = material_labels
        part["vertex_count"] = len(part["vertex_indices"])
        part["local_name"] = strip_vehicle_prefix(part["object_name"])
        del part["vertex_indices"]
        del part["materials"]

    return parts


def strip_vehicle_prefix(name: str) -> str:
    return name.split("#", 1)[1] if "#" in name else name


def side_name(code: str) -> str:
    return {"l": "left", "r": "right"}[code]


def radius_from_bbox(size_xyz: list[float]) -> float:
    return 0.25 * (size_xyz[1] + size_xyz[2])


def bbox_union(parts: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float], list[float]]:
    mins = [min(part["bbox_min_xyz"][k] for part in parts) for k in range(3)]
    maxs = [max(part["bbox_max_xyz"][k] for part in parts) for k in range(3)]
    center = [(mins[k] + maxs[k]) * 0.5 for k in range(3)]
    size = [maxs[k] - mins[k] for k in range(3)]
    return mins, maxs, center, size


def classify_part(part: dict[str, Any]) -> dict[str, Any]:
    local = part["local_name"]
    cy = part["center_xyz"][1]

    if re.fullmatch(r"root_\d+", local):
        return {
            "category": "hull_main",
            "body_name": "hull",
            "body_role": "hull",
            "parent_body": None,
            "joint_type": "free_base",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 1.0,
        }

    if match := re.fullmatch(r"bone_turret_\d+", local):
        return {
            "category": "turret_core",
            "body_name": "turret",
            "body_role": "turret",
            "parent_body": "hull",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 1.0,
        }

    if re.fullmatch(r"bone_gun_\d+", local):
        return {
            "category": "gun_cradle",
            "body_name": "gun",
            "body_role": "gun",
            "parent_body": "turret",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 1.0,
        }

    if re.fullmatch(r"gun_barrel(?:_\d+)+", local):
        return {
            "category": "gun_barrel",
            "body_name": "gun",
            "body_role": "gun",
            "parent_body": "turret",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 1.0,
        }

    if re.fullmatch(r"bone_mg_aa_h_01_\d+", local):
        return {
            "category": "aa_mg_traverse_core",
            "body_name": "aa_mg_traverse",
            "body_role": "aa_mg_traverse",
            "parent_body": "turret",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.98,
        }

    if re.fullmatch(r"bone_mg_aa_v_01_\d+", local):
        return {
            "category": "aa_mg_elevation_core",
            "body_name": "aa_mg_elevation",
            "body_role": "aa_mg_elevation",
            "parent_body": "aa_mg_traverse",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.98,
        }

    if re.fullmatch(r"mg_aa_mount_h_\d+", local):
        return {
            "category": "aa_mg_mount",
            "body_name": "aa_mg_traverse",
            "body_role": "aa_mg_traverse",
            "parent_body": "turret",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.90,
        }

    if re.fullmatch(r"antenna(?:_\d+)+", local):
        return {
            "category": "antenna",
            "body_name": "turret",
            "body_role": "turret",
            "parent_body": "turret",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.85,
        }

    if match := re.fullmatch(r"track_([lr])_\d+", local):
        side = side_name(match.group(1))
        return {
            "category": "track_loop_visual",
            "body_name": f"{side}_track",
            "body_role": "track_loop",
            "parent_body": "hull",
            "joint_type": "track_loop",
            "joint_axis_xyz": [1.0, 0.0, 0.0],
            "side": side,
            "station": None,
            "classification_confidence": 1.0,
        }

    if match := re.fullmatch(r"wheel_([lr])_drive_\d+", local):
        side = side_name(match.group(1))
        return {
            "category": "drive_sprocket",
            "body_name": f"{side}_drive_sprocket",
            "body_role": "drive_sprocket",
            "parent_body": "hull",
            "joint_type": "revolute",
            "joint_axis_xyz": [1.0, 0.0, 0.0],
            "side": side,
            "station": 0,
            "classification_confidence": 1.0,
        }

    if match := re.fullmatch(r"wheel_([lr])_(?:front|back)_\d+", local):
        side = side_name(match.group(1))
        return {
            "category": "idler",
            "body_name": f"{side}_idler",
            "body_role": "idler",
            "parent_body": "hull",
            "joint_type": "revolute",
            "joint_axis_xyz": [1.0, 0.0, 0.0],
            "side": side,
            "station": 0,
            "classification_confidence": 1.0,
        }

    if match := re.fullmatch(r"wheel_([lr])_top_(\d{2})_\d+", local):
        side = side_name(match.group(1))
        station = int(match.group(2))
        return {
            "category": "return_roller",
            "body_name": f"{side}_return_roller_{station:02d}",
            "body_role": "return_roller",
            "parent_body": "hull",
            "joint_type": "revolute",
            "joint_axis_xyz": [1.0, 0.0, 0.0],
            "side": side,
            "station": station,
            "classification_confidence": 1.0,
        }

    if match := re.fullmatch(r"wheel_([lr])_(\d{2})_\d+", local):
        side = side_name(match.group(1))
        station = int(match.group(2))
        return {
            "category": "road_wheel",
            "body_name": f"{side}_road_wheel_{station:02d}",
            "body_role": "road_wheel",
            "parent_body": f"{side}_suspension_station_{station:02d}",
            "joint_type": "revolute",
            "joint_axis_xyz": [1.0, 0.0, 0.0],
            "side": side,
            "station": station,
            "classification_confidence": 1.0,
        }

    if match := re.fullmatch(r"suspension_([lr])_(\d{2})(?:_(\d{2}))?_\d+", local):
        side = side_name(match.group(1))
        station = int(match.group(2))
        subindex = match.group(3)
        category = "suspension_primary" if subindex is None else f"suspension_aux_{subindex}"
        return {
            "category": category,
            "body_name": f"{side}_suspension_station_{station:02d}",
            "body_role": "suspension_station",
            "parent_body": "hull",
            "joint_type": "revolute",
            "joint_axis_xyz": [1.0, 0.0, 0.0],
            "side": side,
            "station": station,
            "classification_confidence": 0.96,
        }

    if re.fullmatch(r"ex_armor_turret_\d{2}_\d+", local):
        return {
            "category": "turret_armor",
            "body_name": "turret",
            "body_role": "turret",
            "parent_body": "turret",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.98,
        }

    if local.startswith("ex_armor_body_") or re.fullmatch(r"ex_armor_\d{2}_\d+", local):
        return {
            "category": "hull_armor",
            "body_name": "hull",
            "body_role": "hull",
            "parent_body": "hull",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.88,
        }

    if re.fullmatch(r"hatch_01_\d+", local):
        return {
            "category": "driver_hatch",
            "body_name": "hull",
            "body_role": "hull",
            "parent_body": "hull",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.70,
        }

    if re.fullmatch(r"hatch_\d{2}_\d+", local):
        return {
            "category": "turret_hatch",
            "body_name": "turret",
            "body_role": "turret",
            "parent_body": "turret",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.75,
        }

    if re.fullmatch(r"ex_lantern(?:_b)?_\d{2}_\d+", local):
        target = "turret" if cy > 1.8 else "hull"
        role = "turret" if target == "turret" else "hull"
        return {
            "category": "lantern",
            "body_name": target,
            "body_role": role,
            "parent_body": target,
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.72,
        }

    if local.startswith("ex_decor_") or local.startswith("ex_decor_l_") or local.startswith("ex_decor_r_"):
        target = "turret" if cy > 1.65 else "hull"
        role = "turret" if target == "turret" else "hull"
        return {
            "category": "stowage_or_decor",
            "body_name": target,
            "body_role": role,
            "parent_body": target,
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.65,
        }

    if re.fullmatch(r"bone_mg_gun_twin_\d+", local):
        return {
            "category": "coaxial_mg_core",
            "body_name": "gun",
            "body_role": "gun",
            "parent_body": "turret",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.82,
        }

    if re.fullmatch(r"bone_commander_sight_h_\d+", local):
        return {
            "category": "commander_sight",
            "body_name": "turret",
            "body_role": "turret",
            "parent_body": "turret",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.76,
        }

    if re.fullmatch(r"breakwater(?:_\d{2})?_\d+", local):
        return {
            "category": "hull_breakwater",
            "body_name": "hull",
            "body_role": "hull",
            "parent_body": "hull",
            "joint_type": "fixed",
            "joint_axis_xyz": None,
            "side": None,
            "station": None,
            "classification_confidence": 0.78,
        }

    return {
        "category": "unclassified",
        "body_name": "unclassified",
        "body_role": "unclassified",
        "parent_body": None,
        "joint_type": "unknown",
        "joint_axis_xyz": None,
        "side": None,
        "station": None,
        "classification_confidence": 0.0,
    }


def attach_classification(parts: list[dict[str, Any]]) -> None:
    for part in parts:
        part.update(classify_part(part))


def part_by_category(parts: list[dict[str, Any]], category: str) -> dict[str, Any] | None:
    for part in parts:
        if part["category"] == category:
            return part
    return None


def body_joint_origin(body_name: str, parts: list[dict[str, Any]], hull_part: dict[str, Any] | None) -> list[float]:
    primary = part_by_category(parts, "hull_main") or parts[0]

    if body_name == "hull":
        return [round(v, 6) for v in primary["center_xyz"]]

    if body_name == "turret":
        core = part_by_category(parts, "turret_core") or primary
        y_origin = hull_part["bbox_max_xyz"][1] if hull_part else core["bbox_min_xyz"][1]
        return [0.0, round(y_origin, 6), round(core["center_xyz"][2], 6)]

    if body_name == "gun":
        cradle = part_by_category(parts, "gun_cradle") or primary
        barrel = part_by_category(parts, "gun_barrel")
        z_origin = cradle["bbox_max_xyz"][2]
        if barrel:
            z_origin = min(z_origin, barrel["bbox_min_xyz"][2]) - 0.02
        return [0.0, round(cradle["center_xyz"][1], 6), round(z_origin, 6)]

    if body_name == "aa_mg_traverse":
        core = part_by_category(parts, "aa_mg_traverse_core") or primary
        return [round(v, 6) for v in core["center_xyz"]]

    if body_name == "aa_mg_elevation":
        core = part_by_category(parts, "aa_mg_elevation_core") or primary
        return [round(v, 6) for v in core["center_xyz"]]

    return [round(v, 6) for v in primary["center_xyz"]]


def representative_body_part(body_name: str, body_parts: list[dict[str, Any]]) -> dict[str, Any]:
    preferred_categories = {
        "hull": ("hull_main",),
        "turret": ("turret_core",),
        "gun": ("gun_cradle", "gun_barrel", "coaxial_mg_core"),
        "aa_mg_traverse": ("aa_mg_traverse_core", "aa_mg_mount"),
        "aa_mg_elevation": ("aa_mg_elevation_core",),
    }
    for category in preferred_categories.get(body_name, ()):
        part = part_by_category(body_parts, category)
        if part:
            return part

    if body_parts and body_parts[0]["body_role"] == "suspension_station":
        part = part_by_category(body_parts, "suspension_primary")
        if part:
            return part

    return body_parts[0]


def build_body_specs(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_body: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for part in parts:
        by_body[part["body_name"]].append(part)

    hull_part = part_by_category(parts, "hull_main")
    bodies: list[dict[str, Any]] = []

    for body_name, body_parts in sorted(by_body.items()):
        if body_name == "unclassified":
            continue

        mins, maxs, center, size = bbox_union(body_parts)
        roles = Counter(part["body_role"] for part in body_parts)
        role = roles.most_common(1)[0][0]
        representative = representative_body_part(body_name, body_parts)
        parent_body = representative["parent_body"]
        joint_type = representative["joint_type"]
        joint_axis = representative["joint_axis_xyz"]
        side = representative["side"]
        station = representative["station"]
        origin = body_joint_origin(body_name, body_parts, hull_part)

        notes: list[str] = []
        if role == "track_loop":
            notes.append("Visual track loop detected as one closed mesh; individual links are not separable in the source OBJ.")
        if role == "suspension_station":
            notes.append("Multiple suspension subparts were merged into one station assembly body.")
        if body_name in {"hull", "turret"}:
            notes.append("Fixed armor, lantern, hatch, and decor meshes were welded into this primary rigid body.")

        materials = sorted({label for part in body_parts for label in part["material_labels"]})
        bodies.append(
            {
                "name": body_name,
                "body_role": role,
                "parent_body": parent_body,
                "side": side,
                "station": station,
                "joint": {
                    "type": joint_type,
                    "axis_xyz": joint_axis,
                    "origin_xyz": origin,
                    "origin_rule": "name_and_bbox_heuristic",
                },
                "bbox_min_xyz": [round(v, 6) for v in mins],
                "bbox_max_xyz": [round(v, 6) for v in maxs],
                "center_xyz": [round(v, 6) for v in center],
                "size_xyz": [round(v, 6) for v in size],
                "visual_part_count": len(body_parts),
                "visual_parts": [part["object_name"] for part in body_parts],
                "materials": materials,
                "notes": notes,
            }
        )

    return bodies


def infer_simplified_measurements(parts: list[dict[str, Any]]) -> dict[str, Any]:
    hull = part_by_category(parts, "hull_main")
    tracks_by_side = {
        part["side"]: part
        for part in parts
        if part["category"] == "track_loop_visual" and part["side"] in {"left", "right"}
    }
    left_track = tracks_by_side.get("left")
    right_track = tracks_by_side.get("right")

    road_wheels = [part for part in parts if part["category"] == "road_wheel"]
    return_rollers = [part for part in parts if part["category"] == "return_roller"]
    drive = [part for part in parts if part["category"] == "drive_sprocket"]
    idlers = [part for part in parts if part["category"] == "idler"]

    left_road = sorted(
        [part for part in road_wheels if part["side"] == "left"],
        key=lambda part: part["station"] or 0,
    )
    right_road = sorted(
        [part for part in road_wheels if part["side"] == "right"],
        key=lambda part: part["station"] or 0,
    )

    return {
        "coordinate_convention": {
            "x_axis": "left_positive_right_negative_lateral",
            "y_axis": "up",
            "z_axis": "forward",
        },
        "hull_bbox_size_m": [round(v, 6) for v in (hull["size_xyz"] if hull else [0.0, 0.0, 0.0])],
        "track_center_x_m": {
            "left": round(left_track["center_xyz"][0], 6) if left_track else None,
            "right": round(right_track["center_xyz"][0], 6) if right_track else None,
        },
        "track_width_m": {
            "left": round(left_track["size_xyz"][0], 6) if left_track else None,
            "right": round(right_track["size_xyz"][0], 6) if right_track else None,
        },
        "track_loop_lateral_span_m": {
            "left": round(left_track["size_xyz"][0], 6) if left_track else None,
            "right": round(right_track["size_xyz"][0], 6) if right_track else None,
        },
        "road_wheel_radius_m": round(sum(radius_from_bbox(part["size_xyz"]) for part in road_wheels) / max(len(road_wheels), 1), 6),
        "return_roller_radius_m": round(
            sum(radius_from_bbox(part["size_xyz"]) for part in return_rollers) / max(len(return_rollers), 1), 6
        ),
        "drive_sprocket_radius_m": round(sum(radius_from_bbox(part["size_xyz"]) for part in drive) / max(len(drive), 1), 6),
        "idler_radius_m": round(sum(radius_from_bbox(part["size_xyz"]) for part in idlers) / max(len(idlers), 1), 6),
        "left_road_wheel_z_stations_m": [round(part["center_xyz"][2], 6) for part in left_road],
        "right_road_wheel_z_stations_m": [round(part["center_xyz"][2], 6) for part in right_road],
    }


def build_track_systems(bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    body_map = {body["name"]: body for body in bodies}
    systems: list[dict[str, Any]] = []

    for side in ("left", "right"):
        track_name = f"{side}_track"
        drive_name = f"{side}_drive_sprocket"
        idler_name = f"{side}_idler"
        if track_name not in body_map or drive_name not in body_map or idler_name not in body_map:
            continue

        systems.append(
            {
                "name": track_name,
                "side": side,
                "visual_body": track_name,
                "drive_sprocket_body": drive_name,
                "idler_body": idler_name,
                "return_roller_bodies": [
                    body["name"]
                    for body in bodies
                    if body["body_role"] == "return_roller" and body["side"] == side
                ],
                "road_wheel_bodies": [
                    body["name"]
                    for body in bodies
                    if body["body_role"] == "road_wheel" and body["side"] == side
                ],
                "suspension_bodies": [
                    body["name"]
                    for body in bodies
                    if body["body_role"] == "suspension_station" and body["side"] == side
                ],
                "track_phase_origin_xyz": body_map[drive_name]["joint"]["origin_xyz"],
                "notes": [
                    "Track is represented as one closed mesh body plus supporting wheel references.",
                    "Individual track shoes are not separately named in the source model.",
                ],
            }
        )

    return systems


def analyze_track_meshes(obj_path: Path, target_names: set[str] | None = None) -> dict[str, Any]:
    vertices: list[VEC3] = []
    track_faces: dict[str, list[list[int]]] = {}
    current: str | None = None
    targets = set(target_names or ())

    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                _, xs, ys, zs = line.split()[:4]
                vertices.append((float(xs), float(ys), float(zs)))
                continue

            if line.startswith("o "):
                current = line[2:].strip()
                is_track_mesh = current in targets if targets else bool(
                    re.fullmatch(r"track_[lr]_\d+", strip_vehicle_prefix(current))
                )
                if is_track_mesh:
                    track_faces[current] = []
                continue

            if current in track_faces and line.startswith("f "):
                track_faces[current].append([parse_obj_index(token, len(vertices)) for token in line.split()[1:]])

    analysis: dict[str, Any] = {}

    for object_name, faces in sorted(track_faces.items()):
        vertex_to_faces: dict[int, list[int]] = defaultdict(list)
        for face_index, face in enumerate(faces):
            for vertex_index in face:
                vertex_to_faces[vertex_index].append(face_index)

        seen: set[int] = set()
        component_sizes: list[list[float]] = []
        component_faces: list[int] = []

        for start in range(len(faces)):
            if start in seen:
                continue

            queue = [start]
            seen.add(start)
            component_vertices: set[int] = set()
            face_count = 0

            while queue:
                face_index = queue.pop()
                face_count += 1
                for vertex_index in faces[face_index]:
                    component_vertices.add(vertex_index)
                    for neighbor in vertex_to_faces[vertex_index]:
                        if neighbor not in seen:
                            seen.add(neighbor)
                            queue.append(neighbor)

            pts = [vertices[i] for i in component_vertices]
            mins = [min(p[k] for p in pts) for k in range(3)]
            maxs = [max(p[k] for p in pts) for k in range(3)]
            component_sizes.append([maxs[k] - mins[k] for k in range(3)])
            component_faces.append(face_count)

        x_sizes = sorted(size[0] for size in component_sizes)
        y_sizes = sorted(size[1] for size in component_sizes)
        z_sizes = sorted(size[2] for size in component_sizes)
        strip_like = sum(1 for size in component_sizes if size[0] < 0.05 and size[2] > 1.0)

        analysis[object_name] = {
            "connected_component_count": len(component_sizes),
            "component_face_count_median": median(component_faces) if component_faces else 0,
            "component_bbox_size_median_xyz": [
                round(median(x_sizes), 6) if x_sizes else 0.0,
                round(median(y_sizes), 6) if y_sizes else 0.0,
                round(median(z_sizes), 6) if z_sizes else 0.0,
            ],
            "strip_like_component_count": strip_like,
            "likely_individual_shoe_meshes": strip_like == 0 and len(component_sizes) > 20,
            "interpretation": (
                "Connected components are mostly long, thin surface strips rather than clean one-shoe solids."
                if strip_like > 0
                else "Connected components may be usable as shoe candidates, but should still be checked visually."
            ),
        }

    return analysis


def write_inventory_csv(csv_path: Path, parts: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "object_name",
        "local_name",
        "category",
        "body_name",
        "body_role",
        "parent_body",
        "joint_type",
        "side",
        "station",
        "classification_confidence",
        "face_count",
        "vertex_count",
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "materials",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for part in parts:
            writer.writerow(
                {
                    "object_name": part["object_name"],
                    "local_name": part["local_name"],
                    "category": part["category"],
                    "body_name": part["body_name"],
                    "body_role": part["body_role"],
                    "parent_body": part["parent_body"],
                    "joint_type": part["joint_type"],
                    "side": part["side"],
                    "station": part["station"],
                    "classification_confidence": part["classification_confidence"],
                    "face_count": part["face_count"],
                    "vertex_count": part["vertex_count"],
                    "center_x": round(part["center_xyz"][0], 6),
                    "center_y": round(part["center_xyz"][1], 6),
                    "center_z": round(part["center_xyz"][2], 6),
                    "size_x": round(part["size_xyz"][0], 6),
                    "size_y": round(part["size_xyz"][1], 6),
                    "size_z": round(part["size_xyz"][2], 6),
                    "materials": ";".join(part["material_labels"]),
                }
            )


def write_report(report_path: Path, data: dict[str, Any]) -> None:
    category_counts = data["summary"]["category_counts"]
    body_role_counts = data["summary"]["body_role_counts"]
    measurements = data["simplified_measurements"]
    track_mesh_analysis = data["track_mesh_analysis"]
    vehicle_specs = data.get("vehicle_specifications", {})
    spec_sources = vehicle_specs.get("sources", [])

    lines = [
        f"# {data.get('model_name', 'Tracked Vehicle')} Multibody Conversion Report",
        "",
        "## Summary",
        "",
        f"- Source OBJ: `{data['source_obj']}`",
        f"- Visual objects parsed: `{data['summary']['visual_object_count']}`",
        f"- Dynamic bodies generated: `{data['summary']['body_count']}`",
        f"- Track assemblies detected: `{data['summary']['track_system_count']}`",
        "",
        "## Reference Vehicle Parameters",
        "",
    ]

    reference_fields = [
        ("mass_kg", "Mass (kg)"),
        ("nominal_length_m", "Overall length (m)"),
        ("nominal_width_m", "Overall width (m)"),
        ("nominal_height_m", "Overall height (m)"),
        ("engine_power_kw", "Engine power (kW)"),
        ("max_road_speed_mps", "Maximum road speed (m/s)"),
        ("range_km", "Range (km)"),
        ("crew", "Crew"),
        ("passengers", "Passengers"),
    ]
    wrote_reference = False
    for key, label in reference_fields:
        if vehicle_specs.get(key) is None:
            continue
        lines.append(f"- {label}: `{vehicle_specs[key]}`")
        wrote_reference = True
    if not wrote_reference:
        lines.append("- No external vehicle parameters were supplied to the converter.")
    for source in spec_sources:
        lines.append(f"- Source: {source}")

    lines += [
        "",
        "## Identified Major Components",
        "",
        f"- Hull-related meshes: `{category_counts.get('hull_main', 0) + category_counts.get('hull_armor', 0)}`",
        f"- Turret-related meshes: `{category_counts.get('turret_core', 0) + category_counts.get('turret_armor', 0)}`",
        f"- Gun meshes: `{category_counts.get('gun_cradle', 0) + category_counts.get('gun_barrel', 0)}`",
        f"- Road wheels: `{category_counts.get('road_wheel', 0)}`",
        f"- Drive sprockets: `{category_counts.get('drive_sprocket', 0)}`",
        f"- Idlers: `{category_counts.get('idler', 0)}`",
        f"- Return rollers: `{category_counts.get('return_roller', 0)}`",
        f"- Suspension station meshes: `{category_counts.get('suspension_primary', 0) + category_counts.get('suspension_aux_01', 0) + category_counts.get('suspension_aux_02', 0)}`",
        f"- Track loop meshes: `{category_counts.get('track_loop_visual', 0)}`",
        "",
        "## Simplified Geometry Measurements",
        "",
        f"- Hull bounding-box size (m): `{measurements['hull_bbox_size_m']}`",
        f"- Left/right track centers along X (m): `{measurements['track_center_x_m']}`",
        f"- Left/right track-loop lateral spans (m): `{measurements['track_loop_lateral_span_m']}`",
        f"- Mean road wheel radius (m): `{measurements['road_wheel_radius_m']}`",
        f"- Mean return roller radius (m): `{measurements['return_roller_radius_m']}`",
        f"- Mean drive sprocket radius (m): `{measurements['drive_sprocket_radius_m']}`",
        f"- Mean idler radius (m): `{measurements['idler_radius_m']}`",
        "",
        "## Track Mesh Analysis",
        "",
    ]

    for object_name, summary in sorted(track_mesh_analysis.items()):
        lines.extend(
            [
                f"- `{object_name}` connected components: `{summary['connected_component_count']}`",
                f"- `{object_name}` median component bbox (m): `{summary['component_bbox_size_median_xyz']}`",
                f"- `{object_name}` strip-like components: `{summary['strip_like_component_count']}`",
                f"- `{object_name}` interpretation: {summary['interpretation']}",
                "",
            ]
        )

    lines += [
        "## Multibody Bodies by Role",
        "",
    ]

    for role, count in sorted(body_role_counts.items()):
        lines.append(f"- `{role}`: `{count}`")

    lines += [
        "",
        "## Assumptions",
        "",
        "- Hatches, lanterns, stowage, and armor add-ons were welded to the hull or turret unless the source naming clearly implied a joint.",
        "- The source OBJ contains one left track mesh and one right track mesh; the internal connected components behave like surface strips, not clean one-shoe bodies.",
        "- Joint origins were inferred from part names plus bounding boxes, which is suitable for skeleton generation but should be refined before high-fidelity dynamics calibration.",
        "- Mass and inertia tensors are not identifiable from the visual mesh alone and are intentionally left for downstream calibration.",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    obj_path = args.obj.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_prefix_from_model_name(args.model_name, args.output_prefix)

    material_semantics = load_material_semantics(obj_path.with_suffix(".mtl"))
    parts = load_obj_parts(obj_path, material_semantics)
    attach_classification(parts)
    bodies = build_body_specs(parts)
    track_systems = build_track_systems(bodies)
    track_object_names = {
        part["object_name"]
        for part in parts
        if part["category"] == "track_loop_visual"
    }
    track_mesh_analysis = analyze_track_meshes(obj_path, track_object_names)
    measurements = infer_simplified_measurements(parts)

    category_counts = Counter(part["category"] for part in parts)
    body_role_counts = Counter(body["body_role"] for body in bodies)
    has_reference_values = any(
        value is not None
        for value in (
            args.mass_kg,
            args.nominal_length_m,
            args.nominal_width_m,
            args.nominal_height_m,
            args.engine_power_kw,
            args.max_road_speed_mps,
            args.range_km,
            args.crew,
            args.passengers,
        )
    )
    vehicle_specs = {
        "status": (
            "open_source_reference"
            if args.spec_source
            else "user_supplied_reference"
            if has_reference_values
            else "not_supplied"
        ),
        "mass_kg": args.mass_kg,
        "nominal_length_m": args.nominal_length_m,
        "nominal_width_m": args.nominal_width_m,
        "nominal_height_m": args.nominal_height_m,
        "engine_power_kw": args.engine_power_kw,
        "max_road_speed_mps": args.max_road_speed_mps,
        "range_km": args.range_km,
        "crew": args.crew,
        "passengers": args.passengers,
        "sources": args.spec_source,
    }

    model = {
        "source_obj": str(obj_path),
        "source_mtl": str(obj_path.with_suffix(".mtl")),
        "model_name": args.model_name,
        "coordinate_system": measurements["coordinate_convention"],
        "summary": {
            "visual_object_count": len(parts),
            "body_count": len(bodies),
            "track_system_count": len(track_systems),
            "category_counts": dict(sorted(category_counts.items())),
            "body_role_counts": dict(sorted(body_role_counts.items())),
        },
        "mass_properties": {
            "status": "open_source_reference" if args.mass_kg is not None else "requires_calibration",
            "mass_kg": args.mass_kg,
            "reason": "Visual meshes do not provide trustworthy material thickness, density, or internal layout.",
        },
        "vehicle_specifications": vehicle_specs,
        "simplified_measurements": measurements,
        "calibrated_geometry": {
            "track_width_m": args.track_width_m,
            "track_pitch_m": args.track_pitch_m,
            "shoe_length_m": args.shoe_length_m,
            "notes": "Optional manual overrides consumed by the runtime model builder.",
        },
        "track_mesh_analysis": track_mesh_analysis,
        "visual_parts": parts,
        "bodies": bodies,
        "track_systems": track_systems,
    }

    json_path = out_dir / f"{output_prefix}_multibody_model.json"
    csv_path = out_dir / f"{output_prefix}_part_inventory.csv"
    report_path = out_dir / f"{output_prefix}_multibody_report.md"

    json_path.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    write_inventory_csv(csv_path, parts)
    write_report(report_path, model)

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {report_path}")
    print(f"[Info] visual_objects={len(parts)} bodies={len(bodies)} track_systems={len(track_systems)}")


if __name__ == "__main__":
    main()
