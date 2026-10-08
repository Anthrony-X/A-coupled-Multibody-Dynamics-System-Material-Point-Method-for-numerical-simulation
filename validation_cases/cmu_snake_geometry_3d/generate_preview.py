from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .geometry import (
    JOINT_PITCH,
    JOINT_YAW,
    SnakeGeometryConfig,
    build_contact_envelope,
    build_contact_patches,
    build_ground_plane,
    build_module_surfaces,
    build_pose,
    build_topology_lines,
    contact_patch_summary,
    watertight_edge_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ParaView geometry/contact previews for the 17-body CMU snake robot."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output") / "cmu_snake_geometry_preview",
        help="Output directory for VTK, PVD and manifest files.",
    )
    parser.add_argument("--frames", type=int, default=12, help="Number of sidewinding shape-cycle frames.")
    parser.add_argument("--circumferential-divisions", type=int, default=16)
    parser.add_argument("--axial-patch-divisions", type=int, default=3)
    return parser.parse_args()


def _write_pvd(path: Path, datasets: list[tuple[float, Path]]) -> None:
    root = ET.Element(
        "VTKFile",
        attrib={"type": "Collection", "version": "0.1", "byte_order": "LittleEndian"},
    )
    collection = ET.SubElement(root, "Collection")
    for timestep, dataset_path in datasets:
        ET.SubElement(
            collection,
            "DataSet",
            attrib={
                "timestep": f"{timestep:.9g}",
                "group": "",
                "part": "0",
                "file": dataset_path.relative_to(path.parent).as_posix(),
            },
        )
    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_pose_files(
    config: SnakeGeometryConfig,
    pose,
    directory: Path,
    stem: str,
) -> dict[str, Path]:
    files = {
        "modules": directory / f"{stem}_modules.vtk",
        "topology": directory / f"{stem}_topology.vtk",
        "contact_envelope": directory / f"{stem}_contact_envelope.vtk",
        "contact_patches": directory / f"{stem}_contact_patches.vtk",
        "ground": directory / f"{stem}_ground.vtk",
    }
    build_module_surfaces(config, pose).write_legacy_vtk(files["modules"], "CMU snake: 17 visual rigid modules")
    build_topology_lines(config, pose).write_legacy_vtk(files["topology"], "CMU snake: 17 bodies and 16 revolute joints")
    build_contact_envelope(config, pose).write_legacy_vtk(files["contact_envelope"], "CMU snake: watertight MPM contact envelope")
    build_contact_patches(config, pose).write_legacy_vtk(files["contact_patches"], "CMU snake: MPM rectangular contact patches")
    build_ground_plane(config, pose).write_legacy_vtk(files["ground"], "Reference sand surface at z=0")
    return files


def generate(out_dir: Path, config: SnakeGeometryConfig, frame_count: int) -> dict:
    if frame_count < 2:
        raise ValueError("frames must be at least 2")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    straight_pose = build_pose(config, straight=True)
    straight_files = _write_pose_files(config, straight_pose, out_dir, "straight")

    sidewinding_pose = build_pose(config, phase_rad=0.0)
    sidewinding_files = _write_pose_files(config, sidewinding_pose, out_dir, "sidewinding_phase_000")

    frame_dir = out_dir / "frames"
    module_datasets: list[tuple[float, Path]] = []
    envelope_datasets: list[tuple[float, Path]] = []
    patch_datasets: list[tuple[float, Path]] = []
    topology_datasets: list[tuple[float, Path]] = []
    for frame_id in range(frame_count):
        phase = 2.0 * math.pi * frame_id / frame_count
        pose = build_pose(config, phase_rad=phase)
        stem = f"frame_{frame_id:03d}"
        files = _write_pose_files(config, pose, frame_dir, stem)
        module_datasets.append((phase, files["modules"]))
        envelope_datasets.append((phase, files["contact_envelope"]))
        patch_datasets.append((phase, files["contact_patches"]))
        topology_datasets.append((phase, files["topology"]))

    pvd_files = {
        "modules": out_dir / "snake_shape_cycle_modules.pvd",
        "contact_envelope": out_dir / "snake_shape_cycle_contact_envelope.pvd",
        "contact_patches": out_dir / "snake_shape_cycle_contact_patches.pvd",
        "topology": out_dir / "snake_shape_cycle_topology.pvd",
    }
    _write_pvd(pvd_files["modules"], module_datasets)
    _write_pvd(pvd_files["contact_envelope"], envelope_datasets)
    _write_pvd(pvd_files["contact_patches"], patch_datasets)
    _write_pvd(pvd_files["topology"], topology_datasets)

    envelope = build_contact_envelope(config, sidewinding_pose)
    patches = build_contact_patches(config, sidewinding_pose)
    boundary_edges, nonmanifold_edges = watertight_edge_counts(envelope)
    module_type_counts = {"head": 1, "middle": config.module_count - 2, "tail": 1}
    manifest = {
        "model": {
            "name": "CMU modular snake robot contact-oriented preview",
            "topology": "Head + 15 middle modules + Tail",
            "module_type_counts": module_type_counts,
            "module_count": config.module_count,
            "joint_count": config.joint_count,
            "joint_sequence": [
                "yaw" if value == JOINT_YAW else "pitch" for value in sidewinding_pose.joint_types
            ],
            "total_length_m": config.total_length_m,
            "total_mass_kg": config.total_mass_kg,
            "diameter_m": 2.0 * config.radius_m,
            "module_pitch_m": config.module_pitch_m,
            "provisional_uniform_module_mass_kg": config.module_mass_kg,
        },
        "prescribed_shape_preview": {
            "mode": "continuous_backbone_fit",
            "horizontal_wave_amplitude_m": config.horizontal_wave_amplitude_m,
            "vertical_wave_amplitude_m": config.vertical_wave_amplitude_m,
            "wavelength_m": config.wavelength_m,
            "body_wave_phase_offset_rad": config.body_wave_phase_offset_rad,
            "note": "This is a kinematic shape-cycle preview, not a solved MBD-MPM trajectory.",
        },
        "contact_discretization": contact_patch_summary(config, patches),
        "contact_envelope_topology": {
            "boundary_edge_count": boundary_edges,
            "nonmanifold_edge_count": nonmanifold_edges,
            "watertight": boundary_edges == 0 and nonmanifold_edges == 0,
        },
        "vtk_cell_field_codes": {
            "module_type": {"0": "Head", "1": "Middle", "2": "Tail"},
            "joint_type": {"0": "not a joint", "1": "Yaw (local z axis)", "2": "Pitch (local y axis)"},
            "surface_id": {"0": "mantle", "1": "head end cap", "2": "tail end cap"},
            "entity_type": {"0": "module centerline", "1": "joint axis"},
        },
        "files": {
            "straight": {key: str(value.relative_to(out_dir)) for key, value in straight_files.items()},
            "sidewinding_phase_000": {
                key: str(value.relative_to(out_dir)) for key, value in sidewinding_files.items()
            },
            "shape_cycle_pvd": {key: str(value.relative_to(out_dir)) for key, value in pvd_files.items()},
        },
        "geometry_assumptions": [
            "The published overall length is divided into 17 equal kinematic pitches.",
            "The published total mass is provisionally distributed uniformly over 17 bodies.",
            "Visual bodies are slightly inset and separated to expose joint topology.",
            "The MPM envelope uses the published approximate 50 mm diameter and is continuous across joints.",
            "The contact envelope is the numerical contact geometry; module surfaces are visualization-only.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    args = parse_args()
    config = SnakeGeometryConfig(
        circumferential_divisions=args.circumferential_divisions,
        axial_patch_divisions=args.axial_patch_divisions,
    )
    manifest = generate(args.out, config, args.frames)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
