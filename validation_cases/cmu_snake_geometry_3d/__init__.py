"""Geometry and contact-topology preview for the 17-body CMU snake robot."""

from .geometry import (
    SnakeGeometryConfig,
    SnakePose,
    build_contact_envelope,
    build_contact_patches,
    build_module_surfaces,
    build_pose,
    build_topology_lines,
)

__all__ = [
    "SnakeGeometryConfig",
    "SnakePose",
    "build_contact_envelope",
    "build_contact_patches",
    "build_module_surfaces",
    "build_pose",
    "build_topology_lines",
]
