import numpy as np


_VTK_FLOAT_DTYPE = np.dtype(">f4")
_VTK_INT_DTYPE = np.dtype(">i4")


def _write_line(handle, text: str) -> None:
    handle.write(text.encode("ascii"))
    handle.write(b"\n")


def _write_binary_array(handle, values, dtype: np.dtype) -> None:
    data = np.ascontiguousarray(values, dtype=dtype)
    data.tofile(handle)
    handle.write(b"\n")


def _validate_point_array(name: str, values, count: int, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values)
    expected = (count, *shape_tail)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    return array

def von_mises_from_sigma(sig: np.ndarray):
    sxx = sig[:, 0, 0]
    syy = sig[:, 1, 1]
    szz = sig[:, 2, 2]
    sxy = sig[:, 0, 1]
    syz = sig[:, 1, 2]
    szx = sig[:, 2, 0]
    vm2 = (
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
    )
    return np.sqrt(np.maximum(vm2, 0.0))


def write_soil_vtk(path, points, velocity, displacement, stress):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {points.shape}")
    n = points.shape[0]
    velocity = _validate_point_array("velocity", velocity, n, (3,))
    displacement = _validate_point_array("displacement", displacement, n, (3,))
    stress = _validate_point_array("stress", stress, n, (3, 3))
    vm = von_mises_from_sigma(stress)

    vertices = np.empty((n, 2), dtype=np.int32)
    vertices[:, 0] = 1
    vertices[:, 1] = np.arange(n, dtype=np.int32)

    with open(path, "wb") as handle:
        _write_line(handle, "# vtk DataFile Version 3.0")
        _write_line(handle, "Soil MPM particles")
        _write_line(handle, "BINARY")
        _write_line(handle, "DATASET POLYDATA")
        _write_line(handle, f"POINTS {n} float")
        _write_binary_array(handle, points, _VTK_FLOAT_DTYPE)

        _write_line(handle, f"VERTICES {n} {2 * n}")
        _write_binary_array(handle, vertices, _VTK_INT_DTYPE)

        _write_line(handle, f"POINT_DATA {n}")
        _write_line(handle, "VECTORS velocity float")
        _write_binary_array(handle, velocity, _VTK_FLOAT_DTYPE)

        _write_line(handle, "VECTORS displacement float")
        _write_binary_array(handle, displacement, _VTK_FLOAT_DTYPE)

        _write_line(handle, "TENSORS stress float")
        _write_binary_array(handle, stress, _VTK_FLOAT_DTYPE)

        _write_line(handle, "SCALARS von_mises float 1")
        _write_line(handle, "LOOKUP_TABLE default")
        _write_binary_array(handle, vm, _VTK_FLOAT_DTYPE)


def write_polydata_vtk(path, title: str, points, faces, cell_scalars) -> None:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {points.shape}")

    face_lists = [list(face) for face in faces]
    polygon_size = sum(len(face) + 1 for face in face_lists)
    polygons = np.empty(polygon_size, dtype=np.int32)
    cursor = 0
    for face in face_lists:
        width = len(face)
        polygons[cursor] = width
        if width:
            polygons[cursor + 1 : cursor + 1 + width] = face
        cursor += width + 1

    scalar_items = list(cell_scalars.items())
    for name, (_vtk_type, values) in scalar_items:
        if len(values) != len(face_lists):
            raise ValueError(
                f"Cell scalar {name!r} has {len(values)} values for {len(face_lists)} faces"
            )

    with open(path, "wb") as handle:
        _write_line(handle, "# vtk DataFile Version 3.0")
        _write_line(handle, title)
        _write_line(handle, "BINARY")
        _write_line(handle, "DATASET POLYDATA")
        _write_line(handle, f"POINTS {len(points)} float")
        _write_binary_array(handle, points, _VTK_FLOAT_DTYPE)

        _write_line(handle, f"POLYGONS {len(face_lists)} {polygon_size}")
        _write_binary_array(handle, polygons, _VTK_INT_DTYPE)

        _write_line(handle, f"CELL_DATA {len(face_lists)}")
        for name, (vtk_type, values) in scalar_items:
            _write_line(handle, f"SCALARS {name} {vtk_type} 1")
            _write_line(handle, "LOOKUP_TABLE default")
            if vtk_type == "int":
                _write_binary_array(handle, values, _VTK_INT_DTYPE)
            elif vtk_type == "float":
                _write_binary_array(handle, values, _VTK_FLOAT_DTYPE)
            else:
                raise ValueError(f"Unsupported VTK scalar type: {vtk_type!r}")


def write_fem_vtk(
    path,
    points,
    cells,
    velocity,
    displacement,
    stress,
    contact_force=None,
    contact_pressure=None,
    cell_kind="hex",
):
    n = points.shape[0]
    m = cells.shape[0]
    vm = von_mises_from_sigma(stress)
    if contact_force is None:
        contact_force = np.zeros((n, 3), dtype=np.float64)
    if contact_pressure is None:
        contact_pressure = np.zeros(n, dtype=np.float64)
    contact_force_mag = np.linalg.norm(contact_force, axis=1)
    cell_kind = str(cell_kind).lower()
    if cell_kind == "hex":
        vtk_cell_type = 12
        cell_width = 8
    elif cell_kind == "quad":
        vtk_cell_type = 9
        cell_width = 4
    else:
        raise ValueError(f"Unsupported FEM VTK cell_kind: {cell_kind}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FEM cylinder mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {n} float\n")
        for p in points:
            f.write(f"{p[0]:.7e} {p[1]:.7e} {p[2]:.7e}\n")

        f.write(f"CELLS {m} {m * (cell_width + 1)}\n")
        for c in cells:
            conn = [int(v) for v in c[:cell_width]]
            f.write(f"{cell_width} {' '.join(str(v) for v in conn)}\n")

        f.write(f"CELL_TYPES {m}\n")
        for _ in range(m):
            f.write(f"{vtk_cell_type}\n")

        f.write(f"\nPOINT_DATA {n}\n")

        f.write("VECTORS velocity float\n")
        for v in velocity:
            f.write(f"{v[0]:.7e} {v[1]:.7e} {v[2]:.7e}\n")

        f.write("VECTORS displacement float\n")
        for u in displacement:
            f.write(f"{u[0]:.7e} {u[1]:.7e} {u[2]:.7e}\n")

        f.write("TENSORS stress float\n")
        for s in stress:
            f.write(f"{s[0,0]:.7e} {s[0,1]:.7e} {s[0,2]:.7e}\n")
            f.write(f"{s[1,0]:.7e} {s[1,1]:.7e} {s[1,2]:.7e}\n")
            f.write(f"{s[2,0]:.7e} {s[2,1]:.7e} {s[2,2]:.7e}\n")

        f.write("SCALARS von_mises float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in vm:
            f.write(f"{val:.7e}\n")

        f.write("VECTORS contact_force float\n")
        for fc in contact_force:
            f.write(f"{fc[0]:.7e} {fc[1]:.7e} {fc[2]:.7e}\n")

        f.write("SCALARS contact_force_magnitude float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in contact_force_mag:
            f.write(f"{val:.7e}\n")

        f.write("SCALARS contact_pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in contact_pressure:
            f.write(f"{val:.7e}\n")


