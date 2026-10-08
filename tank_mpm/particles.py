import numpy as np

def sample_soil_particles(nx, ny, nz, lo, hi, jitter_ratio=0.2, seed=42):
    lo = np.array(lo, dtype=np.float64)
    hi = np.array(hi, dtype=np.float64)
    cell = (hi - lo) / np.array([nx, ny, nz], dtype=np.float64)

    xs = np.linspace(lo[0], hi[0], nx, endpoint=False) + 0.5 * cell[0]
    ys = np.linspace(lo[1], hi[1], ny, endpoint=False) + 0.5 * cell[1]
    zs = np.linspace(lo[2], hi[2], nz, endpoint=False) + 0.5 * cell[2]

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    rng = np.random.default_rng(seed)
    jitter = (rng.random(pts.shape) - 0.5) * (cell * jitter_ratio)
    pts = pts + jitter

    pts = np.clip(pts, lo + 0.05 * cell, hi - 0.05 * cell)
    p_vol = float(cell[0] * cell[1] * cell[2])

    return pts.astype(np.float32), p_vol


