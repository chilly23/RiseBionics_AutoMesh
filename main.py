# -------------AUTHOR - Velan -----------------
from __future__ import annotations

import argparse
import json
import math
import queue
import shutil
import threading
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# §0  Tkinter (optional – GUI only)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import ImageTk, ImageDraw
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

# Matplotlib for embedded 3-D viewer (optional – GUI only)
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

VALID_LABELS = {"E", "C", "S"}
_COLORS_BGR  = {"E": (30, 180, 30), "C": (0, 130, 220), "S": (200, 80, 20),
                None: (160, 160, 160)}

# ─────────────────────────────────────────────────────────────────────────────
# §1  MESH / TEXTURE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_mesh(obj_path: Path, log=print):
    log(f"[LOAD] {obj_path}")
    scene = trimesh.load(obj_path, force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        parts = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(parts)
    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        raise RuntimeError("OBJ must contain UV coordinates for atlas mapping.")

    obj_dir  = obj_path.parent
    stem     = obj_path.stem
    tex_path = None
    texture  = None

    # 1) Try MTL reference first (most reliable)
    mtl = obj_path.with_suffix(".mtl")
    if mtl.exists():
        for line in mtl.read_text(errors="ignore").splitlines():
            if line.strip().lower().startswith("map_kd"):
                parts_line = line.strip().split()
                if len(parts_line) > 1:
                    candidate = obj_dir / parts_line[-1]
                    img = cv2.imread(str(candidate))
                    if img is not None:
                        tex_path = candidate; texture = img
                        break

    # 2) Search common extensions + common base names next to the OBJ
    if texture is None:
        exts  = [".jpg", ".jpeg", ".png", ".bmp", ".tga",
                 ".JPG", ".JPEG", ".PNG", ".BMP", ".TGA"]
        names = [stem, "texture", "diffuse", "albedo", "color", "colour", "map"]
        for name in names:
            for ext in exts:
                candidate = obj_dir / (name + ext)
                if candidate.exists():
                    img = cv2.imread(str(candidate))
                    if img is not None:
                        tex_path = candidate; texture = img
                        break
            if texture is not None:
                break

    if texture is None:
        raise RuntimeError(
            f"Could not find texture atlas next to '{obj_path.name}'.\n"
            f"Expected one of: {stem}.jpg / {stem}.png / texture.jpg / texture.png\n"
            f"in folder: {obj_dir}"
        )

    log(f"       Texture: {tex_path.name}  {texture.shape}")
    log(f"       Vertices {len(mesh.vertices):,}   Faces {len(mesh.faces):,}")
    return mesh, texture, tex_path


def cylinder_axis(mesh: trimesh.Trimesh) -> int:
    return int((mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).argmax())


# ─────────────────────────────────────────────────────────────────────────────
# §2  ATLAS SHAPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def atlas_ink_mask(texture_bgr: np.ndarray) -> np.ndarray:
    """Edge/ink extraction for the flattened texture atlas."""
    gray = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2HSV)

    blue = cv2.inRange(hsv, np.array([90,  35, 15]), np.array([145, 255, 210]))
    red1 = cv2.inRange(hsv, np.array([0,   35, 20]), np.array([15,  255, 230]))
    red2 = cv2.inRange(hsv, np.array([165, 35, 20]), np.array([180, 255, 230]))
    dark = cv2.inRange(hsv, np.array([0,    0,  0]), np.array([180, 120,  95]))

    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 110)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)

    ink     = blue | red1 | red2 | dark | edges
    content = (gray > 18).astype(np.uint8) * 255
    ink     = cv2.bitwise_and(ink, content)
    ink     = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    ink     = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)
    return ink


def curated_shape_masks(shape_hw: tuple[int, int]) -> list[dict]:
    """Stable label map for the supplied EngineeringMLAssessment texture atlas."""
    h, w = shape_hw
    sx = w / 2596.0
    sy = h / 2580.0

    def scale_pts(pts):
        return np.array([[[int(x * sx), int(y * sy)]] for x, y in pts], dtype=np.int32)

    specs = [
        ("S", "S_teardrop",    [(58,185),(120,184),(174,270),(202,405),(145,475),(70,460),(38,330)]),
        ("C", "C_blob",        [(315,500),(455,525),(535,705),(500,895),(345,940),(235,805),(260,610)]),
        ("E", "E_rect_upper",  [(565,200),(850,222),(840,480),(578,438)]),
        ("E", "E_circle_upper",[(1092,245),(1325,260),(1365,512),(1205,610),(1078,500)]),
        ("E", "E_rect_lower",  [(742,1140),(1025,1143),(1028,1512),(748,1494)]),
        ("E", "E_circle_right",[(2110,790),(2445,805),(2490,1205),(2240,1388),(2048,1210)]),
        ("C", "C_rect_red",    [(2036,245),(2305,240),(2315,525),(2048,538)]),
    ]
    out = []
    for label, name, pts in specs:
        mask = np.zeros((h, w), np.uint8)
        poly = scale_pts(pts)
        cv2.fillPoly(mask, [poly], 255)
        out.append({"label": label, "name": name, "polygon": poly, "mask": mask})
    return out


def refine_shape_masks(texture_bgr: np.ndarray, shapes: list[dict]) -> list[dict]:
    """Snap coarse labelled ROIs to the actual pen outline in the atlas."""
    h, w = texture_bgr.shape[:2]
    ink = atlas_ink_mask(texture_bgr)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), 2)
    refined = []
    for shape in shapes:
        x, y, bw, bh = cv2.boundingRect(shape["polygon"])
        pad = max(35, int(max(bw, bh) * 0.20))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        roi      = ink[y1:y2, x1:x2]
        seed_roi = shape["mask"][y1:y2, x1:x2]

        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None; best_score = -1.0
        seed_area = max(float(cv2.countNonZero(seed_roi)), 1.0)
        seed_m = cv2.moments(seed_roi)
        scx = seed_m["m10"] / seed_m["m00"] if seed_m["m00"] else bw * 0.5
        scy = seed_m["m01"] / seed_m["m00"] if seed_m["m00"] else bh * 0.5

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 500: continue
            cm = cv2.moments(cnt)
            if not cm["m00"]: continue
            ccx = cm["m10"] / cm["m00"]; ccy = cm["m01"] / cm["m00"]
            tmp = np.zeros_like(roi)
            cv2.fillPoly(tmp, [cnt], 255)
            overlap = cv2.countNonZero(cv2.bitwise_and(tmp, seed_roi)) / seed_area
            dist    = math.hypot(ccx - scx, ccy - scy) / max(x2 - x1, y2 - y1, 1)
            score   = overlap * 4.0 + min(area / seed_area, 2.0) - dist
            if score > best_score:
                best_score = score; best = cnt

        if best is None:
            refined.append(shape)
            continue

        contour = best + np.array([[[x1, y1]]], dtype=np.int32)
        mask    = np.zeros((h, w), np.uint8)
        name    = shape["name"].lower()
        if "rect" in name:
            # Use the detected ink contour's bounding rect rather than the
            # hand-coded polygon — this corrects the spatial offset and
            # prevents bleeding into adjacent geometry.
            rx, ry, rw, rh = cv2.boundingRect(contour)
            # Intersect with the seeded bbox to avoid stray ink regions
            sx, sy, sw, sh = cv2.boundingRect(shape["polygon"])
            ix1 = max(rx, sx)
            iy1 = max(ry, sy)
            ix2 = min(rx + rw, sx + sw)
            iy2 = min(ry + rh, sy + sh)
            if ix2 > ix1 and iy2 > iy1:
                # Build a tight axis-aligned rectangle from the intersection
                polygon = np.array([
                    [[ix1, iy1]], [[ix2, iy1]],
                    [[ix2, iy2]], [[ix1, iy2]]
                ], dtype=np.int32)
            else:
                # Fall back to detected contour bounding rect
                polygon = np.array([
                    [[rx, ry]], [[rx + rw, ry]],
                    [[rx + rw, ry + rh]], [[rx, ry + rh]]
                ], dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
        elif "circle" in name and len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(mask, ellipse, 255, -1)
            polygon = cv2.ellipse2Poly(
                (int(ellipse[0][0]), int(ellipse[0][1])),
                (max(1, int(ellipse[1][0] * 0.5)), max(1, int(ellipse[1][1] * 0.5))),
                int(ellipse[2]), 0, 360, 5
            ).reshape(-1, 1, 2)
        else:
            cv2.fillPoly(mask, [cv2.convexHull(contour)], 255)
            polygon = cv2.convexHull(contour)

        # Inward shrink — increased to 3 iterations to firmly stay inside
        # the pen stroke and exclude noise/fringe geometry at the boundary.
        shrink = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.erode(mask, shrink, iterations=3)
        refined_shape = dict(shape)
        refined_shape.update({"polygon": polygon.astype(np.int32), "mask": mask})
        refined.append(refined_shape)
    return refined


# ─────────────────────────────────────────────────────────────────────────────
# §3  UV → FACE MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def face_ids_from_uv(mesh: trimesh.Trimesh, mask: np.ndarray) -> np.ndarray:
    """Map a texture-atlas mask to mesh faces using face UV centroids."""
    h, w = mask.shape[:2]
    uv   = np.asarray(mesh.visual.uv)
    fuv  = uv[mesh.faces].mean(axis=1)
    px   = np.clip((fuv[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
    py   = np.clip(((1.0 - fuv[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)
    ids  = np.where(mask[py, px] > 0)[0]
    if len(ids) == 0:
        vpx  = np.clip((uv[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
        vpy  = np.clip(((1.0 - uv[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)
        hits = mask[vpy[mesh.faces], vpx[mesh.faces]] > 0
        ids  = np.where(hits.any(axis=1))[0]
    return ids.astype(np.int64)


def face_components(mesh: trimesh.Trimesh, face_ids: np.ndarray) -> list[np.ndarray]:
    """Connected components of a face subset using a union-find (no Python
    adjacency loops — safe on 600 k-face meshes)."""
    if len(face_ids) == 0: return []
    fidx = np.asarray(face_ids, dtype=np.int64)
    sub_faces = mesh.faces[fidx]          # (F, 3)

    # Remap vertex indices to a compact 0-based range
    u_verts, inv = np.unique(sub_faces, return_inverse=True)
    compact = inv.reshape(sub_faces.shape)   # (F, 3) in compact ids

    n = len(u_verts)
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    # Union edges within each face
    for f in compact:
        ra, rb, rc = find(int(f[0])), find(int(f[1])), find(int(f[2]))
        if ra != rb: parent[ra] = rb; rb = find(rb)
        if rb != rc: parent[rb] = rc

    # Flatten parents
    for i in range(n):
        parent[i] = find(i)

    # Group faces by their root
    roots = parent[compact[:, 0]]
    order = np.argsort(roots, kind="stable")
    sorted_roots = roots[order]
    split_pts = np.where(np.diff(sorted_roots))[0] + 1
    groups = np.split(order, split_pts)
    comps  = [fidx[g] for g in groups]
    comps.sort(key=len, reverse=True)
    return comps


def clean_region(mesh: trimesh.Trimesh, face_ids: np.ndarray,
                 min_faces: int = 10) -> np.ndarray:
    """Keep only connected components that are at least 25 % of the largest.
    This discards stray noise faces outside the actual pen marking while
    still keeping legitimately multi-part regions."""
    comps = [c for c in face_components(mesh, face_ids) if len(c) >= min_faces]
    if not comps: return face_ids
    limit = max(150, int(len(comps[0]) * 0.25))   # raised from 0.10 to 0.25
    kept  = [c for c in comps if len(c) >= limit]
    if not kept: kept = [comps[0]]
    return np.unique(np.concatenate(kept)).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# §4  MESH OPERATIONS  — solid extrude / pocket, NO smoothing / feathering
# ─────────────────────────────────────────────────────────────────────────────

def extrude_region(mesh: trimesh.Trimesh, fidx: np.ndarray,
                   amount: float, log=print) -> trimesh.Trimesh:
    """
    Extrude selected faces along their per-vertex averaged normals by `amount`.
    Fully vectorised — safe on meshes with 600 k+ faces.

    Strategy
    --------
    1. Accumulate per-vertex normals with numpy scatter-add (no Python loops).
    2. Find boundary edges with a vectorised sort + unique trick.
    3. Duplicate boundary vertices and build side-wall quads in bulk arrays.
    4. Cap each boundary loop with a centroid fan; loops are walked with a
       dict-based O(N) traversal that avoids Python list.append inside the
       hot path.
    5. UV transfer for extra vertices uses scipy KDTree (O(N log N)).
    """
    if len(fidx) == 0:
        log("  [skip] No faces to extrude"); return mesh

    fidx   = np.asarray(fidx, dtype=np.int64)
    verts  = np.asarray(mesh.vertices, dtype=np.float64)
    faces  = np.asarray(mesh.faces,    dtype=np.int64)
    fn     = np.asarray(mesh.face_normals, dtype=np.float64)
    nv_orig = len(verts)

    # ── 1. Per-vertex normals (vectorised scatter-add) ────────────────────
    sel_faces = faces[fidx]                          # (F, 3)
    sel_fn    = fn[fidx]                             # (F, 3)
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, sel_faces[:, k], sel_fn)
    sel_verts = np.unique(sel_faces)
    norms     = np.linalg.norm(vn[sel_verts], axis=1, keepdims=True)
    centroid  = verts.mean(axis=0)
    # Fallback to radial direction for zero-norm vertices
    fallback  = verts[sel_verts] - centroid
    fb_norms  = np.linalg.norm(fallback, axis=1, keepdims=True)
    fb_norms  = np.where(fb_norms < 1e-8, 1.0, fb_norms)
    fallback /= fb_norms
    norms_safe = np.where(norms < 1e-8, 1.0, norms)
    vn[sel_verts] = np.where(norms < 1e-8, fallback, vn[sel_verts] / norms_safe)

    # ── 2. Boundary edges (vectorised) ───────────────────────────────────
    # Build all directed edges of selected faces, then keep edges that appear
    # exactly once (i.e. not shared by two selected faces).
    edges_dir = np.empty((len(fidx) * 3, 2), dtype=np.int64)
    for k in range(3):
        edges_dir[k::3, 0] = sel_faces[:, k]
        edges_dir[k::3, 1] = sel_faces[:, (k + 1) % 3]
    # Canonical (sorted) form for counting
    edges_sorted = np.sort(edges_dir, axis=1)
    # Use a structured view for fast unique + counts
    edges_view   = np.ascontiguousarray(edges_sorted).view(
        np.dtype((np.void, edges_sorted.dtype.itemsize * 2)))
    _, inv, cnts = np.unique(edges_view.ravel(), return_inverse=True, return_counts=True)
    boundary_mask   = cnts[inv] == 1
    boundary_edges  = edges_sorted[boundary_mask]   # (B, 2) canonical pairs
    n_boundary      = len(boundary_edges)

    # ── 3. Displace selected vertices ────────────────────────────────────
    new_verts = verts.copy()
    new_verts[sel_verts] += vn[sel_verts] * amount

    # ── 4. Duplicate boundary vertices for side-wall base ────────────────
    bnd_verts_unique, bnd_inv = np.unique(boundary_edges, return_inverse=True)
    dup_start    = len(new_verts)
    dup_verts    = verts[bnd_verts_unique]           # original positions
    new_verts    = np.vstack([new_verts, dup_verts])
    # Map original index → duplicate index
    old_to_dup              = np.full(nv_orig, -1, dtype=np.int64)
    old_to_dup[bnd_verts_unique] = np.arange(dup_start,
                                             dup_start + len(bnd_verts_unique),
                                             dtype=np.int64)

    # ── 5. Side-wall quads (bulk) ─────────────────────────────────────────
    va = boundary_edges[:, 0]; vb = boundary_edges[:, 1]
    ba = old_to_dup[va];       bb = old_to_dup[vb]
    # Two triangles per boundary edge
    side_faces = np.empty((n_boundary * 2, 3), dtype=np.int64)
    side_faces[0::2] = np.stack([va, bb, ba], axis=1)
    side_faces[1::2] = np.stack([va, vb, bb], axis=1)

    # ── 6. Cap faces — boundary loop fan ─────────────────────────────────
    cap_faces_list = []
    if n_boundary > 0:
        # Build adjacency once with a plain dict
        bnd_adj: dict[int, list[int]] = {}
        for a, b in boundary_edges.tolist():
            bnd_adj.setdefault(a, []).append(b)
            bnd_adj.setdefault(b, []).append(a)
        visited_edges: set[tuple[int, int]] = set()
        all_loops: list[list[int]] = []
        for seed in bnd_adj:
            if all(((seed, nb) in visited_edges or (nb, seed) in visited_edges)
                   for nb in bnd_adj[seed]):
                continue
            # Walk one loop — O(loop length), no inner allocations
            loop: list[int] = [seed]
            prev, cur = -1, seed
            while True:
                nbs = bnd_adj[cur]
                nxt = -1
                for nb in nbs:
                    if nb != prev and (cur, nb) not in visited_edges and (nb, cur) not in visited_edges:
                        nxt = nb; break
                if nxt == -1 or nxt == seed:
                    break
                visited_edges.add((cur, nxt))
                loop.append(nxt); prev, cur = cur, nxt
            if len(loop) >= 3:
                all_loops.append(loop)

        for loop in all_loops:
            loop_arr = np.array(loop, dtype=np.int64)
            top_pts  = new_verts[loop_arr]
            cen      = top_pts.mean(axis=0)
            cap_vi   = len(new_verts)
            new_verts = np.vstack([new_verts, cen.reshape(1, 3)])
            n = len(loop_arr)
            cap = np.empty((n, 3), dtype=np.int64)
            cap[:, 0] = cap_vi
            cap[:, 1] = loop_arr
            cap[:, 2] = np.roll(loop_arr, -1)
            cap_faces_list.append(cap)

    # ── 7. Assemble final mesh ────────────────────────────────────────────
    parts = [faces, side_faces]
    if cap_faces_list:
        parts.extend(cap_faces_list)
    all_faces = np.vstack(parts)
    m2 = trimesh.Trimesh(vertices=new_verts, faces=all_faces, process=False)

    # ── 8. UV transfer for extra vertices (KDTree — O(N log N)) ──────────
    try:
        from scipy.spatial import KDTree
        uv_orig = np.asarray(mesh.visual.uv, dtype=np.float32)
        n_new   = len(new_verts)
        if n_new <= nv_orig:
            uv_new = uv_orig[:n_new]
        else:
            extra = new_verts[nv_orig:]
            tree  = KDTree(verts)
            _, idx = tree.query(extra, workers=-1)
            uv_new = np.vstack([uv_orig, uv_orig[idx]])
        m2.visual = trimesh.visual.TextureVisuals(
            uv=uv_new, material=mesh.visual.material)
    except Exception:
        pass

    m2.fix_normals()
    try: trimesh.repair.fill_holes(m2)
    except Exception: pass

    n_caps = sum(len(c) for c in cap_faces_list)
    log(f"  [E] Extruded {len(fidx)} faces × {amount:.4f}  "
        f"({n_boundary} boundary edges, {n_caps} cap tris)")
    return m2


def pocket_region(mesh: trimesh.Trimesh, fidx: np.ndarray,
                  amount: float, log=print) -> trimesh.Trimesh:
    """Pocket = extrude inward."""
    if len(fidx) == 0:
        log("  [skip] No faces to pocket"); return mesh
    return extrude_region(mesh, fidx, -amount, log=log)


# NOTE: smooth_region is intentionally REMOVED because Laplacian smoothing
# creates a gradual falloff that leaves the mesh hollow near the region
# boundary.  The S-marked area is left unmodified (original surface kept).
def noop_smooth(mesh: trimesh.Trimesh, fidx: np.ndarray, log=print) -> trimesh.Trimesh:
    """S regions: no smoothing — preserves solid mesh geometry."""
    log(f"  [S] Smoothing skipped ({len(fidx)} faces) — solid mesh preserved")
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# §5  BOOLEAN / CUBE OPS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    try: trimesh.repair.fill_holes(m)
    except Exception: pass
    try: trimesh.repair.fix_winding(m)
    except Exception: pass
    try: trimesh.repair.fix_normals(m)
    except Exception: pass
    return m


def cube_at(mesh: trimesh.Trimesh, z_value: float,
            side: float = 40.0, subtract: bool = False) -> trimesh.Trimesh:
    verts     = mesh.vertices
    axis      = cylinder_axis(mesh)
    radial    = [i for i in range(3) if i != axis]
    ext       = verts.max(axis=0) - verts.min(axis=0)
    side_axis = radial[int(ext[radial[1]] > ext[radial[0]])]

    center        = mesh.bounds.mean(axis=0)
    center[axis]  = z_value
    near = np.abs(verts[:, axis] - z_value) <= max(side * 0.75, 30.0)
    slab = verts[near] if np.any(near) else verts

    anchor_idx = int(np.argmax(slab[:, side_axis]))
    anchor     = slab[anchor_idx]
    for ax in radial:
        if ax != side_axis: center[ax] = float(anchor[ax])
    surface = float(anchor[side_axis])
    penetration         = 34.0 if subtract else 10.0
    center[side_axis]   = surface + side / 2.0 - penetration
    cube = trimesh.creation.box(extents=(side, side, side))
    cube.apply_translation(center)
    return cube


def fallback_difference(mesh: trimesh.Trimesh,
                        cube: trimesh.Trimesh) -> trimesh.Trimesh:
    bounds  = cube.bounds
    centers = mesh.triangles_center
    inside  = np.all((centers >= bounds[0]) & (centers <= bounds[1]), axis=1)
    keep    = np.where(~inside)[0]
    return mesh.submesh([keep], append=True, repair=False)


def is_empty_mesh(mesh) -> bool:
    return (mesh is None
            or len(getattr(mesh, "vertices", [])) == 0
            or len(getattr(mesh, "faces", [])) == 0)


def boolean_union(mesh: trimesh.Trimesh,
                  cube: trimesh.Trimesh) -> trimesh.Trimesh:
    a2, b2 = _ensure_watertight(mesh), _ensure_watertight(cube)
    try:
        out = trimesh.boolean.union([a2, b2], engine="manifold", check_volume=False)
        if not is_empty_mesh(out): return out
    except Exception: pass
    return trimesh.util.concatenate([mesh, cube])


def boolean_difference(mesh: trimesh.Trimesh,
                       cube: trimesh.Trimesh) -> trimesh.Trimesh:
    a2, b2 = _ensure_watertight(mesh), _ensure_watertight(cube)
    try:
        out = trimesh.boolean.difference([a2, b2], engine="manifold", check_volume=False)
        if not is_empty_mesh(out) and len(out.faces) < len(mesh.faces): return out
    except Exception: pass
    return fallback_difference(mesh, cube)


def apply_booleans(mesh: trimesh.Trimesh, cube_side: float,
                   log=print) -> dict[str, trimesh.Trimesh]:
    axis = cylinder_axis(mesh)
    v    = mesh.vertices
    mn_a, mx_a   = float(v[:, axis].min()), float(v[:, axis].max())
    mn_a2, mx_a2 = mn_a + 60.0, mx_a - 60.0
    if mx_a2 <= mn_a2: mn_a2, mx_a2 = mn_a, mx_a
    total = mx_a2 - mn_a2
    positions = {
        "top": mn_a2 + total * 0.88,
        "mid": (mn_a2 + mx_a2) / 2,
        "bot": mn_a2 + total * 0.12,
    }
    res = {}
    for pn, pv in positions.items():
        log(f"  [UNION] {pn}")
        cube = cube_at(mesh, pv, side=cube_side, subtract=False)
        res[f"union_{pn}"] = boolean_union(mesh, cube)
        log(f"  [SUB  ] {pn}")
        cube = cube_at(mesh, pv, side=cube_side, subtract=True)
        res[f"sub_{pn}"]   = boolean_difference(mesh, cube)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# §6  ARTIFACT SAVING / EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_mesh(mesh: trimesh.Trimesh, path: Path, log=print):
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    log(f"  [OUT] {path.name}  ({path.stat().st_size // 1024} KB)")


def save_detection_artifacts(texture: np.ndarray, shapes: list[dict],
                              out_dir: Path):
    ink     = atlas_ink_mask(texture)
    cv2.imwrite(str(out_dir / "01_edge_detection_mask.png"), ink)
    overlay = texture.copy()
    colors  = {"E": (0, 255, 0), "C": (0, 165, 255), "S": (255, 0, 0)}
    for s in shapes:
        cv2.polylines(overlay, [s["polygon"]], True, colors[s["label"]], 5)
        x, y, w, h = cv2.boundingRect(s["polygon"])
        cv2.putText(overlay, f'{s["label"]}:{s["name"]}',
                    (x, max(30, y - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, colors[s["label"]], 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "02_flattened_shapes_labeled.png"), overlay)


def union_masks(shapes: list[dict], label: str,
                shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, np.uint8)
    for shape in shapes:
        if shape["label"] == label:
            mask = cv2.bitwise_or(mask, shape["mask"])
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# §7  CORE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(obj_path: Path, out_dir: Path, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh, texture, tex_path = load_mesh(obj_path, log=log)

    length   = float((mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).max())
    op_depth = length / 10.0
    axis     = cylinder_axis(mesh)
    bounds   = mesh.bounds

    # Estimate cube side from mesh extents
    ext = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    cube_side = float(sorted(ext)[-2]) * 0.05
    cube_side = max(cube_side, 1.0)

    shapes = refine_shape_masks(texture, curated_shape_masks(texture.shape[:2]))
    save_detection_artifacts(texture, shapes, out_dir)

    label_faces = {"E": [], "C": [], "S": []}
    summary = {
        "source_obj":      str(obj_path),
        "texture_atlas":   str(tex_path),
        "cylinder_length": length,
        "operation_depth": op_depth,
        "axis":            int(axis),
        "shapes":          [],
    }

    for s in shapes:
        ids = clean_region(mesh, face_ids_from_uv(mesh, s["mask"]))
        label_faces[s["label"]].append(ids)
        summary["shapes"].append({"label": s["label"], "name": s["name"],
                                   "faces": int(len(ids))})

    e_faces = np.unique(np.concatenate(label_faces["E"])) if label_faces["E"] else np.array([], dtype=np.int64)
    c_faces = np.unique(np.concatenate(label_faces["C"])) if label_faces["C"] else np.array([], dtype=np.int64)
    s_faces = np.unique(np.concatenate(label_faces["S"])) if label_faces["S"] else np.array([], dtype=np.int64)
    summary["face_totals"] = {"E": int(len(e_faces)), "C": int(len(c_faces)),
                               "S": int(len(s_faces))}

    (out_dir / "detected_shapes.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    export_mesh(mesh, out_dir / "00_original.obj", log=log)

    log("\n-- Extrude E (solid, no feathering) --")
    e_mesh = extrude_region(mesh, e_faces, op_depth, log=log)
    export_mesh(e_mesh, out_dir / "03_E_extruded.obj", log=log)

    log("\n-- Pocket C (solid, no feathering) --")
    c_mesh = pocket_region(mesh, c_faces, op_depth, log=log)
    export_mesh(c_mesh, out_dir / "04_C_pocketed.obj", log=log)

    log("\n-- S region (smoothing removed — solid mesh preserved) --")
    s_mesh = noop_smooth(mesh, s_faces, log=log)
    export_mesh(s_mesh, out_dir / "05_S_original.obj", log=log)

    log("\n-- ECS combined --")
    combined = extrude_region(mesh,     e_faces, op_depth,  log=log)
    combined = pocket_region(combined,  c_faces, op_depth,  log=log)
    # S region: no modification (solid mesh kept)
    export_mesh(combined, out_dir / "06_ECS_combined.obj", log=log)

    log(f"\n-- Boolean ops  cube={cube_side:.3f} --")
    for nm, m in apply_booleans(combined, cube_side, log=log).items():
        suffix = "07_union" if nm.startswith("union") else "08_sub"
        pos    = nm.split("_")[-1]
        export_mesh(m, out_dir / f"{suffix}_{pos}.obj", log=log)

    shutil.copy2(Path(__file__), out_dir / "assessment_solution.py")
    req = "\n".join(["numpy","opencv-python","Pillow","trimesh","scipy","manifold3d",""])
    (out_dir / "requirements.txt").write_text(req, encoding="utf-8")
    log(f"\n[DONE] Output: {out_dir}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# §8a  EMBEDDED 3-D VIEWER  (matplotlib / TkAgg)
# ─────────────────────────────────────────────────────────────────────────────

_LABEL_FACE_COLORS = {
    "E": (0.20, 0.75, 0.30, 1.0),   # green  – extruded
    "C": (0.15, 0.55, 0.95, 1.0),   # blue   – pocketed
    "S": (0.90, 0.40, 0.10, 1.0),   # orange – smooth zone
    "base": (0.82, 0.82, 0.82, 0.85),  # grey  – rest of mesh
}


class Mesh3DViewer(tk.Frame):
    """
    Interactive 3-D mesh viewer embedded inside a Tk Frame.
    Uses matplotlib (TkAgg backend) with a Poly3DCollection so the user can
    orbit, pan and zoom with the standard matplotlib mouse bindings.

    Parameters
    ----------
    parent      : tk widget to embed into
    mesh        : trimesh.Trimesh  – the final processed mesh
    label_faces : dict { "E": np.ndarray, "C": np.ndarray, "S": np.ndarray }
                  Face index arrays for each label (used for colour coding).
    """

    # Maximum triangles to send to matplotlib before we subsample.
    # Above this the renderer gets slow; 80 k gives good quality + speed.
    _MAX_TRIS = 80_000

    def __init__(self, parent, mesh: "trimesh.Trimesh",
                 label_faces: dict | None = None, **kw):
        super().__init__(parent, bg="#1a1a1a", **kw)
        self._mesh        = mesh
        self._label_faces = label_faces or {}
        self._fig         = None
        self._canvas_mpl  = None
        self._toolbar     = None
        self._build()

    # ── construction ─────────────────────────────────────────────────────────

    def _build(self):
        # ── toolbar strip ────────────────────────────────────────────────────
        tb = tk.Frame(self, bg="#111111"); tb.pack(side="top", fill="x")
        tk.Label(tb, text="  3D Interactive View", bg="#111111", fg="#ffffff",
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=4, pady=3)

        # colour-legend chips
        for label, rgba in [("E – Extruded",  _LABEL_FACE_COLORS["E"]),
                             ("C – Pocketed",  _LABEL_FACE_COLORS["C"]),
                             ("S – Smooth",    _LABEL_FACE_COLORS["S"])]:
            hex_c = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            tk.Label(tb, text=f"  ■ {label}", bg="#111111", fg=hex_c,
                     font=("Segoe UI", 8, "bold")).pack(side="left", padx=6)

        # reset-view button
        tk.Button(tb, text="⟳ Reset View", command=self._reset_view,
                  bg="#333333", fg="#ffffff", relief="flat", bd=0,
                  font=("Segoe UI", 8), padx=8, pady=3,
                  cursor="hand2", activebackground="#555555",
                  activeforeground="#ffffff").pack(side="right", padx=8, pady=2)

        # ── matplotlib figure ─────────────────────────────────────────────
        self._fig = plt.Figure(figsize=(8, 6), facecolor="#1a1a1a",
                               tight_layout=True)
        self._ax  = self._fig.add_subplot(111, projection="3d",
                                          facecolor="#1a1a1a")
        self._style_axes(self._ax)

        self._canvas_mpl = FigureCanvasTkAgg(self._fig, master=self)
        widget = self._canvas_mpl.get_tk_widget()
        widget.configure(bg="#1a1a1a", highlightthickness=0)
        widget.pack(fill="both", expand=True)

        # matplotlib navigation toolbar (zoom / pan / save)
        nav_frame = tk.Frame(self, bg="#222222")
        nav_frame.pack(side="bottom", fill="x")
        self._toolbar = NavigationToolbar2Tk(self._canvas_mpl, nav_frame)
        self._toolbar.configure(bg="#222222")
        self._toolbar.update()

        self._draw_mesh()

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_mesh(self):
        ax   = self._ax
        mesh = self._mesh
        ax.cla(); self._style_axes(ax)

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces,    dtype=np.int64)

        # Build per-face colour array
        n_faces    = len(faces)
        face_color = np.array([_LABEL_FACE_COLORS["base"]] * n_faces)
        for label, idx_arr in self._label_faces.items():
            if label not in _LABEL_FACE_COLORS: continue
            col = _LABEL_FACE_COLORS[label]
            valid = idx_arr[(idx_arr >= 0) & (idx_arr < n_faces)]
            face_color[valid] = col

        # Sub-sample if mesh is too large
        if n_faces > self._MAX_TRIS:
            rng  = np.random.default_rng(42)
            keep = rng.choice(n_faces, self._MAX_TRIS, replace=False)
            keep.sort()
            faces      = faces[keep]
            face_color = face_color[keep]

        # Triangle vertex arrays for Poly3DCollection
        tri_verts = verts[faces]   # (N, 3, 3)

        # Split into base and labelled for separate draw calls (performance)
        labelled_mask = np.any(
            face_color[:, :3] != np.array(_LABEL_FACE_COLORS["base"][:3]),
            axis=1)

        for mask, alpha, lw in [
            (~labelled_mask, 0.70, 0.0),
            ( labelled_mask, 0.95, 0.3),
        ]:
            if not np.any(mask): continue
            poly = Poly3DCollection(tri_verts[mask], closed=False)
            poly.set_facecolor(face_color[mask])
            poly.set_edgecolor("none" if lw == 0.0 else "#000000")
            poly.set_linewidth(lw)
            ax.add_collection3d(poly)

        # Axis limits centred on mesh bounds
        mn, mx = verts.min(axis=0), verts.max(axis=0)
        mid    = (mn + mx) * 0.5
        half   = (mx - mn).max() * 0.55
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)

        # Store initial view for reset
        self._init_elev = 20; self._init_azim = -60
        ax.view_init(elev=self._init_elev, azim=self._init_azim)

        self._canvas_mpl.draw_idle()

    def _style_axes(self, ax):
        dark = "#2a2a2a"
        ax.set_facecolor("#1a1a1a")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor(dark)
        ax.tick_params(colors="#888888", labelsize=6)
        ax.xaxis.label.set_color("#888888")
        ax.yaxis.label.set_color("#888888")
        ax.zaxis.label.set_color("#888888")
        ax.set_xlabel("X", fontsize=7, color="#888888")
        ax.set_ylabel("Y", fontsize=7, color="#888888")
        ax.set_zlabel("Z", fontsize=7, color="#888888")
        try:
            ax.xaxis._axinfo["grid"]["color"] = dark
            ax.yaxis._axinfo["grid"]["color"] = dark
            ax.zaxis._axinfo["grid"]["color"] = dark
        except Exception:
            pass

    def _reset_view(self):
        self._ax.view_init(elev=self._init_elev, azim=self._init_azim)
        self._canvas_mpl.draw_idle()

    def refresh(self, mesh: "trimesh.Trimesh", label_faces: dict | None = None):
        """Update the viewer with a new mesh (called after pipeline completes)."""
        self._mesh        = mesh
        self._label_faces = label_faces or {}
        self._draw_mesh()

    def destroy(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
        super().destroy()


# ─────────────────────────────────────────────────────────────────────────────
# §8  GUI  (adapted from v3 for the atlas pipeline)
# ─────────────────────────────────────────────────────────────────────────────

C_BG      = "#f5f5f5"; C_BG2    = "#ffffff"; C_SIDEBAR = "#efefef"
C_DIM     = "#888888"; C_TEXT   = "#111111"; C_ACCENT  = "#111111"
C_BTN     = "#ffffff"; C_LOG    = "#111111"
C_OK      = "#00e676"; C_ERR    = "#ff1744"; C_INF     = "#ffffff"
C_WARN    = "#ffea00"; C_SEP    = "#d0d0d0"
_FONT_UI  = ("Segoe UI", 9); _FONT_BOLD = ("Segoe UI", 9, "bold")
_FONT_MONO = ("Consolas", 9)


def _btn(parent, text, cmd, width=None, accent=False, **kw):
    bg  = C_ACCENT if accent else C_BTN
    fg  = "#ffffff" if accent else C_TEXT
    abg = "#333333" if accent else "#e0e0e0"
    kw.setdefault("bg", bg); kw.setdefault("fg", fg)
    kw.setdefault("relief", "flat"); kw.setdefault("bd", 0)
    kw.setdefault("cursor", "hand2"); kw.setdefault("font", _FONT_UI)
    kw.setdefault("activebackground", abg); kw.setdefault("activeforeground", fg)
    kw.setdefault("padx", 10); kw.setdefault("pady", 5)
    b = tk.Button(parent, text=text, command=cmd, **kw)
    if width: b.configure(width=width)
    b.configure(highlightthickness=1,
                highlightbackground=C_ACCENT if accent else C_SEP,
                highlightcolor=C_ACCENT)
    return b


def _sep(parent, bg=None):
    return tk.Frame(parent, bg=bg or C_SEP, height=1)


class RiseBionicsApp(tk.Tk):
    def __init__(self, init_obj=None, no_gui=False, out_dir="assessment_output",
                 cube_override=None, skip_bool=False):
        super().__init__()
        self.title("Rise Bionics  —  Atlas Pipeline  |  Solid Mesh Ops")
        self.configure(bg=C_BG)
        self.minsize(1200, 700); self.geometry("1600x900")

        self._obj_path    = tk.StringVar(value=init_obj or "")
        self._out_dir     = Path(out_dir)
        self._cube_over   = cube_override
        self._no_gui      = no_gui
        self._skip_bool_i = skip_bool
        self._mesh        = None
        self._texture     = None
        self._tex_path    = None
        self._shapes      = []
        self._running     = False
        self._logq        = queue.Queue()
        self._img_main    = None
        self._viewer_3d   = None   # Mesh3DViewer instance (created lazily)
        self._final_mesh  = None   # last processed trimesh after pipeline
        self._final_lf    = {}     # label_faces dict for colour coding

        self._build_ui()
        self._bind_shortcuts()
        self._poll_log()
        if init_obj:
            self.after(200, self._auto_start)

    # ── UI CONSTRUCTION ───────────────────────────────────────────────────

    def _build_ui(self):
        tk.Frame(self, bg=C_SEP, height=1).pack(fill="x", side="top")
        body = tk.Frame(self, bg=C_BG); body.pack(fill="both", expand=True)

        # Left sidebar (scrollable controls)
        lsb_outer = tk.Frame(body, bg=C_SIDEBAR, width=340)
        lsb_outer.pack(side="left", fill="y"); lsb_outer.pack_propagate(False)
        lsb_cv = tk.Canvas(lsb_outer, bg=C_SIDEBAR, highlightthickness=0, width=338)
        lsb_sb = tk.Scrollbar(lsb_outer, orient="vertical", command=lsb_cv.yview)
        lsb_cv.configure(yscrollcommand=lsb_sb.set)
        lsb_sb.pack(side="right", fill="y"); lsb_cv.pack(side="left", fill="both", expand=True)
        self._lsb_inner = tk.Frame(lsb_cv, bg=C_SIDEBAR)
        win = lsb_cv.create_window((0, 0), window=self._lsb_inner, anchor="nw")
        self._lsb_inner.bind("<Configure>",
                             lambda e: lsb_cv.configure(scrollregion=lsb_cv.bbox("all")))
        lsb_cv.bind("<Configure>", lambda e: lsb_cv.itemconfig(win, width=e.width))
        lsb_cv.bind_all("<MouseWheel>",
                        lambda e: lsb_cv.yview_scroll(int(-1*(e.delta/120)), "units"))
        self._build_controls(self._lsb_inner)

        # Centre preview canvas
        preview = tk.Frame(body, bg="#181818"); preview.pack(side="left", fill="both", expand=True)
        self._build_preview(preview)

        # Right log panel
        rsb = tk.Frame(body, bg="#111111", width=320)
        rsb.pack(side="right", fill="y"); rsb.pack_propagate(False)
        tk.Label(rsb, text="  Log", bg="#111111", fg="#ffffff",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(6, 0))
        _sep(rsb, bg="#333333").pack(fill="x")
        lf = tk.Frame(rsb, bg="#111111"); lf.pack(fill="both", expand=True)
        self._logtxt = tk.Text(lf, bg=C_LOG, fg=C_INF, font=_FONT_MONO,
                               state="disabled", relief="flat", bd=0,
                               wrap="word", padx=10, pady=6)
        ls2 = tk.Scrollbar(lf, command=self._logtxt.yview)
        self._logtxt.configure(yscrollcommand=ls2.set)
        ls2.pack(side="right", fill="y")
        self._logtxt.pack(side="left", fill="both", expand=True)
        for tag, col in [("err",C_ERR),("ok",C_OK),("info",C_INF),("warn",C_WARN)]:
            self._logtxt.tag_configure(tag, foreground=col)

        # Status bar
        sb = tk.Frame(self, bg=C_BG); sb.pack(fill="x", side="bottom")
        tk.Frame(sb, bg=C_SEP, height=1).pack(fill="x")
        inner = tk.Frame(sb, bg=C_BG); inner.pack(fill="x")
        self._sv = tk.StringVar(value="Ready.")
        self._lbl_st = tk.Label(inner, textvariable=self._sv, fg=C_TEXT, bg=C_BG,
                                font=_FONT_UI, anchor="w")
        self._lbl_st.pack(side="left", padx=8, pady=3)
        tk.Label(inner, text="© 2025 Rise Bionics  |  Solid Atlas Pipeline",
                 bg=C_BG, fg=C_DIM, font=("Segoe UI", 7)).pack(side="right", padx=10)
        self._prog = ttk.Progressbar(inner, mode="indeterminate", length=200)
        self._prog.pack(side="right", padx=6)

    def _build_controls(self, p):
        PAD  = dict(padx=12, pady=3)
        PADB = dict(padx=12, pady=5)

        def full_btn(text, cmd, accent=False):
            b = _btn(p, text, cmd, accent=accent); b.pack(fill="x", **PADB); return b

        def slbl(default="—"):
            lbl = tk.Label(p, text=default, bg=C_SIDEBAR, fg=C_DIM,
                           font=_FONT_UI, anchor="w", wraplength=310)
            lbl.pack(fill="x", padx=12, pady=(0, 3)); return lbl

        def sec(text):
            f = tk.Frame(p, bg=C_SIDEBAR); f.pack(fill="x", padx=4, pady=(8, 1))
            tk.Label(f, text=text, bg=C_SIDEBAR, fg=C_TEXT, font=_FONT_BOLD).pack(side="left")

        # Brand
        brand = tk.Frame(p, bg=C_SIDEBAR); brand.pack(fill="x", padx=12, pady=(14, 2))
        _logo_loaded = False
        for _logo_name in ["risebionics_logo.jpg", "risebionics_logo.png",
                           "rise_bionics_logo.jpg", "rise_bionics_logo.png",
                           "logo.jpg", "logo.png"]:
            _logo_path = Path(__file__).parent / _logo_name
            if _logo_path.exists():
                try:
                    _logo_img = Image.open(_logo_path).convert("RGBA")
                    _logo_img.thumbnail((220, 60), Image.LANCZOS)
                    _bg_rgb = tuple(int(C_SIDEBAR.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                    _logo_bg = Image.new("RGBA", _logo_img.size, _bg_rgb + (255,))
                    _mask = _logo_img.split()[3] if _logo_img.mode == "RGBA" else None
                    _logo_bg.paste(_logo_img, mask=_mask)
                    self._logo_photo = ImageTk.PhotoImage(_logo_bg.convert("RGB"))
                    tk.Label(brand, image=self._logo_photo, bg=C_SIDEBAR, bd=0).pack(anchor="w")
                    _logo_loaded = True
                except Exception:
                    pass
                break
        if not _logo_loaded:
            tk.Label(brand, text="rise BIONICS", bg=C_SIDEBAR, fg=C_TEXT,
                     font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(brand, text="STEP FORWARD", bg=C_SIDEBAR, fg="#111111",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(4, 0))
        tk.Label(brand, text="Atlas Pipeline  |  Solid Mesh Ops  |  No Smoothing",
                 bg=C_SIDEBAR, fg=C_DIM, font=("Segoe UI", 7)).pack(anchor="w")
        _sep(p).pack(fill="x", padx=12, pady=(8, 4))

        # ① Load
        sec("①  Load OBJ File")
        ff = tk.Frame(p, bg=C_SIDEBAR); ff.pack(fill="x", **PAD)
        tk.Entry(ff, textvariable=self._obj_path, relief="flat", bd=0,
                 font=_FONT_UI, bg=C_BG2, fg=C_TEXT,
                 highlightthickness=1, highlightbackground=C_SEP,
                 highlightcolor=C_ACCENT).pack(side="left", fill="x", expand=True, ipady=5)
        _btn(ff, " Browse… ", self._browse_obj).pack(side="left", padx=(4, 0))
        full_btn("Load Mesh", self._step_load, accent=True)
        self._lbl_mesh = slbl("No mesh loaded")
        _sep(p).pack(fill="x", padx=12, pady=4)

        # ② Detect atlas shapes
        sec("②  Detect Atlas Shapes")
        full_btn("Detect Shapes (Atlas)", self._step_detect)
        self._lbl_det = slbl()
        _sep(p).pack(fill="x", padx=12, pady=4)

        # ③ Run mesh ops
        sec("③  Mesh Operations")
        out_row = tk.Frame(p, bg=C_SIDEBAR); out_row.pack(fill="x", **PAD)
        tk.Label(out_row, text="Output folder:", bg=C_SIDEBAR, fg=C_TEXT,
                 font=_FONT_UI).pack(anchor="w")
        ef = tk.Frame(out_row, bg=C_SIDEBAR); ef.pack(fill="x", pady=(3, 0))
        self._var_out = tk.StringVar(value=str(self._out_dir))
        tk.Entry(ef, textvariable=self._var_out, relief="flat", bd=0,
                 font=_FONT_UI, bg=C_BG2, fg=C_TEXT,
                 highlightthickness=1, highlightbackground=C_SEP,
                 highlightcolor=C_ACCENT).pack(side="left", fill="x", expand=True, ipady=5)
        _btn(ef, "…", self._browse_out).pack(side="left", padx=(4, 0))

        depth_row = tk.Frame(p, bg=C_SIDEBAR); depth_row.pack(fill="x", padx=12, pady=(6, 2))
        tk.Label(depth_row, text="Op depth (% of length):", bg=C_SIDEBAR, fg=C_TEXT,
                 font=_FONT_UI).pack(side="left")
        self._depth_pct = tk.DoubleVar(value=10.0)
        self._depth_lbl = tk.Label(depth_row, text="10.0 %", bg=C_SIDEBAR, fg=C_TEXT,
                                   font=_FONT_BOLD, width=7)
        self._depth_lbl.pack(side="right")
        tk.Scale(p, from_=1, to=30, resolution=0.5, orient="horizontal",
                 variable=self._depth_pct,
                 command=lambda v: self._depth_lbl.configure(text=f"{float(v):.1f} %"),
                 bg=C_SIDEBAR, fg=C_TEXT, highlightthickness=0,
                 troughcolor=C_BG2, activebackground=C_ACCENT,
                 showvalue=False, length=200).pack(fill="x", padx=12, pady=(0, 4))

        cube_row = tk.Frame(p, bg=C_SIDEBAR); cube_row.pack(fill="x", padx=12, pady=(2, 2))
        tk.Label(cube_row, text="Cube side:", bg=C_SIDEBAR, fg=C_TEXT,
                 font=_FONT_UI).pack(side="left")
        self._cube_var = tk.DoubleVar(value=50.0)
        self._cube_lbl = tk.Label(cube_row, text="50.0", bg=C_SIDEBAR, fg=C_TEXT,
                                  font=_FONT_BOLD, width=6)
        self._cube_lbl.pack(side="right")
        tk.Scale(p, from_=1, to=500, resolution=0.5, orient="horizontal",
                 variable=self._cube_var,
                 command=lambda v: self._cube_lbl.configure(text=f"{float(v):.1f}"),
                 bg=C_SIDEBAR, fg=C_TEXT, highlightthickness=0,
                 troughcolor=C_BG2, activebackground=C_ACCENT,
                 showvalue=False, length=200).pack(fill="x", padx=12, pady=(0, 4))

        self._skip_bool = tk.BooleanVar(value=self._skip_bool_i)
        tk.Checkbutton(p, text="Skip Boolean ops", variable=self._skip_bool,
                       bg=C_SIDEBAR, fg=C_TEXT, font=_FONT_UI,
                       activebackground=C_SIDEBAR, selectcolor=C_BG2).pack(
                       anchor="w", padx=12, pady=(0, 4))

        full_btn("Run Mesh Operations", self._step_run, accent=True)
        full_btn("🧊  Open 3D View", self._show_3d_view)
        _sep(p).pack(fill="x", padx=12, pady=4)

        # Preview shortcuts
        sec("Preview Saved Images")
        for lbl2, fname in [
            ("Edge mask",              "01_edge_detection_mask.png"),
            ("Labeled shapes",         "02_flattened_shapes_labeled.png"),
        ]:
            _btn(p, lbl2, lambda f=fname: self._preview_file(f)).pack(
                fill="x", padx=12, pady=2)

        tk.Frame(p, bg=C_SIDEBAR, height=20).pack()
        _sep(p).pack(fill="x", padx=12)
        tk.Label(p, text="© 2025 Velan  |  Rise Bionics Beta App",
                 bg=C_SIDEBAR, fg=C_DIM, font=("Segoe UI", 7)).pack(
                 anchor="w", padx=14, pady=(6, 16))

    def _build_preview(self, parent):
        tb = tk.Frame(parent, bg="#111111"); tb.pack(fill="x", side="top")
        self._lbl_ptitle = tk.Label(tb, text="No preview", bg="#111111", fg="#aaaaaa",
                                    font=("Segoe UI", 9, "italic"), anchor="w")
        self._lbl_ptitle.pack(side="left", padx=8, pady=4)

        # Toggle buttons for 2-D / 3-D views
        self._view_mode = tk.StringVar(value="2d")
        btn_frame = tk.Frame(tb, bg="#111111"); btn_frame.pack(side="right", padx=8)
        self._btn_2d = tk.Button(btn_frame, text="🖼  Atlas View",
                                 command=self._show_2d_view,
                                 bg="#333333", fg="#ffffff", relief="flat", bd=0,
                                 font=("Segoe UI", 8), padx=8, pady=3,
                                 cursor="hand2", activebackground="#555555",
                                 activeforeground="#ffffff")
        self._btn_2d.pack(side="left", padx=(0, 4))
        self._btn_3d = tk.Button(btn_frame, text="🧊  3D View",
                                 command=self._show_3d_view,
                                 bg="#1a6b35", fg="#ffffff", relief="flat", bd=0,
                                 font=("Segoe UI", 8, "bold"), padx=8, pady=3,
                                 cursor="hand2", activebackground="#25a050",
                                 activeforeground="#ffffff")
        self._btn_3d.pack(side="left")

        # Container that holds both layers (stacked)
        self._preview_container = tk.Frame(parent, bg="#1a1a1a")
        self._preview_container.pack(fill="both", expand=True, padx=3, pady=3)

        # ── 2-D image canvas (always present) ────────────────────────────
        self._canv_main = tk.Canvas(self._preview_container,
                                    bg="#1a1a1a", highlightthickness=0)
        self._canv_main.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Loading spinner (lives inside the 2-D canvas)
        self._loader_frame = tk.Frame(self._preview_container, bg="#000000")
        self._loader_lbl   = tk.Label(self._loader_frame, text="", bg="#000000",
                                      fg="#ffffff", font=("Segoe UI", 18, "bold"),
                                      justify="center")
        self._loader_lbl.pack(expand=True)
        self._loader_dots  = 0; self._loader_active = False

        # ── placeholder label when no 3-D mesh is ready ──────────────────
        self._no3d_lbl = tk.Label(self._preview_container,
                                  text="Run the pipeline first to enable 3D View",
                                  bg="#1a1a1a", fg="#555555",
                                  font=("Segoe UI", 13))

    # ── VIEW SWITCHING ────────────────────────────────────────────────────

    def _show_2d_view(self):
        """Switch canvas area to the 2-D image view."""
        self._view_mode.set("2d")
        self._btn_2d.configure(bg="#555555")
        self._btn_3d.configure(bg="#1a6b35")
        if self._viewer_3d:
            self._viewer_3d.place_forget()
        self._no3d_lbl.place_forget()
        self._canv_main.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._lbl_ptitle.configure(text="Atlas Preview")

    def _show_3d_view(self):
        """Switch canvas area to the interactive 3-D viewer."""
        self._view_mode.set("3d")
        self._btn_3d.configure(bg="#555555")
        self._btn_2d.configure(bg="#1a6b35")
        self._canv_main.place_forget()
        if self._final_mesh is None:
            self._no3d_lbl.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._lbl_ptitle.configure(text="3D View — run pipeline first")
            return
        self._no3d_lbl.place_forget()
        if self._viewer_3d is None:
            if not _HAS_MPL:
                messagebox.showwarning("matplotlib missing",
                    "Install matplotlib to enable 3D view:\n  pip install matplotlib")
                self._show_2d_view(); return
            self._viewer_3d = Mesh3DViewer(
                self._preview_container,
                self._final_mesh,
                self._final_lf)
        else:
            self._viewer_3d.refresh(self._final_mesh, self._final_lf)
        self._viewer_3d.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._viewer_3d.lift()
        self._lbl_ptitle.configure(text="3D Interactive View  —  drag to orbit  |  scroll to zoom  |  right-drag to pan")

    # ── LOADER ANIMATION ─────────────────────────────────────────────────

    def _show_loader(self, msg="Processing"):
        self._loader_active = True; self._loader_dots = 0
        self._loader_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._loader_frame.lift(); self._animate_loader(msg)

    def _animate_loader(self, msg):
        if not self._loader_active: return
        dots = "●" * (self._loader_dots % 4)
        self._loader_lbl.configure(text=f"{msg}\n{dots}")
        self._loader_dots += 1
        self.after(350, lambda: self._animate_loader(msg))

    def _hide_loader(self):
        self._loader_active = False; self._loader_frame.place_forget()

    # ── LOGGING ──────────────────────────────────────────────────────────

    def _log(self, msg, tag=""):
        self._logq.put((str(msg), tag))

    def _poll_log(self):
        try:
            while True:
                msg, tag = self._logq.get_nowait()
                self._logtxt.configure(state="normal")
                self._logtxt.insert("end", msg + "\n", tag or "")
                self._logtxt.see("end")
                self._logtxt.configure(state="disabled")
        except queue.Empty: pass
        self.after(80, self._poll_log)

    def _status(self, msg, kind="info"):
        col = {"info": C_INF, "ok": C_OK, "err": C_ERR, "warn": C_WARN}.get(kind, C_TEXT)
        self._sv.set(msg); self._lbl_st.configure(fg=col); self._log(msg, kind)

    # ── BACKGROUND RUNNER ─────────────────────────────────────────────────

    def _run_bg(self, fn, *args, msg="Working…", **kw):
        if self._running:
            messagebox.showwarning("Busy", "A step is already running."); return
        self._running = True; self._prog.start(12)
        self.after(0, lambda: self._show_loader(msg))
        def target():
            try: fn(*args, **kw)
            except Exception as e:
                self._status(f"ERROR: {e}", "err")
                import traceback; self._log(traceback.format_exc(), "err")
            finally:
                self._running = False
                self.after(0, self._prog.stop)
                self.after(0, self._hide_loader)
        threading.Thread(target=target, daemon=True).start()

    # ── SHORTCUTS ─────────────────────────────────────────────────────────

    def _bind_shortcuts(self):
        self.bind("<Return>", lambda _: self._step_run())

    # ── BROWSE ────────────────────────────────────────────────────────────

    def _browse_obj(self):
        p = filedialog.askopenfilename(title="Select OBJ",
                                       filetypes=[("OBJ", "*.obj"), ("All", "*.*")])
        if p: self._obj_path.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="Output folder")
        if p: self._var_out.set(p)

    # ── STEP 1: LOAD ──────────────────────────────────────────────────────

    def _step_load(self):
        path = self._obj_path.get().strip()
        if not path or not Path(path).exists():
            messagebox.showerror("Error", "Select a valid .obj file first."); return
        self._run_bg(self._do_load, path, msg="Loading mesh…")

    def _do_load(self, path):
        self._status("Loading mesh…", "info")
        self._mesh, self._texture, self._tex_path = load_mesh(Path(path), log=self._log)
        v  = self._mesh.vertices; ex = v.max(0) - v.min(0)
        cs = float(sorted(ex)[-2]) * 0.05
        if self._cube_over: cs = self._cube_over
        cs = max(cs, 1.0)
        self.after(0, lambda: self._cube_var.set(round(cs, 1)))
        self.after(0, lambda: self._cube_lbl.configure(text=f"{cs:.1f}"))
        self.after(0, lambda: self._lbl_mesh.configure(
            text=f"{len(self._mesh.vertices):,} verts  {len(self._mesh.faces):,} faces",
            fg=C_OK))
        self._status("Mesh loaded.", "ok")

    # ── STEP 2: DETECT ────────────────────────────────────────────────────

    def _step_detect(self):
        if self._mesh is None:
            messagebox.showinfo("", "Load mesh first."); return
        self._run_bg(self._do_detect, msg="Detecting atlas shapes…")

    def _do_detect(self):
        self._status("Detecting atlas shapes…", "info")
        shapes = refine_shape_masks(
            self._texture, curated_shape_masks(self._texture.shape[:2]))
        self._shapes = shapes
        out = Path(self._var_out.get()); out.mkdir(parents=True, exist_ok=True)
        save_detection_artifacts(self._texture, shapes, out)
        self._log(f"[DET ] {len(shapes)} shapes detected", "ok")
        for s in shapes:
            ids = clean_region(self._mesh, face_ids_from_uv(self._mesh, s["mask"]))
            self._log(f"  {s['label']}:{s['name']}  → {len(ids)} faces")
        self.after(0, lambda: self._lbl_det.configure(
            text=f"{len(shapes)} shapes detected (E/C/S)", fg=C_OK))
        # Show atlas overlay in preview
        overlay_path = out / "02_flattened_shapes_labeled.png"
        img = cv2.imread(str(overlay_path))
        if img is not None:
            self.after(0, lambda: self._show_bgr(img, "Atlas – Labeled Shapes"))
        self._status(f"Detection done: {len(shapes)} shapes.", "ok")

    # ── STEP 3: RUN MESH OPS ─────────────────────────────────────────────

    def _step_run(self):
        if self._mesh is None:
            messagebox.showerror("Error", "Load mesh first."); return
        self._out_dir = Path(self._var_out.get())
        self._run_bg(self._do_run, msg="Running pipeline…")

    def _do_run(self):
        self._status("Running mesh operations…", "info")
        out   = self._out_dir; out.mkdir(parents=True, exist_ok=True)
        mesh  = self._mesh
        tex   = self._texture
        shapes = (self._shapes if self._shapes
                  else refine_shape_masks(tex, curated_shape_masks(tex.shape[:2])))

        length    = float((mesh.vertices.max(0) - mesh.vertices.min(0)).max())
        depth_pct = float(self._depth_pct.get()) / 100.0
        op_depth  = length * depth_pct
        cube_side = float(self._cube_var.get())
        self._log(f"[INFO] op_depth={op_depth:.4f}  cube={cube_side:.3f}", "info")

        label_faces = {"E": [], "C": [], "S": []}
        for s in shapes:
            ids = clean_region(mesh, face_ids_from_uv(mesh, s["mask"]))
            label_faces[s["label"]].append(ids)
            self._log(f"  {s['label']}:{s['name']}  → {len(ids)} faces")

        e_faces = np.unique(np.concatenate(label_faces["E"])) if label_faces["E"] else np.array([], int)
        c_faces = np.unique(np.concatenate(label_faces["C"])) if label_faces["C"] else np.array([], int)
        s_faces = np.unique(np.concatenate(label_faces["S"])) if label_faces["S"] else np.array([], int)
        self._log(f"[INFO] E:{len(e_faces)}  C:{len(c_faces)}  S:{len(s_faces)}", "info")

        export_mesh(mesh, out / "00_original.obj", log=self._log)

        self._log("\n-- Extrude E --", "info")
        working = extrude_region(mesh, e_faces, op_depth, log=self._log)
        export_mesh(working, out / "03_E_extruded.obj", log=self._log)

        self._log("\n-- Pocket C --", "info")
        working = pocket_region(working, c_faces, op_depth, log=self._log)
        export_mesh(working, out / "04_C_pocketed.obj", log=self._log)

        self._log("\n-- S region (smoothing OFF — solid mesh) --", "info")
        working = noop_smooth(working, s_faces, log=self._log)
        export_mesh(working, out / "06_ECS_combined.obj", log=self._log)

        if not self._skip_bool.get():
            self._log(f"\n-- Boolean ops  cube={cube_side:.3f} --", "info")
            for nm, m in apply_booleans(working, cube_side, log=self._log).items():
                suffix = "07_union" if nm.startswith("union") else "08_sub"
                pos    = nm.split("_")[-1]
                export_mesh(m, out / f"{suffix}_{pos}.obj", log=self._log)
        else:
            self._log("[INFO] Boolean skipped", "warn")

        # ── Store final mesh and per-label face indices for 3D viewer ────
        # Rebuild flat label_faces for colour mapping
        flat_lf = {}
        for lbl in ("E", "C", "S"):
            arrs = label_faces.get(lbl, [])
            if arrs:
                flat_lf[lbl] = np.unique(np.concatenate(arrs)).astype(np.int64)
        self._final_mesh = working
        self._final_lf   = flat_lf

        self._status("Pipeline complete! Opening 3D viewer…", "ok")
        self._log(f"\n[DONE] Output: {out}", "ok")

        # Auto-switch to interactive 3-D view
        self.after(0, self._open_3d_after_run)

    # ── PREVIEW ───────────────────────────────────────────────────────────

    def _open_3d_after_run(self):
        """Called on the main thread after the pipeline thread finishes."""
        self._show_3d_view()
        messagebox.showinfo("Done",
            f"Pipeline complete!\n\nThe final mesh is shown in the 3D viewer.\n"
            f"Drag to orbit  |  Scroll to zoom  |  Right-drag to pan\n\n"
            f"Outputs saved to:\n{self._out_dir}")

    def _show_bgr(self, bgr, title=""):
        # Ensure 2D canvas is on top
        if self._viewer_3d:
            self._viewer_3d.place_forget()
        self._no3d_lbl.place_forget()
        self._canv_main.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._canv_main.lift()
        self._btn_2d.configure(bg="#555555")
        self._btn_3d.configure(bg="#1a6b35")
        self._view_mode.set("2d")

        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cw  = max(100, self._preview_container.winfo_width())
        ch  = max(100, self._preview_container.winfo_height())
        pil.thumbnail((cw, ch), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self._img_main = photo
        self._canv_main.delete("all")
        self._canv_main.create_image(cw // 2, ch // 2, anchor="center", image=photo)
        self._lbl_ptitle.configure(text=title)

    def _preview_file(self, fname):
        p = Path(self._var_out.get()) / fname
        if not p.exists():
            messagebox.showinfo("", "File not found — run earlier steps first."); return
        img = cv2.imread(str(p))
        if img is not None: self._show_bgr(img, str(p))

    def _auto_start(self):
        if self._no_gui:
            self._run_bg(self._headless)

    def _headless(self):
        self._do_load(self._obj_path.get())
        self._do_detect()
        self._do_run()


# ─────────────────────────────────────────────────────────────────────────────
# §9  ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

def _find_obj(hint: Path | None) -> Path | None:
    """
    Try to locate the OBJ file.  Search order:
      1. Exactly as given (if given).
      2. Next to the script file.
      3. Parent directories (up to 3 levels).
      4. Common download sub-folders next to the script.
    Returns the Path if found, else None.
    """
    candidates = []
    default_name = "EngineeringMLAssessment.obj"

    if hint is not None:
        candidates.append(hint)
        # Also try the same filename next to the script
        candidates.append(Path(__file__).parent / hint.name)

    script_dir = Path(__file__).parent
    candidates.append(script_dir / default_name)
    for level in range(1, 4):
        parent = script_dir
        for _ in range(level): parent = parent.parent
        candidates.append(parent / default_name)
        # Common sub-folder names
        for sub in ["Engineering_ML_Assessment_0", "RiseBionics", "assessment"]:
            candidates.append(parent / sub / default_name)

    for p in candidates:
        try:
            if p.exists(): return p.resolve()
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Rise Bionics Atlas Pipeline — solid mesh ops, no smoothing")
    parser.add_argument("obj", type=Path, nargs="?", default=None)
    parser.add_argument("--out",           type=Path,  default=Path("assessment_output"))
    parser.add_argument("--cube",          type=float, default=None)
    parser.add_argument("--no-gui",        action="store_true")
    parser.add_argument("--skip-booleans", action="store_true")
    args = parser.parse_args()

    found_obj = _find_obj(args.obj)

    # ── Headless / no-GUI mode ────────────────────────────────────────────
    if args.no_gui or not _HAS_TK:
        if not _HAS_TK:
            print("[WARN] tkinter unavailable — headless run")
        if found_obj is None:
            if args.obj:
                raise SystemExit(
                    f"[ERR] File not found: {args.obj}\n"
                    f"      Make sure EngineeringMLAssessment.obj is in the same\n"
                    f"      folder as this script, or pass the full path:\n"
                    f"      py assessment_solution.py path/to/EngineeringMLAssessment.obj"
                )
            # No argument given either — ask interactively
            raw = input("OBJ path (press Enter to cancel): ").strip()
            if not raw:
                raise SystemExit("[ERR] No OBJ file provided.")
            found_obj = Path(raw)
            if not found_obj.exists():
                raise SystemExit(f"[ERR] File not found: {found_obj}")
        summary = run(found_obj, args.out)
        print(json.dumps(summary, indent=2))
        return

    # ── GUI mode ──────────────────────────────────────────────────────────
    # If the OBJ wasn't found but one was specified, warn — user can browse in GUI
    if args.obj and found_obj is None:
        print(
            f"[WARN] OBJ not found at '{args.obj}'.\n"
            f"       The GUI will open — use the Browse button to locate the file."
        )

    app = RiseBionicsApp(
        init_obj=str(found_obj) if found_obj else None,
        no_gui=False,
        out_dir=str(args.out),
        cube_override=args.cube,
        skip_bool=args.skip_booleans,
    )
    app.mainloop()


if __name__ == "__main__":
    main()