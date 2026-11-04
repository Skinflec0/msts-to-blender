#!/usr/bin/env python3
"""
msts_to_glb - convert MSTS / Open Rails shapes (.s) and textures (.ace) to .glb

Usage
-----
    python msts_to_glb.py                      # no arguments: pick a folder in a dialog
    python msts_to_glb.py path/to/folder       # convert every .s file inside (recursive)
    python msts_to_glb.py wagon.s              # convert a single file
    python msts_to_glb.py a.s b.s folder -o out

Drag-and-drop a folder or .s files onto the script on Windows and it works too.
The resulting .glb files import directly into Blender (File > Import > glTF 2.0).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import traceback
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

__version__ = "2.0.0"
U = struct.unpack_from

# Windows consoles are often cp1252; never let a print() crash a conversion.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# =============================================================================
# ACE texture loader
# =============================================================================
def _unwrap(data: bytes) -> bytes:
    """Strip the SIMISA header and inflate if the payload is compressed."""
    sig = data[:8]
    if sig == b"SIMISA@F":
        body = data[16:]
        try:
            return zlib.decompress(body)
        except zlib.error:
            return zlib.decompress(body, -15)  # raw deflate stream
    if sig == b"SIMISA@@":
        return data[16:]
    return data


def load_ace(path) -> Image.Image:
    r = _unwrap(Path(path).read_bytes())
    if len(r) < 152:
        raise ValueError("file too short to be an ACE texture")

    w, h, _fmt, nch = struct.unpack_from("<4I", r, 8)
    if not (0 < w <= 16384 and 0 < h <= 16384 and 0 < nch <= 8):
        raise ValueError("corrupt ACE header")

    off = 24 + 128
    chs = [U("<QQ", r, off + k * 16) for k in range(nch)]
    if any(t not in (2, 3, 4, 5, 6) or b not in (1, 8) for b, t in chs):
        raise ValueError("unsupported ACE pixel layout (DXT-compressed ACE is not supported)")

    tab = off + nch * 16
    rows = np.array(U(f"<{h}I", r, tab), dtype=np.int64)
    row_bytes = sum(w if b == 8 else (w + 7) // 8 for b, _ in chs)

    # Row offsets are normally relative to the decompressed data; some
    # uncompressed files count from the start of the file instead.
    base = None
    for cand in (0, 16):
        if rows.min() + cand >= 0 and (rows + cand + row_bytes <= len(r)).all():
            base = cand
            break
    if base is None:
        raise ValueError("row table points outside the file")

    planes = {t: np.zeros((h, w), np.uint8) for _, t in chs}
    for y in range(h):
        p = int(rows[y]) + base
        for bits, t in chs:
            if bits == 8:
                planes[t][y] = np.frombuffer(r, np.uint8, w, p)
                p += w
            else:
                n = (w + 7) // 8
                planes[t][y] = np.unpackbits(np.frombuffer(r, np.uint8, n, p), bitorder="little")[:w] * 255
                p += n

    zero = np.zeros((h, w), np.uint8)
    R, G, B = planes.get(3, zero), planes.get(4, zero), planes.get(5, zero)
    A = planes.get(6, planes.get(2))
    if A is not None:
        return Image.fromarray(np.dstack([R, G, B, A]), "RGBA")
    return Image.fromarray(np.dstack([R, G, B]), "RGB")


def load_image(path: str) -> Image.Image:
    if path.lower().endswith(".ace"):
        return load_ace(path)
    im = Image.open(path)
    return im.convert("RGBA" if "A" in im.getbands() else "RGB")


def alpha_kind(im: Image.Image):
    """None = opaque, 'MASK' = cut-out, 'BLEND' = soft transparency."""
    if im.mode != "RGBA":
        return None
    a = np.asarray(im.getchannel("A"))
    if (a >= 250).all():
        return None
    soft = float(((a > 20) & (a < 235)).mean()) > 0.015
    return "BLEND" if soft else "MASK"


# =============================================================================
# Binary .s shape parser
# =============================================================================
class Shape:
    """Walks the block tree of a binary MSTS shape file."""

    def __init__(self, path):
        d = Path(path).read_bytes()
        if d[:8] == b"SIMISA@F":
            self.r = zlib.decompress(d[16:])
        elif d[:8] == b"SIMISA@@":
            self.r = d[16:]
        else:
            raise ValueError("not an MSTS shape file (missing SIMISA header)")
        if self.r[:8] == b"JINX0s1t":
            raise ValueError(
                "this is a TEXT-format shape. Re-save it as binary first "
                "(e.g. with Shape Viewer / ffeditc_unicode.exe) and try again"
            )
        self.top = {}
        _t, ln = U("<II", self.r, 16)
        for h in self.kids(25, 16 + 8 + ln):
            self.top[h[0]] = h

    def hdr(self, p):
        t, ln = U("<II", self.r, p)
        n = self.r[p + 8]
        name = self.r[p + 9 : p + 9 + 2 * n].decode("utf-16le", "replace")
        return (t, ln, name, p + 9 + 2 * n, p + 8 + ln)

    def kids(self, cs, end):
        out, p = [], cs
        while p < end:
            h = self.hdr(p)
            if h[4] <= p:
                break
            out.append(h)
            p = h[4]
        return out

    def try_kids(self, h):
        """Children of a block, tolerating an optional 4-byte count prefix."""
        cs, end = h[3], h[4]
        for skip in (0, 4):
            p, out = cs + skip, []
            while p < end:
                if p + 9 > end:
                    break
                t, ln = U("<II", self.r, p)
                if t > 200 or ln < 1 or p + 8 + ln > end:
                    break
                out.append(self.hdr(p))
                p += 8 + ln
            else:
                if p == end and out:
                    return out
        return None

    def need(self, tok):
        if tok not in self.top:
            raise ValueError(f"section {tok} missing - not a supported shape file")
        return self.top[tok]

    def counted(self, tok):
        h = self.need(tok)
        return self.kids(h[3] + 4, h[4])

    def arr(self, tok, n):
        h = self.need(tok)
        cnt = U("<I", self.r, h[3])[0]
        stride = 9 + 4 * n
        b = np.frombuffer(self.r, np.uint8, cnt * stride, h[3] + 4).reshape(cnt, stride)[:, 9:]
        return np.ascontiguousarray(b).view(np.float32).reshape(cnt, n)


KNOWN_SHADERS = {"texdiff", "tex", "blendatexdiff", "blendatex", "addatex", "addatexdiff", "diffuse"}


def _find_shader_names(sh: Shape, used):
    """Best-effort lookup of the shader name table (token id differs between exporters)."""
    for tok in sh.top:
        if tok in used:
            continue
        try:
            names = []
            for c in sh.counted(tok):
                n = U("<H", sh.r, c[3])[0]
                names.append(sh.r[c[3] + 2 : c[3] + 2 + 2 * n].decode("utf-16le", "replace").lower())
            if names and any(x in KNOWN_SHADERS for x in names):
                return names
        except Exception:
            continue
    return None


def _read_vertices(r, vh):
    cnt = U("<I", r, vh[3])[0]
    p = vh[3] + 4
    pt = np.zeros(cnt, np.uint32)
    nm = np.zeros(cnt, np.uint32)
    uvi = np.zeros(cnt, np.uint32)
    if cnt == 0:
        return pt, nm, uvi

    if p + cnt * 46 <= vh[4]:  # fast path: every vertex has exactly one UV set
        raw = np.frombuffer(r, np.uint8, cnt * 46, p).reshape(cnt, 46)
        lens = raw[:, 4:8].copy().view(np.uint32).ravel()
        if (lens == 38).all():
            d = raw[:, 8:]
            return (
                d[:, 5:9].copy().view(np.uint32).ravel(),
                d[:, 9:13].copy().view(np.uint32).ravel(),
                d[:, 34:38].copy().view(np.uint32).ravel(),
            )
    for i in range(cnt):  # slow path: variable-sized vertex blocks
        _t, ln = U("<II", r, p)
        c = p + 9
        pt[i], nm[i] = U("<II", r, c + 4)
        q = c + 20
        if q + 17 <= p + 8 + ln and U("<I", r, q + 9)[0] > 0:
            uvi[i] = U("<I", r, q + 13)[0]
        p += 8 + ln
    return pt, nm, uvi


def parse(path, lod=0):
    sh = Shape(path)
    r = sh.r
    S = {}

    S["pts"] = sh.arr(7, 3)
    S["uv"] = sh.arr(9, 2)
    S["nrm"] = sh.arr(5, 3)
    S["mat"] = [(c[2], np.array(U("<12f", r, c[3]), np.float32)) for c in sh.counted(66)]
    S["imgnames"] = [
        r[c[3] + 2 : c[3] + 2 + 2 * U("<H", r, c[3])[0]].decode("utf-16le", "replace")
        for c in sh.counted(14)
    ]
    S["tex2img"] = [U("<I", r, c[3])[0] for c in sh.counted(16)]

    def kid(h, tok):
        return [c for c in (sh.try_kids(h) or []) if c[0] == tok]

    def first(h, tok, what):
        k = kid(h, tok)
        if not k:
            raise ValueError(f"{what} not found in shape")
        return k[0]

    lod_ctrl = first(sh.need(31), 32, "lod_control")
    dls_blk = first(lod_ctrl, 36, "distance_levels")
    dls = sh.kids(dls_blk[3] + 4, dls_blk[4])
    if not dls:
        raise ValueError("shape has no distance levels")
    S["n_lods"] = len(dls)
    dl = dls[min(max(lod, 0), len(dls) - 1)]

    hier = first(first(dl, 34, "distance_level_header"), 67, "hierarchy")
    n = U("<I", r, hier[3])[0]
    S["hier"] = list(U(f"<{n}i", r, hier[3] + 4))

    # prim_states / vtx_states first: sub-objects refer to them
    S["vtx"] = [U("<I", r, c[3] + 4)[0] for c in sh.counted(47)]
    shader_names = _find_shader_names(sh, {5, 7, 9, 14, 16, 31, 47, 55, 66})

    S["prim"] = []
    for c in sh.counted(55):
        d = c[3]
        _flags, shader = U("<2I", r, d)
        tx = sh.hdr(d + 8)  # tex_idxs sub-block
        cnt = U("<I", r, tx[3])[0]
        tex = U("<I", r, tx[3] + 4)[0] if cnt > 0 else -1
        # after tex_idxs: z_bias (float), then the vertex-state index
        vtx = U("<I", r, tx[4] + 4)[0]
        if vtx >= len(S["vtx"]):
            vtx = U("<I", r, tx[4])[0]
        if vtx >= len(S["vtx"]):
            vtx = 0
        if shader_names and shader < len(shader_names):
            nm = shader_names[shader]
            alpha = nm.startswith(("blenda", "adda"))
        else:
            alpha = shader == 1
        S["prim"].append(dict(shader=shader, tex=tex, vtx=vtx, alpha=alpha))

    sos = first(dl, 38, "sub_objects")
    S["subs"] = []
    for so in sh.kids(sos[3] + 4, sos[4]):
        sub = {}
        sub["pt"], sub["nm"], sub["uvi"] = _read_vertices(r, first(so, 50, "vertices"))
        tris, cur = [], 0
        pr = first(so, 53, "primitives")
        for c in sh.kids(pr[3] + 4, pr[4]):
            if c[0] == 56:
                cur = U("<I", r, c[3])[0]
            elif c[0] == 60:
                vi = [x for x in (sh.try_kids(c) or []) if x[0] == 63]
                if not vi:
                    continue
                m = U("<I", r, vi[0][3])[0]
                idx = np.frombuffer(r, np.uint32, m, vi[0][3] + 4)
                idx = idx[: (len(idx) // 3) * 3].reshape(-1, 3)
                if len(idx) and idx.max() < len(sub["pt"]) and cur < len(S["prim"]):
                    tris.append((cur, idx))
        sub["tris"] = tris
        S["subs"].append(sub)
    return S


# =============================================================================
# Texture lookup
# =============================================================================
class TextureFinder:
    """Case-insensitive texture search: shape folder, nearby TEXTURES folders, then the whole input tree."""

    def __init__(self, roots, extra_dirs=()):
        self.global_index = {}
        self.extra_dirs = [Path(d) for d in extra_dirs]
        for root in roots:
            for dp, _, files in os.walk(root):
                for f in files:
                    self.global_index.setdefault(f.lower(), os.path.join(dp, f))
        self._dir_cache = {}

    def _dir(self, d: Path):
        if d not in self._dir_cache:
            try:
                self._dir_cache[d] = {f.lower(): str(d / f) for f in os.listdir(d)}
            except OSError:
                self._dir_cache[d] = {}
        return self._dir_cache[d]

    def find(self, name, shape_dir: Path):
        name = name.strip().replace("\\", "/")
        base = os.path.basename(name)
        stem = os.path.splitext(base)[0]
        cands = [base, stem + ".ace", stem + ".dds", stem + ".png", stem + ".tga", base + ".ace"]
        near = [shape_dir, shape_dir / "textures", shape_dir.parent / "textures", shape_dir / "texture"]
        near += self.extra_dirs
        for d in near:
            idx = self._dir(d)
            for c in cands:
                if c.lower() in idx:
                    return idx[c.lower()]
        for c in cands:
            if c.lower() in self.global_index:
                return self.global_index[c.lower()]
        return None


# =============================================================================
# glTF export
# =============================================================================
FL = np.array([1.0, 1.0, -1.0], np.float32)  # MSTS is left-handed (Z forward); glTF is right-handed


def export(shape_path, out_path, finder: TextureFinder, opts, log=print):
    S = parse(shape_path, opts.lod)
    shape_dir = Path(shape_path).resolve().parent

    buf = bytearray()
    views, accs = [], []

    def pad():
        while len(buf) % 4:
            buf.append(0)

    def add_view(data, target=None):
        pad()
        v = {"buffer": 0, "byteOffset": len(buf), "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        buf.extend(data)
        return len(views) - 1

    def add_acc(a, ctype, target, minmax=False):
        ac = {
            "bufferView": add_view(a.tobytes(), target),
            "componentType": ctype,
            "count": int(a.shape[0]),
            "type": {1: "SCALAR", 2: "VEC2", 3: "VEC3"}[a.shape[1]],
        }
        if minmax:
            ac["min"], ac["max"] = a.min(0).tolist(), a.max(0).tolist()
        accs.append(ac)
        return len(accs) - 1

    # ---- textures
    images, img_alpha = [], []
    if not opts.no_textures:
        for name in S["imgnames"]:
            path = finder.find(name, shape_dir)
            im, kind = None, None
            if path:
                try:
                    im = load_image(path)
                    kind = alpha_kind(im)
                    log(f"    texture ok      {name}  ({im.size[0]}x{im.size[1]})")
                except Exception as e:
                    log(f"    texture FAILED  {name}: {e}")
            else:
                log(f"    texture missing {name}")
            if im is None:
                im = Image.new("RGB", (4, 4), (140, 140, 140))
            b = io.BytesIO()
            im.save(b, "PNG")
            images.append({
                "bufferView": add_view(b.getvalue()),
                "mimeType": "image/png",
                "name": os.path.splitext(os.path.basename(name))[0],
            })
            img_alpha.append(kind)

    textures = []
    if images:
        for src in S["tex2img"]:
            textures.append({"source": min(src, len(images) - 1), "sampler": 0})

    # ---- materials
    materials, mat_index = [], {}

    def get_mat(key):
        if key in mat_index:
            return mat_index[key]
        t, alpha = key
        pbr = {"metallicFactor": 0.0, "roughnessFactor": 0.82}
        kind, name = None, "untextured"
        if 0 <= t < len(textures):
            pbr["baseColorTexture"] = {"index": t}
            src = textures[t]["source"]
            name, kind = images[src]["name"], img_alpha[src]
        else:
            pbr["baseColorFactor"] = [0.6, 0.6, 0.6, 1.0]
        m = {"name": name, "pbrMetallicRoughness": pbr}
        if alpha and kind:
            m["name"] += "_alpha"
            m["alphaMode"] = kind
            if kind == "MASK":
                m["alphaCutoff"] = 0.42
        if opts.double_sided:
            m["doubleSided"] = True
        materials.append(m)
        mat_index[key] = len(materials) - 1
        return mat_index[key]

    # ---- geometry, grouped by (node, texture, alpha)
    groups = {}
    for sub in S["subs"]:
        for ps, idx in sub["tris"]:
            p = S["prim"][ps]
            node = S["vtx"][p["vtx"]] if p["vtx"] < len(S["vtx"]) else 0
            groups.setdefault((node, (p["tex"], p["alpha"])), []).append((sub, idx))
    if not groups:
        raise ValueError("no triangles found in the selected LOD")

    node_prims = {}
    for (node, key), lst in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        P, N, UV, I = [], [], [], []
        base = 0
        for sub, idx in lst:
            u, inv = np.unique(idx.ravel(), return_inverse=True)
            pts = S["pts"][sub["pt"][u]].astype(np.float32) * FL
            nrm = S["nrm"][sub["nm"][u]].astype(np.float32) * FL
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8)
            uvs = S["uv"][sub["uvi"][u]]
            if opts.zfight:
                pts = pts + nrm * np.float32(opts.zfight)  # optional, off by default
            P.append(pts)
            N.append(nrm)
            UV.append(uvs)
            I.append((inv.reshape(-1, 3) + base)[:, [0, 2, 1]].astype(np.uint32))  # flip winding
            base += len(u)
        P, N, UV, I = np.vstack(P), np.vstack(N), np.vstack(UV), np.vstack(I)
        node_prims.setdefault(node, []).append({
            "attributes": {
                "POSITION": add_acc(P, 5126, 34962, True),
                "NORMAL": add_acc(N, 5126, 34962),
                "TEXCOORD_0": add_acc(UV, 5126, 34962),
            },
            "indices": add_acc(I.reshape(-1, 1), 5125, 34963),
            "material": get_mat(key),
            "mode": 4,
        })

    # ---- nodes and hierarchy
    meshes, node_mesh = [], {}
    for n, prims in node_prims.items():
        meshes.append({"name": S["mat"][n][0] if n < len(S["mat"]) else f"node{n}", "primitives": prims})
        node_mesh[n] = len(meshes) - 1

    nodes = []
    F4 = np.diag([1.0, 1.0, -1.0, 1.0])
    for i, (name, m) in enumerate(S["mat"]):
        R = m[:9].reshape(3, 3)
        if np.allclose(R, np.eye(3), atol=1e-4):
            nd = {"name": name, "translation": (m[9:12] * FL).tolist()}
        else:
            A = np.eye(4)
            A[:3, :3] = R.T
            A[:3, 3] = m[9:12]
            nd = {"name": name, "matrix": (F4 @ A @ F4).T.ravel().tolist()}
        if i in node_mesh:
            nd["mesh"] = node_mesh[i]
        nodes.append(nd)

    roots = []
    for i, p in enumerate(S["hier"][: len(nodes)]):
        if 0 <= p < len(nodes) and p != i:
            nodes[p].setdefault("children", []).append(i)
        else:
            roots.append(i)
    nodes.append({"name": Path(shape_path).stem, "children": roots})

    pad()
    g = {
        "asset": {"version": "2.0", "generator": f"msts_to_glb {__version__}"},
        "scene": 0,
        "scenes": [{"nodes": [len(nodes) - 1]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accs,
        "bufferViews": views,
        "buffers": [{"byteLength": len(buf)}],
    }
    if images:
        g.update(images=images, textures=textures,
                 samplers=[{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}])

    js = json.dumps(g, separators=(",", ":")).encode()
    while len(js) % 4:
        js += b" "
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(buf), 0x004E4942) + bytes(buf))
    return out_path.stat().st_size / (1024 * 1024)


# =============================================================================
# Command line
# =============================================================================
def pick_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        d = filedialog.askdirectory(title="Choose the folder with your MSTS .s and .ace files")
        root.destroy()
        return d or None
    except Exception:
        return None


def collect_shapes(targets):
    shapes = []  # (shape_path, root_it_came_from)
    for t in targets:
        t = Path(t)
        if t.is_file() and t.suffix.lower() == ".s":
            shapes.append((t, t.parent))
        elif t.is_dir():
            for dp, dn, files in os.walk(t):
                dn.sort()
                for f in sorted(files):
                    if f.lower().endswith(".s"):
                        shapes.append((Path(dp) / f, t))
    return shapes


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="msts_to_glb",
        description="Convert MSTS/Open Rails .s shapes and .ace textures to .glb (opens in Blender).",
    )
    ap.add_argument("paths", nargs="*", help="folder(s) and/or .s file(s); omit to pick a folder in a dialog")
    ap.add_argument("-o", "--out-dir", help="write .glb files here (default: next to each .s file)")
    ap.add_argument("-t", "--texture-dir", action="append", default=[], help="extra folder to search for textures (repeatable)")
    ap.add_argument("--lod", type=int, default=0, help="level of detail to export, 0 = most detailed (default 0)")
    ap.add_argument("--skip-existing", action="store_true", help="do not overwrite existing .glb files")
    ap.add_argument("--no-textures", action="store_true", help="export geometry only")
    ap.add_argument("--double-sided", action="store_true", help="render both sides of every face")
    ap.add_argument("--zfight", type=float, default=0.0, metavar="METRES",
                    help="push vertices along their normal by this amount to hide z-fighting (try 0.002; default off)")
    ap.add_argument("--debug", action="store_true", help="show full error tracebacks")
    ap.add_argument("--version", action="version", version=__version__)
    opts = ap.parse_args(argv)

    interactive = not opts.paths
    targets = list(opts.paths)
    if interactive:
        print(f"msts_to_glb {__version__}")
        d = pick_folder()
        if not d:
            d = input("Folder with your .s files (Enter = current folder): ").strip().strip('"') or "."
        targets = [d]

    missing = [t for t in targets if not os.path.exists(t)]
    if missing:
        print("Not found: " + ", ".join(missing))
        return 2

    shapes = collect_shapes(targets)
    if not shapes:
        print("No .s files found.")
        return 2

    roots = [Path(t) if os.path.isdir(t) else Path(t).parent for t in targets]
    print(f"Found {len(shapes)} shape file(s). Indexing textures...")
    finder = TextureFinder(roots, opts.texture_dir)

    ok = fail = skipped = 0
    for i, (sp, root) in enumerate(shapes, 1):
        if opts.out_dir:
            try:
                rel = sp.relative_to(root)
            except ValueError:
                rel = Path(sp.name)
            out = Path(opts.out_dir) / rel.with_suffix(".glb")
        else:
            out = sp.with_suffix(".glb")
        tag = f"[{i}/{len(shapes)}] {sp.name}"
        if opts.skip_existing and out.exists():
            print(f"{tag}  skipped (exists)")
            skipped += 1
            continue
        print(tag)
        try:
            mb = export(sp, out, finder, opts, log=print)
            print(f"    saved {out}  ({mb:.1f} MB)")
            ok += 1
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            if opts.debug:
                traceback.print_exc()
            fail += 1

    print(f"\nDone: {ok} converted, {fail} failed" + (f", {skipped} skipped" if skipped else "") + ".")
    if interactive and sys.stdin and sys.stdin.isatty():
        input("Press Enter to close...")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
