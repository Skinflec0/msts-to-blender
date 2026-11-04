# msts-to-glb

Convert **Microsoft Train Simulator / Open Rails** models (`.s` shapes and `.ace` textures) into `.glb` files that open straight in **Blender**, three.js, Godot, Unity and anything else that reads glTF 2.0.

One script, no build step. Point it at a folder and it converts everything in it.

## Install

You need Python 3.8+.

```bash
pip install numpy Pillow
```

## Use

```bash
python msts_to_glb.py                    # no arguments: a folder picker opens
python msts_to_glb.py path/to/folder     # every .s file inside, including subfolders
python msts_to_glb.py wagon.s            # one file
python msts_to_glb.py rolling_stock -o glb_out
```

On Windows you can also drag a folder or `.s` files onto `msts_to_glb.py`.

Each `.glb` is written next to its `.s` file (or into `--out-dir`). In Blender: **File > Import > glTF 2.0**.

Textures are found automatically: the shape's own folder, `TEXTURES` folders beside it, then anywhere inside the folder you passed. Use `-t` to add more search folders.

## Options

| Option | What it does |
| --- | --- |
| `-o, --out-dir DIR` | Write output here, mirroring the input folder layout |
| `-t, --texture-dir DIR` | Extra texture folder (repeatable) |
| `--lod N` | Export level of detail N; `0` is the most detailed (default) |
| `--skip-existing` | Leave existing `.glb` files alone |
| `--no-textures` | Geometry only |
| `--double-sided` | Render both sides of every face |
| `--zfight METRES` | Nudge vertices along their normals to hide z-fighting, e.g. `0.002` (off by default) |
| `--debug` | Print full tracebacks on errors |

## What converts

- Geometry, normals and UVs, with the left-handed MSTS axes converted to glTF
- The node hierarchy (bogies, doors, wheels) with their transforms
- `.ace` textures: uncompressed RGB/RGBA, with or without the zlib wrapper
- Alpha: cut-out and soft transparency are picked from the texture's alpha channel
- Untextured faces get a plain grey material

## Known limits

- **Text-format `.s` files** are not read. Re-save them as binary first (Shape Viewer or `ffeditc_unicode.exe`).
- **DXT-compressed `.ace` files** are not decoded. The shape still converts and uses a grey placeholder; the log tells you which texture was affected.
- Animations are not exported, only the static pose.
- Only one level of detail is exported per run.

If a file fails, the error is printed and the rest of the batch continues. Re-run with `--debug` and open an issue with the traceback.

## Changes in 2.0

- Removed the per-triangle "z-fight" push that moved shared vertices several times and distorted meshes. It is now an optional, vectorised `--zfight`.
- Windows consoles no longer crash on the old Unicode check marks.
- Texture folders are indexed once instead of once per shape.
- Vertices with zero or several UV sets no longer break parsing; untextured faces no longer crash.
- Alpha shader detection reads shader names when present instead of assuming index 1.
- Clear error messages replace bare `IndexError`s; the glTF buffer length is now exact.
- New: folder picker, `--out-dir`, `--lod`, `--skip-existing`, proper exit codes.

## Licence

MIT
