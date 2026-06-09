# Rise Bionics Atlas Pipeline

Author: Velan

A 3D mesh processing pipeline for OBJ models with UV-mapped texture atlases.

The application automatically detects labeled regions from a texture atlas and performs geometry operations on the corresponding mesh regions:

- E → Extrude
- C → Pocket / Cut
- S → Preserve original geometry

The system provides:

- OBJ mesh loading
- Texture atlas detection
- UV-to-face mapping
- Region segmentation
- Face extraction
- Mesh extrusion
- Mesh pocketing
- Boolean operations
- Interactive GUI
- Interactive 3D viewer
- Export of processed meshes

---

## Features

### Mesh Loading

- Loads OBJ files with UV coordinates
- Automatically detects texture atlases
- Supports MTL texture references

### Shape Detection

Detects predefined atlas regions:

- Rectangles
- Circles
- Custom polygon regions

Labels:

| Label | Operation |
|---------|---------|
| E | Extrude |
| C | Pocket |
| S | Preserve |

### Mesh Operations

#### Extrusion

Extrudes selected regions outward.

#### Pocket

Cuts selected regions inward.

#### Combined ECS

Applies all atlas operations into a single mesh.

### Boolean Operations

Creates:

- Union Top
- Union Mid
- Union Bottom
- Subtract Top
- Subtract Mid
- Subtract Bottom

### GUI

Includes:

- OBJ file browser
- Atlas preview
- Detection preview
- Operation controls
- Live logging
- Interactive 3D visualization

---

## Installation

```bash
pip install -r requirements.txt