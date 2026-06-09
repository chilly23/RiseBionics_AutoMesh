# Rise Bionics Engineering ML Assessment Submission

This package contains the refined deterministic Python solution and generated OBJ outputs.

## Run

```powershell
py -3 assessment_solution.py EngineeringMLAssessment.obj --out assessment_output_refined
```

## Method

- Uses edge detection on the supplied texture atlas as the flattened 2D plane.
- Snaps labelled E/C/S regions to the visible pen outlines in UV space.
- Maps the refined atlas masks back to the OBJ mesh through UV coordinates.
- Applies E/C as continuous normal displacement with a soft boundary falloff, avoiding torn boundary triangles or voids.
- Applies stronger local Laplacian smoothing to the S region.
- Performs cube union/subtract outputs from the ECS-combined mesh.

## Main Outputs

- `01_edge_detection_mask.png`
- `02_flattened_shapes_labeled.png`
- `03_E_extruded.obj`
- `04_C_pocketed.obj`
- `05_S_smoothed.obj`
- `06_ECS_combined.obj`
- `07_union_cube_bottom.obj`, `07_union_cube_mid.obj`, `07_union_cube_top.obj`
- `08_subtract_cube_bottom.obj`, `08_subtract_cube_mid.obj`, `08_subtract_cube_top.obj`
