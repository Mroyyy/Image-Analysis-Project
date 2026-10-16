# Nuclei segmentation benchmark for multiplex immunofluorescence (lung cancer TMA)

Code for the report **"Segmentation choice shifts absolute CD8+ T-cell density but preserves patient ranking and continuous survival estimates in lung cancer"** (Applied Precision Medicine).

The project **compares three nuclei-segmentation methods** (watershed, StarDist, Cellpose) applied to the **DAPI channel of multiplex immunofluorescence tissue microarray (TMA) images**, and follows the choice of method through to CD8+ T-cell density, patient ranking and survival analysis. All steps after segmentation are identical for the three methods, so any **difference in the results comes from the segmentation alone**.

There is **no manual ground truth**, so the code does not say which method is more accurate. **It measures whether the choice matters for the downstream results.**
