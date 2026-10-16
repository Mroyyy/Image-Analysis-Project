# Nuclei segmentation benchmark for multiplex immunofluorescence (lung cancer TMA)

Code for the report **"Segmentation choice shifts absolute CD8+ T-cell density but preserves patient ranking and continuous survival estimates in lung cancer"** (Applied Precision Medicine).

The project **compares three nuclei-segmentation methods** (watershed, StarDist, Cellpose) applied to the **DAPI channel of multiplex immunofluorescence tissue microarray (TMA) images**, and follows the choice of method through to CD8+ T-cell density, patient ranking and survival analysis. All steps after segmentation are identical for the three methods, so any **difference in the results comes from the segmentation alone**.

There is **no manual ground truth**, so the code does not say which method is more accurate. **It measures whether the choice matters for the downstream results.**

## Main findings
- **Watershed detected fewer nuclei** per core (mean 4,655) than StarDist (7,545) and Cellpose (7,280).
- Median patient-level **CD8+ density was 37–38% lower with watershed** than with the deep-learning methods (fold-change ×0.63).
- **Patient ranking was almost identical** between methods (Spearman ρ = 0.994–0.996).
- Continuous Cox models gave the **same hazard ratio per SD of log-density for all methods** (HR 0.85, 95% CI 0.59–1.21); no method showed a significant survival association.
- Median-split log-rank p-values differed between methods (0.10–0.30).

_See the report for the full results and the limitations._

## Repository contents
- _pipeline_segmentation_benchmark.ipynb_	Main notebook: masks, segmentation, phenotyping, agreement analysis, survival analysis, export of tables
- _stardist_runner_batch.py_	Runs StarDist (2D_versatile_fluo) on all cores; called from the notebook through a separate environment
- _cellpose_runner_batch.py_	Runs Cellpose (cpsam) on all cores and writes its own configuration record
- _image_analysis.yml_, _stardist.yml_, _cellpose.yml_	Conda environments (see below)
- _cellpose_manifest.json_ Exact parameters used for the reported results from Cellpose
- _stardist_manifest.json_ Exact parameters used for the reported results from StarDist

## Data (not included)
The images and clinical table were provided by the course organizers and are not included in this repository.

Images: one .ome.tif per TMA core, 3 channels (CD8, CK, DAPI), 0.5 µm/pixel; 75 cores, of which 73 (from 46 patients) were analyzed.
Clinical table: one row per core, with a patient ID column used to pool cores per patient, plus survival time, event status, age, sex and stage.

To rerun the analysis you need the same kind of data. Set the paths in Section 1 of the notebook:
```
DATA_DIR     = "./Data/DAPI_PanCK_CD8"
CLINICAL_CSV = "./Data/samples_clinical_merged.csv"
```

## Environments
Three separate environments are used because the deep-learning packages require different versions of the same libraries.

Environment	Used for	Key packages
notebook	masks, watershed, statistics, figures	scikit-image 0.26.0, plus pandas, numpy, scipy, matplotlib and a survival-analysis package (see env_notebook.yml)
StarDist	StarDist segmentation	stardist 0.9.2, csbdeep 0.8.2, TensorFlow 2.21.0
Cellpose	Cellpose segmentation	cellpose 4.2.1.1, torch 2.5.1

Create each environment from its file, for example:
```
conda env create -f env_notebook.yml
conda env create -f env_stardist.yml
conda env create -f env_cellpose.yml
```
StarDist and Cellpose were run on an NVIDIA GTX 1050 Max-Q GPU; watershed ran on the CPU.


## How to run
Put the data in the folders set in Section 1 of the notebook.
In Section 1, set the paths to the Python interpreters and runner scripts of the StarDist and Cellpose environments:
```
 STARDIST_PYTHON       = "<path to the StarDist environment's python>"
 STARDIST_RUNNER_BATCH = "<path to stardist_runner_batch.py>"
 CELLPOSE_PYTHON       = "<path to the Cellpose environment's python>"
 CELLPOSE_RUNNER       = "./cellpose_runner_batch.py"
```
Open the notebook in the notebook environment and run the cells in order. Segmentation outputs are saved as one .npy label image per core and method (seg_outputs/, stardist_outputs/) and are reused on later runs. Delete a folder to recompute it.
The final section writes the result tables to report_tables/<setting>/.


## Notebook outline
1. Setup and configuration
2. Utilities
3. Tissue, tumour and CD8 masks from the raw channels (independent of segmentation)
4. Nuclei segmentation with the three methods and their recorded configuration
5. Phenotyping (CD8+ call in a 3 µm ring) and tumour/stroma assignment
6. Descriptive comparison of the segmentations
7. Clinical data and per-method patient density tables
8. Agreement between methods (IoU matching, Spearman, Bland-Altman)
9. Survival analysis per method (Cox, median split, bootstrap, sensitivity analyses)
10. Export of the tables

## Key parameters
Pixel size: 0.5 µm.
CD8 mask: Gaussian σ = 1, intensity threshold > 150 (chosen by visual inspection; no outcome data were used).
CD8+ call: nucleus dilated by 6 px (3 µm ring), nucleus removed; called CD8+ if more than 50% of the ring overlaps the CD8 mask.
Density: CD8+ nuclei per mm² of compartment (total tissue, tumour, stroma), pooled per patient.
Survival: time from diagnosis to death from any cause or last follow-up.

## Output files
Written to report_tables/<setting>/:

segmentation_summary.csv: nuclei counts per core and method
density_agreement.csv: agreement between methods at patient level
survival_results.csv: hazard ratios, confidence intervals and p-values per method
bootstrap_hr_differences.csv: bootstrap differences in hazard ratios between methods
cd8_call_sensitivity.csv: sensitivity of the CD8+ call
methods_configuration.json: parameters and software versions

Patient-level tables (patient_table_*.csv) contain clinical data and are not included in the repository.

## Limitations
No manual annotation, so no accuracy claims.
Small single cohort with no detectable survival association, so the study cannot show how segmentation would affect a true effect.
The CD8 threshold was chosen by visual inspection.
Two cores with uneven illumination were excluded.
