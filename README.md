# Seoul Fire Flood Accessibility

Code, derived data, figures, and manuscript materials for the CEUS study:
`A hybrid geospatial framework for flood-induced fire accessibility: Integrating weighted network analysis and structural diagnostics in Seoul`.

## Repository purpose

This repository is organized as a publication companion repository for peer review and reproducibility. It contains:

- final manuscript files for submission
- figure assets used in the paper
- analysis scripts used to reproduce the main outputs
- lightweight derived datasets required for the reported results
- supplementary and submission metadata

## Repository structure

- `manuscript/`
  - `CEUS_manuscript_anonymized_final.docx`: anonymized manuscript for double-anonymized review
  - `CEUS_title_page_final.docx`: title page with author metadata and declarations
  - `CEUS_manuscript_submission_ready_v32.docx`: internal full working version
- `figures/`
  - final Figure 1-5 PNG files used in the submission package
- `code/`
  - scripts used to generate the main analysis outputs and publication figures
- `data/`
  - compact derived datasets used in the manuscript
- `supplementary/`
  - supplementary note for sensitivity analysis
- `metadata/`
  - submission checklist and generative AI declaration
- `environment/`
  - Python package requirements used for the analysis environment

## Data notes

The repository includes the compact derived datasets needed to inspect and reproduce the reported results:

- `table1_population_building_weighted.csv`
- `gap_analysis_population_building_weighted.csv`
- `structural_operational_correlation.csv`
- `fire_station_accessibility_summary_gridbased.csv`
- `population_grid_250m.gpkg`
- `weighted_demand_points.gpkg`
- `priority_defense_corridors_rp100_strategic.gpkg`

The full `priority_defense_corridors.gpkg` file used during the internal workflow is omitted from this GitHub repository because its size exceeds standard GitHub repository limits. The reduced `priority_defense_corridors_rp100_strategic.gpkg` file contains the RP100 strategic-corridor subset used for the paper's final corridor figure. A full archival package can be deposited separately in a citable repository such as Zenodo.

## Reproducibility

The main scripts are:

- `code/run_population_building_weighted_rp_analysis.py`
- `code/generate_publication_multifigure.py`
- `code/generate_workflow_and_hotspot_figures.py`

Python dependencies are listed in:

- `environment/requirements.txt`

## Review and disclosure notes

- The anonymized manuscript is intended for journal peer review.
- The title page contains author-identifying information and declarations and should not be uploaded as the anonymized manuscript file.
- Zenodo archive DOI: https://doi.org/10.5281/zenodo.19137721

## Citation

If you use this repository, cite the associated article manuscript and this repository record. A machine-readable citation file is provided as `CITATION.cff`.


## Archive DOI

- Zenodo DOI: https://doi.org/10.5281/zenodo.19137721
