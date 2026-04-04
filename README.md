# Statistical Evidence for a Kinematic Substructure in the Solar Neighborhood

draft submitted to ApJ

## Repository structure

```
data/
  raw/gcs1.fits              # Original GCS catalog (Holmberg+2007)
  processed/                 # Cleaned subsamples (all stars, F, G)
notebooks/
  01_data_cleaning.ipynb
  02_completeness_vmax.ipynb
  03_spatial_overview.ipynb
  04_vsini_maps.ipynb
  05_statistical_tests.ipynb
  06_robustness_uncertainty.ipynb
src/
  style.py       # Plot style and rcParams
  spatial.py     # Spatial grid maps and interpolation
  statistics.py  # Weighted stats, 1/Vmax weights, permutation tests, Moran's I
  plotting.py    # All figure functions
figures/         # Output figures
```

## Authors

Sarah G. A. Barbosa, Daniel B. de Freitas, Wagner J. B. Corradi, Paulo C. F. da Silva Filho, Kento Masuda, José R. de Medeiros
