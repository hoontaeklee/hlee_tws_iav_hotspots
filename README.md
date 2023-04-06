# Description

This repository contains scripts to perform analyses and create figures of an article:

for transparency.

Please feel free to ask any questions.

# Structure

- `data`: The model simulations and processed data sets, which can be accessed via https:zenodo.
  - `SINDBAD_opti_Nov2021`: Two sets of SINDBAD simulations. One is using MSWEP precipitation forcing in `VEG_MSWEP` directory. The other one is using gpcp1dd v1.3 precipitation forcing. For the set with gpcp1dd, the river water storage is in the `Routing_VEG_3_0.5` directory.
  - `h2m`: Two sets of H2M simulations. One is using gpcp1dd v1.3 precipitation forcing, the other one is using MSWEP.
- `script`: scripts to perform the analyses and create figures. For the latter, scripts for corresponding figures are named as `plot_fig#_r1.py`.
  - `calc_cov.py`: the code to perform the covariance matrix analysis with examples of terrestrial water storage (TWS) interannual variability (IAV).
  - `calc_detrend_data.py`: the code to detrend time series with examples of TWS IAV.
  - `calc_rsq_change_with_trimming.py`: the code to calculate model performance (r2) with increasing the trimming percentage.
- `plot`: This directory stores figures created by scripts `plot_fig#_r1.py` in `script` directory.

# Citation
