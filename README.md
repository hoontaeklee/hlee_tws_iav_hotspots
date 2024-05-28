# Description

This repository contains Python scripts to perform analyses and create figures of the following article:

Lee, H., Jung, M., Carvalhais, N., Trautmann, T., Kraft, B., Reichstein, M., Forkel, M., and Koirala, S.: Diagnosing modeling errors in global terrestrial water storage interannual variability, Hydrol. Earth Syst. Sci., 27, 1531–1563, https://doi.org/10.5194/hess-27-1531-2023, 2023.

# Disclaimer

This repository is created to support the manuscript mentioned above. Any usage beyond the intended purpose are the responsibility of the users.

# Structure

- `data`: The model simulations and processed data sets, which can be accessed via [https:zenodo](https://zenodo.org/records/7813179).
  - `SINDBAD_opti_Nov2021`: Two sets of SINDBAD simulations. One is using MSWEP precipitation forcing in `VEG_MSWEP` directory. The other one is using gpcp1dd v1.3 precipitation forcing. For the set with gpcp1dd, the river water storage is in the `Routing_VEG_3_0.5` directory.
  - `h2m`: Two sets of H2M simulations. One is using gpcp1dd v1.3 precipitation forcing, the other one is using MSWEP.
- `script`: scripts to perform the analyses and create figures. For the latter, scripts for corresponding figures are named as `plot_fig#_r1.py`.
  - `calc_cov.py`: the code to perform the covariance matrix analysis with examples of terrestrial water storage (TWS) interannual variability (IAV).
  - `calc_detrend_data.py`: the code to detrend time series with examples of TWS IAV.
  - `calc_rsq_change_with_trimming.py`: the code to calculate model performance (r2) with increasing the trimming percentage.
- `plot`: This directory stores figures created by scripts `plot_fig#_r1.py` in `script` directory.

# Citation

@Article{hess-27-1531-2023,
AUTHOR = {Lee, H. and Jung, M. and Carvalhais, N. and Trautmann, T. and Kraft, B. and Reichstein, M. and Forkel, M. and Koirala, S.},
TITLE = {Diagnosing modeling errors in global terrestrial water storage interannual variability},
JOURNAL = {Hydrology and Earth System Sciences},
VOLUME = {27},
YEAR = {2023},
NUMBER = {7},
PAGES = {1531--1563},
URL = {https://hess.copernicus.org/articles/27/1531/2023/},
DOI = {10.5194/hess-27-1531-2023}
}

For the SINDBAD model, please cite:

@Article{hess-26-1089-2022,
AUTHOR = {Trautmann, T. and Koirala, S. and Carvalhais, N. and G\"untner, A. and Jung, M.},
TITLE = {The importance of vegetation in understanding terrestrial water storage variations},
JOURNAL = {Hydrology and Earth System Sciences},
VOLUME = {26},
YEAR = {2022},
NUMBER = {4},
PAGES = {1089--1109},
URL = {https://hess.copernicus.org/articles/26/1089/2022/},
DOI = {10.5194/hess-26-1089-2022}
}

For the H2M model, please cite:

@Article{hess-26-1579-2022,
AUTHOR = {Kraft, B. and Jung, M. and K\"orner, M. and Koirala, S. and Reichstein, M.},
TITLE = {Towards hybrid modeling of the global hydrological cycle},
JOURNAL = {Hydrology and Earth System Sciences},
VOLUME = {26},
YEAR = {2022},
NUMBER = {6},
PAGES = {1579--1614},
URL = {https://hess.copernicus.org/articles/26/1579/2022/},
DOI = {10.5194/hess-26-1579-2022}
}

