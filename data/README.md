## EUP<sup>3</sup>M: Evaluating uncertainty and predictive performance of probabilistic models
## <font color="#cc0066">Test Data</font>

## Introduction

   This directory contains raw input data used in the experiments [1].
   The data is made available courtesy of Rio Tinto Kennecott Copper (RTKC).
   Commercially sensitive information have been removed to make data sharing
   possible in accordance with FAIR principles [2,3]. This does not compromise
   the scientific integrity of the data, as explained in Note 1 below.


## General principles

1. To protect the interests of RTKC, location and time sensitive information
   are either removed or reported in modified form. This is feasible since
   the absolute geographic or mine coordinates are inconsequential for this
   study. Therefore, only relative coordinates or scales are preserved in the
   sanitised data. Specifically, the shared data reports only the *relative*
   locations rather than the *actual* locations where blasthole samples were
   taken and where grade predictions are required. All spatial coordinates
   are anonymised following the data curation procedure below.
2. The geological setting of the Bingham Canyon (Kennecott) Copper Mine is
   of interest to the readers of this paper. Discussion is limited to what is
   available in the public domain. In particular, geochemical and mineralogical
   interpretations are based on [4,5,6].



## Data curation

1. The original RTKC coordinates are shifted by an arbitrary amount to conceal
   the actual location of all blastholes, predicted or validation points.
   This translation brings the minimum coordinates of the modelled region
   close to the origin, (0,0,0), for convenience.
2. There is no mention of the site coordinates, pits, true bench elevation/RL
   in relation to the Bingham Canyon (Kennecott) open-pit copper mine.
3. The assigned domain IDs were looked up using the blocks_insitu model.
4. Unused columns from the CSV files (generally, anything other than X, Y, Z,
   the Cu grade and domain ID) have been stripped.


## Contents

In the current directory,
- `domain_ellipse_geometry.csv` describes the domain geometry (rotation and scaling parameters).
- `domain_samples_summary.csv` describes the training blasthole and inference sample counts, as well as the copper grade percentiles in each geological domain `gD` and inference period `mA`.

Each subdirectory `mA_mB_mC` provides the relevant data for inference period `mA`.
- `blastholes_tagged.csv` contains the relative coordinates (northing, easting, RL) of blastholes where assays were taken. The copper grade and relevant domain ID are reported in the `PL_CU` and `lookup_domain` columns.
- `blocks_to_estimate_tagged.csv` provides the inference locations where grade estimation is required via the `X`, `Y`, `Z` columns, the geologist assigned domains via the `domain` column, and validation measurements (groundtruth) via the `cu_bh_nn` column. This file is used only for future-bench prediction (grade extrapolation).
- `blocks_insitu_tagged.csv` is similar, except it is used only for in-situ regression (grade interpolation).


## Referenced Papers

- [1] Raymond Leung, Alexander Lowe and Arman Melkumyan, "Evaluating uncertainty
   and predictive performance of probabilistic models devised for grade
   estimation in a porphyry copper deposit", preprint available in
   [/docs](/docs/background.pdf), 2024.
- [2] Mark D. Wilkinson et al., "The FAIR Guiding Principles for scientific data
   management and stewardship", Scientific Data, 3, Article 160018, Nature 2016
   https://rdm.mpdl.mpg.de/after-research/open-research-data/
- [3] Michelle Barker et al., "Introducing the FAIR principles for research software",
   Scientific Data, 9, Article 622, Nature 2022.
   https://www.nature.com/articles/s41597-022-01710-x
- [4] J.P. Porter, K. Schroeder and G. Austin. "Geology of the Bingham Canyon
   porphyry Cu-Mo-Au deposit, Utah." Society of Economic Geologists, 2012.
   ISBN 9781629490410. doi: https://doi.org/10.5382/SP.16.
- [5] P.B. Redmond and M.T. Einaudi. "The Bingham Canyon porphyry Cu-Mo-Au
   deposit. I. Sequence of intrusions, vein formation, and sulfide deposition."
   Economic Geology, 105:43-68, 2010.
- [6] R. Hayes and S. McInerney. "Rio Tinto Kennecott mineral resources and ore
   reserves." ASX Notice, September 2022. URL
   https://minedocs.com/23/Kennecott_PR_09272022.pdf.
