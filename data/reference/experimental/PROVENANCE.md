# Experimental reference data

Copied verbatim from the sister repository `km-dropplet-onto-bath`, at
`matlab/0_data/manual/Luke Paper Raw Data/`. These are the experiments that
supersede those of Alventosa, Cimpeanu & Harris (2023); the water case is
`R = 0.35 mm`, corresponding to `We = 1.0958`, `Bo = 0.017`, `Oh = 0.006`.

| File | Contents |
|---|---|
| `top_exp.csv` | Height of the droplet's top vs nondimensional time. |
| `bottom_exp.csv` | Height of the droplet's bottom vs nondimensional time. |
| `topbottom_exp.csv` | Both series concatenated. |
| `topbotom_errorbars_exp.csv` | Error-bar endpoints for the above (three rows per point: upper, lower, and the measurement). |

Columns are `t, z`, nondimensionalised by the inertio-capillary time and the
undeformed droplet radius respectively.

## What is NOT here, and why it matters

There is **no experimental contact radius and no experimental droplet width** in
this dataset. The `c_radius_*` and `width_*` series in the source directory come
only from DNS and from the 1PKM model — verified by reading the WebPlotDigitizer
projects `c_radius_dan.json` and `width_dan.json`, whose `datasetColl` contains
exactly two series each, named `*_1PKM` and `*_DNS`.

Any comparison of contact radius against those series is therefore a
model-to-simulation comparison, not a validation against measurement, and is
additionally sensitive to definition: the DNS contact radius is thresholded on
the trapped gas film (the radius at which the gap reaches twice the typical film
thickness), whereas this model's contact angle is the boundary of the region
where the kinematic match is imposed. Those need not coincide.
