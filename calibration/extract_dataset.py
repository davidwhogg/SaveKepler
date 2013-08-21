#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fitsio
import numpy as np
from itertools import product

out_fn = "dataset.txt"
quarter = 10
tmin, tmax = 937.37407678, 969.65921365

targets = map(int, open("targets.txt").read().splitlines())

npts = None
open(out_fn, "w")
for target in targets:
    files = open("{0}/datafiles.txt".format(target)).read().splitlines()
    for fn in files:
        header = fitsio.read_header(fn)
        if header["QUARTER"] == quarter:
            break
    assert header["QUARTER"] == quarter
    data = fitsio.read(fn)

    t = data["TIME"]
    fluxes = data["FLUX"]

    inds = (t > tmin) * (t < tmax) * np.isfinite(t)
    assert npts is None or np.sum(inds) == npts
    npts = np.sum(inds)

    t = t[inds]
    fluxes = fluxes[inds]

    mask = np.sum(np.isfinite(fluxes), axis=0) > 1
    print(np.arange(len(fluxes))[~np.isfinite(fluxes[:, mask][:, 0])])

    # Loop over pixels.
    nx, ny = np.shape(fluxes[0])
    for ix, iy in product(range(nx), range(ny)):
        flux = np.array([f[ix, iy] for f in fluxes])
        if not np.sum(np.isfinite(flux)):
            continue

        with open(out_fn, "a") as f:
            f.write("{0} {1} {2} ".format(target, ix, iy))
            f.write(" ".join(map("{0}".format, flux)))
            f.write("\n")
