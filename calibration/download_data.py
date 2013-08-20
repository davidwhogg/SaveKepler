#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import kplr
import json

client = kplr.API()
targets = map(int, open("targets.txt").read().splitlines())
stars = [client.star(t) for t in targets]
for star in stars:
    try:
        os.makedirs("{0}/pixels".format(star.kepid))
    except os.error:
        pass

    # Save the information about the star.
    json.dump(star.params, open("{0}/params.json".format(star.kepid), "w"),
              sort_keys=True, separators=(",", ": "))

    # Save the file names.
    tpfs = star.get_target_pixel_files(fetch=True)
    with open("{0}/datafiles.txt".format(star.kepid), "w") as f:
        f.write("\n".join([d.filename for d in tpfs]))

    # Plot the time series.
    for i, tpf in enumerate(tpfs):
        fig = tpf.plot()
        print("{0}/pixels/{1:03d}.png".format(star.kepid, i))
        fig.savefig("{0}/pixels/{1:02d}.png".format(star.kepid,
                                                    tpf.sci_data_quarter))
