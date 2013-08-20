#!/usr/bin/env python
# -*- coding: utf-8 -*-

import kplr

client = kplr.API()
targets = map(int, open("targets.txt").read().splitlines())
stars = [client.star(t) for t in targets]
for star in stars:
    tpfs = star.get_target_pixel_files(fetch=True)
    with open("{0}.txt", "w") as f:
        f.write("\n".join([d.filename for d in tpfs]))
