#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import kplr


client = kplr.API()
star = client.star(10592770)
stars = client.stars(ra=star.kic_degree_ra, dec=star.kic_dec, radius=2,
                     max_records=1000)

print(len(stars))
open("targets.txt", "w")
for s in stars:
    if len(s.get_light_curves()):
        with open("targets.txt", "a") as f:
            f.write("{0}\n".format(s.kepid))
