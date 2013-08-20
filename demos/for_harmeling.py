#!/usr/bin/env python
# -*- coding: utf-8 -*-

import kplr

client = kplr.API()

kois = client.kois(where=("koi_period BETWEEN 1 AND 2 AND "
                          "koi_count=1 AND koi_disposition='CANDIDATE' AND "
                          "koi_ror BETWEEN 0.08 AND 0.1"),
                   sort=("koi_ror", -1))

pixel_files = kois[1].get_target_pixel_files()
# ...
# the usual stuff.
