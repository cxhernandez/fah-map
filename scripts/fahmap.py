#!/usr/bin/env python
import sys
import heatmap
import numpy as np
import pandas as pd
from PIL import Image
import geoip2.database



if len(sys.argv) != 5:
    sys.exit('Please specify a GeoLite DB, FAH DB, number of days to access, and world map (PNG).')

reader = geoip2.database.Reader(sys.argv[1])
world = Image.open(sys.argv[-1])

def get_city(entry, lang):
    if hasattr(entry, 'names') and lang in entry.names:
        return entry.names[lang]
    return 'unknown'


def get_coord(addr):
    response = reader.city(addr)
    lat = response.location.latitude
    lng = response.location.longitude
    return (lng, lat)


def get_heatmap(pts, area = ((-180,-80),(180,80)), scheme="Greys", size=(1000,445), dotsize=6, opacity=100):
    hm = heatmap.Heatmap()
    return hm.heatmap(pts, area = area, scheme = scheme,
                      size= size, dotsize = dotsize,opacity = opacity)


ip = np.loadtxt(sys.argv[2], dtype=str)

coords = map(get_coord, ip)

series = pd.Series(coords)
ucounts = series.value_counts()

pts = []
for coord, count in zip(ucounts.keys(), ucounts.get_values()):
    if coord[0] and coord[1]:
        for _ in xrange(count):
            pts.append((coord[0],#+np.random.normal(0,.1),
                        coord[1]))#+np.random.normal(0,.1)))

img = get_heatmap(pts, size = world.size, scheme="custom")

Image.alpha_composite(world,img).save('./static/png/past' + str(sys.argv[-2]) + '.png')
