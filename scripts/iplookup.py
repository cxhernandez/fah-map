#!/usr/bin/env python
import sys
import json
import numpy as np
import pandas as pd
import geoip2.database
from geopy.geocoders import Nominatim


if len(sys.argv) != 3:
    sys.exit('Please specify a GeoLite DB and an ip table.')

reader = geoip2.database.Reader(sys.argv[1])
geolocator = Nominatim()
places = ['city', 'town', 'county', 'state']


def get_city_from_coord(coord):
    try:
        address = geolocator.reverse(coord, timeout=10).raw['address']
        for place in places:
            if place in address.keys() and not address[place][1:].isdigit():
                return address[place]
    except:
        print 'A timeout occurred.'
    return 'unknown'


def get_city(entry, lang):
    if hasattr(entry, 'names') and lang in entry.names:
        return entry.names[lang]
    return 'unknown'


def get_location(addr):
    response = reader.city(addr)
    city = get_city(response.city, 'en')
    lat = response.location.latitude
    lng = response.location.longitude
    return (city, lat, lng)


ip = np.loadtxt(sys.argv[2], dtype=str)

locations = map(get_location, ip)

series = pd.Series(locations)
ucounts = series.value_counts()

info = []
for location, count in zip(ucounts.keys(), ucounts.get_values()):
    if location:
        if location[0] == 'unknown':
            location = (get_city_from_coord(location[1:]),) + location[1:]
        info.append({'city_name': location[0],
                     'lat': location[1],
                     'long': location[-1],
                     'nb_visits': count})

print json.dumps(info)
