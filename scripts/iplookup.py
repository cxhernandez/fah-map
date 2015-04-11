import sys
import json
import numpy as np
import pandas as pd
import geoip2.database


if len(sys.argv) != 3:
    sys.exit('Please specify a GeoLite DB and an ip table.')

reader = geoip2.database.Reader(sys.argv[1])


def get_name(entry, lang):
    if hasattr(entry, 'names') and lang in entry.names:
        return entry.names[lang]

    return 'unknown'


def get_location(addr):
    response = reader.city(addr)
    city = get_name(response.city, 'en')
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
        info.append({'city_name': location[0],
                     'lat': location[1],
                     'long': location[-1],
                     'nb_visits': count})

print json.dumps(info)
