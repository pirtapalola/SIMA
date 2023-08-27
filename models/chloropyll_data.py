#!/usr/bin/env python
import codecs
import csv
from urllib.request import urlopen

# The URL to the collection (as comma-separated values).
collection_url = "https://geoserver-portal.aodn.org.au/geoserver/" \
                 "ows?typeName=imos:aims_mmp_ctd_profiles_data&SERVICE" \
                 "=WFS&outputFormat=csv&REQUEST=GetFeature&VERSION=" \
                 "1.0.0&CQL_FILTER=INTERSECTS(geom%2CPOLYGON((144.931640625%20-" \
                 "25.533203125%2C144.931640625%20-14.810546875%2C153.369140625%20" \
                 "-14.810546875%2C153.369140625%20-25.533203125%2C144.931640625%20-25.533203125)))&userId=Guest"

# Fetch data...
response = urlopen(collection_url)

# Iterate on data...
csvfile = csv.reader(codecs.iterdecode(response, 'utf-8'))
for row in csvfile:
    print(row)
