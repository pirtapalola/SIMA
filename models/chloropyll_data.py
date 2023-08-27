#!/usr/bin/env python

"""

Upload water quality monitoring data from the Great Barrier Reef (GBR)

Dataset citation:
Australian Institute of Marine Science (AIMS) (2023).
Great Barrier Reef Marine Monitoring Program for Inshore Water Quality
- Vertical Profiles of Conductivity Temperature and Depth (CTD).
https://doi.org/10.25845/5b6k-d875, accessed 27-Aug-2023.

Last updated by Pirta Palola on 27 August 2023.

"""

import codecs
import csv
from urllib.request import urlopen

# Specify the URL to the dataset.
gbr_data_url = "https://geoserver-portal.aodn.org.au/geoserver/" \
                 "ows?typeName=imos:aims_mmp_ctd_profiles_data&SERVICE" \
                 "=WFS&outputFormat=csv&REQUEST=GetFeature&VERSION=" \
                 "1.0.0&CQL_FILTER=INTERSECTS(geom%2CPOLYGON((144.931640625%20-" \
                 "25.533203125%2C144.931640625%20-14.810546875%2C153.369140625%20" \
                 "-14.810546875%2C153.369140625%20-25.533203125%2C144.931640625%20-25.533203125)))&userId=Guest"

# Get the data
gbr_dataset = urlopen(gbr_data_url)

# Save the csv file.
csvfile = csv.reader(codecs.iterdecode(gbr_dataset, 'utf-8'))  # a csv reader object
chlorophyll_gbr = [row for row in csvfile]  # a list
