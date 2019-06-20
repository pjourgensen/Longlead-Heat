import os
import numpy as np
from netCDF4 import Dataset
import json

yr_begin = 1982
yr_end = 2015
filenames = [str(i)+'.nc' for i in list(range(1982,2016))]
output_filepath = "../Data/combined.json"

data = {}

for year in filenames:
    curr_year = Dataset("../Data/"+year,mode="r",format="NETCDF4")
    lats = curr_year.variables['lat'][:]
    lons = curr_year.variables['lon'][:]
    time = curr_year.variables['time'][:]
    anom = curr_year.variables['anom'][:]
    year_dict = {}

    for t in range(len(time)):
        print(t)
        day = anom[t].filled(0)
        year_dict[time[t]] = day

    data[year[:4]] = year_dict

with open(output_filepath,'w') as outfile:
    json.dump(data,outfile)
