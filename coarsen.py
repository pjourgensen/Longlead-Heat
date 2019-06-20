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
        condensed = np.zeros((180,360))
        curr_date = anom[t]
        for lat in range(0,len(lats),4):
            for lon in range(0,len(lons),4):
                sst = curr_date[lat:lat+4,lon:lon+4].mean()
                if sst is np.ma.masked:
                    sst = 0
                condensed[int(lat/4),int(lon/4)] = sst

        year_dict[time[t]] = condensed

    data[year[:4]] = year_dict

with open(output_filepath,'w') as outfile:
    json.dump(data,outfile)
