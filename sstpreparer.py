"""
This script is intended to process data into a format to be presented to
a neural network model. The steps for doing so include:
  - Load the individual netcdf files corresponding to annual SST anomaly data
  - Combine them into single xarray, with specification of lat/lon ranges
  - Remove the linear trend over time from the selected grid space
  - Remove the seasonality over time from the selected grid space
"""

import numpy as np
import xarray as xr

import ftplib

class SSTPreparer:

  def __init__(self,config):
      self.yr_begin = config['yr_begin']
      self.yr_end = config['yr_end']
      self.out_path = config['out_path']

  def load_data(self):
      cont = input("Loading may take awhile. Are you sure you want to download data from {} to {}? ('y' or 'n')".format(self.yr_begin,self.yr_end))
      while cont.lower() != 'y' or cont.lower() != 'n':
          cont = input("Please enter 'y' or 'n': ")

      if cont.lower() == 'n':
          print("Exiting load sequence...")
          return
      else:
          host_name = "ftp.cdc.noaa.gov"
          username = "anonymous"
          password = "xxxx"
          path = "/Datasets/noaa.oisst.v2.highres/"

          filenames = ["sst.day.anom."+str(i)+".nc" for i in list(range(self.yr_begin,self.yr_end+1))]

          ftp = ftplib.FTP(host = host_name)
          ftp.login(username,password)
          ftp.cwd(path)

          for j in filenames:
              print("loading "+j[-7:-3])
              ftp.retrbinary("RETR "+j,open(self.out_path+j[-7:],"wb").write)


  def combine_data(self):
      pass

  def train_test_split(self):
      pass

  def remove_trend(self):
      pass

  def remove_seasonality(self):
      pass

  def prepare_target(self):
      pass

  def prepare(self):
      pass
