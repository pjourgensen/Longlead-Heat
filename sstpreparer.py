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
      self.data_fn = config['data_fn']
      self.min_lat = config['min_lat']
      self.max_lat = config['max_lat']
      self.min_lon = config['min_lon']
      self.max_lon = config['max_lon']
      self.nbases = config['nbases']
      self.data = None

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


  def combine_data(self,save_data=False):
      filenames = [str(i)+'.nc' for i in list(range(self.yr_begin,self.yr_end+1))]

      combined = xr.open_dataarray('Data/'+filenames[0])
      lat_idx = np.where(np.logical_and(combined['lat'].values >= self.min_lat,combined['lat'].values <= self.max_lat))[0]
      lon_idx = np.where(np.logical_and(combined['lon'].values >= self.min_lon,combined['lon'].values <= self.max_lon))[0]
      combined = combined[:,lat_idx,lon_idx]

      for i in filenames[1:]:
          print('Adding: {}'.format(i))
          curr = xr.open_dataarray('Data/'+i)[:,lat_idx,lon_idx]
          combined = xr.concat([combined,curr],dim='time')

      combined = combined.fillna(0)

      if save_data:
          combined.to_netcdf(self.out_path+self.data_fn)
      else:
          self.data = combined


  def train_test_split(self):
      pass

  def remove_trend(self):
      if self.data == None:
          self.data = xr.open_dataarray(self.out_path+self.data_fn)

      data = self.data.load()
      time_len = len(data['time'])
      lat_len = len(data['lat'])
      lon_len = len(data['lon'])

      doy_vec = np.arange(1,time_len+1)

      data = data.stack(ll=('lat','lon'))
      coefs = np.polyfit(doy_vec,data,deg=1)

      for i in range(2):
          data -= (np.power(doy_vec,2-(i+1))).reshape((time_len,1))*coefs[i].reshape((1,lat_len*lon_len))

      self.data = data.unstack()

  def remove_daily_trend(self):
      pass

  def remove_seasonality(self):
      if self.data == None:
          self.data = xr.open_dataarray(self.out_path+self.data_fn)

      data = self.data.load()  # need to use a couple times
      time_len = len(data['time'])

      doy_vec = np.arange(1,time_len+1)

      t_basis = (doy_vec - 0.5)/365
      nt = len(t_basis)
      bases = np.empty((self.nbases, nt), dtype=complex)
      for counter in range(self.nbases):
          bases[counter, :] = np.exp(2*(counter + 1)*np.pi*1j*t_basis)


      data -= data.reduce(np.mean, dim='time')

      coeff = 2/nt*(np.sum(bases[..., np.newaxis, np.newaxis]*data.values[np.newaxis, ...], axis=1))

      rec = np.real(np.conj(coeff[:, np.newaxis, ...])*bases[..., np.newaxis, np.newaxis])

      self.data = data - np.sum(rec, axis=0)


  def prepare_target(self):
      pass

  def prepare(self):
      pass
