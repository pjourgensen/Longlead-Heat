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
import pandas as pd

import json
import ftplib

class SSTPreparer:

  def __init__(self,config_json):
      with open(config_json) as json_file:
          config = json.load(json_file)

      self.yr_begin = config['yr_begin']
      self.yr_end = config['yr_end']
      self.out_path = config['out_path']
      self.data_fn = config['data_fn']
      self.target_path = config['target_path']
      self.min_lat = config['min_lat']
      self.max_lat = config['max_lat']
      self.min_lon = config['min_lon']
      self.max_lon = config['max_lon']
      self.nbases = config['nbases']

      self.data = None
      self.data_dates = None
      self.target = None
      self.target_dates = None
      self.coefs = None

  def load_sst_data(self):
      cont = input("Loading may take awhile. Are you sure you want to download data from {} to {}? ('y' or 'n')".format(self.yr_begin,self.yr_end))
      while cont.lower() != 'y' and cont.lower() != 'n':
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


  def combine_sst_data(self,save_data: bool=False):
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

  def load_target(self):
      self.target = xr.open_dataarray(self.target_path)

  def remove_leap_dates(self):
      self.target_dates = pd.to_datetime(self.target.time.values)
      target_nonleap = np.where(np.logical_or(self.target_dates.month!=2,self.target_dates.day!=29))[0]
      self.target_dates = self.target_dates[target_nonleap]
      self.target = self.target[target_nonleap]

      self.data_dates = pd.to_datetime(self.data.time.values)
      data_nonleap = np.where(np.logical_or(self.data_dates.month!=2,self.data_dates.day!=29))[0]
      self.data_dates = self.data_dates[data_nonleap]
      self.data = self.data[data_nonleap]

  def reshape_daily(self):
      days_in_year = 365
      num_years = self.yr_end - self.yr_begin + 1
      lat_len = len(self.data.lat)
      lon_len = len(self.data.lon)

      day_array = xr.DataArray(np.zeros((days_in_year,num_years,lat_len,lon_len)),
                               dims=['doy','year','lat','lon'],
                               coords={'doy':np.arange(1,days_in_year+1),
                                       'year':np.arange(self.yr_begin,self.yr_end+1),
                                       'lat':self.data.lat.values,
                                       'lon':self.data.lon.values})

      doy = 0
      for j in range(1,13):
          for i in range(1,32):
              curr = self.data[np.where(np.logical_and(self.data_dates.day==i,self.data_dates.month==j))]
              if curr.size > 0:
                  day_array[doy] = curr
                  doy += 1

      self.data = day_array

  def fit_trendline(self,day):

      year_len = len(day.year)
      lat_len = len(day.lat)
      lon_len = len(day.lon)

      doy_vec = np.arange(1,year_len+1)
      day = day.stack(ll=('lat','lon'))
      coefs = np.polyfit(doy_vec,day,deg=1)
      coefs = coefs.reshape((2,lat_len,lon_len))

      return coefs

  def fit_all_trendlines(self):
      days_in_year = 365
      num_coefs = 2
      lat_len = len(self.data.lat)
      lon_len = len(self.data.lon)
      self.coefs = xr.DataArray(np.zeros((days_in_year,num_coefs,lat_len,lon_len)),
                                dims=['doy','coef','lat','lon'],
                                coords={'doy':np.arange(1,days_in_year+1),
                                        'coef':['slope','intercept'],
                                        'lat':self.data.lat.values,
                                        'lon':self.data.lon.values})

      for day in self.data.doy.values:
          self.coefs.loc[day,:] = self.fit_trendline(self.data.loc[day,:])

  def remove_seasonality(self,coef):
      time_len = len(coef.doy)
      doy_vec = np.arange(1,time_len+1)

      t_basis = (doy_vec - 0.5)/365
      nt = len(t_basis)
      bases = np.empty((self.nbases, nt), dtype=complex)
      for counter in range(self.nbases):
          bases[counter, :] = np.exp(2*(counter + 1)*np.pi*1j*t_basis)

      coef -= coef.reduce(np.mean, dim='time')
      fourier_coef = 2/nt*(np.sum(bases[..., np.newaxis, np.newaxis]*coef.values[np.newaxis, ...], axis=1))
      rec = np.real(np.conj(fourier_coef[:, np.newaxis, ...])*bases[..., np.newaxis, np.newaxis])

      return coef - np.sum(rec, axis=0)

  def remove_coef_seasonality(self):
      slope = self.coefs.loc[:,'slope',:,:].copy(deep=True)
      intercept = self.coefs.loc[:,'intercept',:,:].copy(deep=True)
      self.coefs.loc[:,'slope',:,:] = self.remove_seasonality(slope) + self.coefs.loc[:,'slope',:,:].mean(axis=0)
      self.coefs.loc[:,'intercept',:,:] = self.remove_seasonality(intercept) + self.coefs.loc[:,'intercept',:,:].mean(axis=0)

  def remove_daily_trends(self):
      year_vec = np.arange(1,len(self.data.year)+1).reshape((1,len(self.data.year),1,1))
      self.data -= self.coefs.loc[:,'slope',:,:].expand_dims(dim='year',axis=1) * year_vec
      self.data = self.data.transpose('year','doy','lat','lon') - self.coefs.loc[:,'intercept',:,:]
      self.data = self.data.transpose('doy','year','lat','lon')

  def reshape_sequential(self):
      seq_len = len(self.data.doy) * len(self.data.year)
      lat_len = len(self.data.lat)
      lon_len = len(self.data.lon)

      data = xr.DataArray(np.zeros((seq_len,lat_len,lon_len)),
                         dims=['time','lat','lon'],
                         coords={'time':np.arange(1,seq_len+1),
                                 'lat':self.data.lat.values,
                                 'lon':self.data.lon.values})

      doy = 0
      for j in range(1,13):
          for i in range(1,32):
              day_idx = np.where(np.logical_and(self.data_dates.day==i,self.data_dates.month==j))[0]
              if len(day_idx) > 0:
                  data[day_idx,:,:] = self.data[doy,:,:,:]
                  doy += 1

      self.data = data

  def run_detrending(self):
      self.reshape_daily()
      self.fit_all_trendlines()
      self.remove_coef_seasonality()
      self.remove_daily_trends()
      self.reshape_sequential()

  def train_test_split(self):
      pass

  def prepare_target(self):
      pass

  def prepare(self):
      pass
