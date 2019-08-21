
import numpy as np
import xarray as xr
import pandas as pd

import json
import ftplib

class SSTPreparer:
    """
    A class used to load SST data and detrend it to prepare it for use with a ConvLSTM model.
    ...

    Attributes
    ----------
    config_json: str
        file path to a config file that contains "Config Attributes"
    download_raw_sst: bool
        True if user needs to download raw SST data from NOAA
    combine_raw_sst: bool
        True if user has not performed detrending on raw SST data
    load_raw_t95: bool
        True if user has not already encoded target
    data: xarray.core.dataarray.DataArray
        current state of the SST data
    data_dates: pandas.core.indexes.datetimes.DatetimeIndex
        dates corresponding to each entry of data
    lat_len: int
        number of latitude indices
    lon_len: int
        number of longitude indices
    channel: int
        number of channels (1)
    target: xarray.core.dataarray.DataArray
        current state of the target
    target_dates: pandas.core.indexes.datetimes.DatetimeIndex
        dates corresponding to each entry of the target

    Config Attributes
    -----------------
    yr_begin: int
        first year of SST data to be included
    yr_end: int
        last year of SST data to be included
    data_path: str
        file path to where data will be stored or accessed
    sst_inpath: str
        file path to detrended SST data (if available)
    sst_outpath: str
        file path to store SST data after it's been detrended
    target_inpath: str
        file path to t95 data if NOT encoded OR target data if encoded
    target_outpath: str
        file path to store target data after it's been encoded
    target_levels: list
        descending list of percentiles (85th percentile -> 85, not 0.85) with which to encode target
    min_lat: int
        minimum latitude to consider for analysis
    max_lat: int
        maximum latitude to consider for analysis
    min_lon: int
        minimum longitude to consider for analysis
    max_lon: int
        maximum longitude to consider for analysis
    seq_len: int
        number of days used by neural network for single prediction
    interval: int
        number of days in between days used for prediction (min=1)
    lead_time: int
        number of days to forecast into the future
    split_year: int
        year used for train/test split (year is included in test set)

    Methods
    -------
    download_sst()
        facilitates download of raw SST data from NOAA
    load_sst()
        loads detrended SST data
    load_target()
        loads encoded target
    prepare_sst()
        facilitates preparation of SST data according to download_raw_sst and combine_raw_sst
    prepare_target()
        facilitates preparation of target data according to load_raw_t95
    prepare()
        runs prepare_sst and prepare_target
    """

    def __init__(self, config_json: str=None, download_raw_sst: bool=False, combine_raw_sst: bool=False, load_raw_t95: bool=False):
        """
        Parameters
        ----------
        config_json: str
            file path to a config file that contains "Config Attributes"
        download_raw_sst: bool
            True if user needs to download raw SST data from NOAA
        combine_raw_sst: bool
            True if user has not performed detrending on raw SST data
        load_raw_t95: bool
            True if user has not already encoded target
        """

        self.download_raw_sst = download_raw_sst
        self.combine_raw_sst = combine_raw_sst
        self.load_raw_t95 = load_raw_t95

        with open(config_json) as json_file:
            config = json.load(json_file)

            self.yr_begin = config['yr_begin']
            self.yr_end = config['yr_end']
            self.data_path = config['data_path']
            self.sst_inpath = config['sst_inpath']
            self.sst_outpath = config['sst_outpath']
            self.target_inpath = config['target_inpath']
            self.target_outpath = config['target_outpath']
            self.target_levels = config['target_levels']
            self.min_lat = config['min_lat']
            self.max_lat = config['max_lat']
            self.min_lon = config['min_lon']
            self.max_lon = config['max_lon']
            self.seq_len = config['seq_len']
            self.interval = config['interval']
            self.lead_time = config['lead_time']
            self.split_year = config['split_year']

            self.data = None
            self.data_dates = None
            self.lat_len = None
            self.lon_len = None
            self.channel = None
            self.target = None
            self.target_dates = None
            self._coefs = None

    def download_sst(self):
        """Facilitates dowonload of raw SST data from NOAA.

        Process takes time, so user is prompted to confirm before it runs.

        Attributes
        ----------
        yr_begin: int
            first year of SST data to be included
        yr_end: int
            last year of SST data to be included
        data_path: str
            file path to where data will be stored or accessed
        """

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

            filenames = ["sst.day.mean."+str(i)+".nc" for i in list(range(self.yr_begin,self.yr_end+1))]

            ftp = ftplib.FTP(host = host_name)
            ftp.login(username,password)
            ftp.cwd(path)

            for j in filenames:
                print("loading "+j[-7:-3])
                ftp.retrbinary("RETR "+j,open(self.data_path+j[-7:],"wb").write)


    def _combine_sst(self):
        """Combines time series of data range specified by min_lat, max_lat, min_lon, and max_lon over all years and assigns to data"""

        filenames = [str(i)+'.nc' for i in list(range(self.yr_begin,self.yr_end+1))]

        combined = xr.open_dataarray(self.data_path+filenames[0])
        lat_idx = np.where(np.logical_and(combined['lat'].values >= self.min_lat,combined['lat'].values <= self.max_lat))[0]
        lon_idx = np.where(np.logical_and(combined['lon'].values >= self.min_lon,combined['lon'].values <= self.max_lon))[0]
        combined = combined[:,lat_idx,lon_idx]

        for i in filenames[1:]:
            print('Adding: {}'.format(i))
            curr = xr.open_dataarray(self.data_path+i)[:,lat_idx,lon_idx]
            combined = xr.concat([combined,curr],dim='time')

        combined = combined.fillna(0)
        self.data = combined
        self.lat_len = len(self.data.lat)
        self.lon_len = len(self.data.lon)

    def _load_t95(self):
        """Loads raw t95 data into target"""

        self.target = xr.open_dataarray(self.target_inpath)
        self.target_dates = pd.to_datetime(self.target.time.values)

    def _remove_leap_sst(self):
        """Removes leap dates from SST data"""

        self.data_dates = pd.to_datetime(self.data.time.values)
        data_nonleap = np.where(np.logical_or(self.data_dates.month!=2,self.data_dates.day!=29))[0]
        self.data_dates = self.data_dates[data_nonleap]
        self.data = self.data[data_nonleap]

    def _remove_leap_t95(self):
        """Removes leap dates from target data"""

        self.target_dates = pd.to_datetime(self.target.time.values)
        target_nonleap = np.where(np.logical_or(self.target_dates.month!=2,self.target_dates.day!=29))[0]
        self.target_dates = self.target_dates[target_nonleap]
        self.target = self.target[target_nonleap]

    def _reshape_daily(self):
        """Creates time series for each individual day of year spanning the range set by yr_begin and yr_end. Shape: (doy, year, lat, lon)"""

        days_in_year = 365
        num_years = self.yr_end - self.yr_begin + 1

        day_array = xr.DataArray(np.zeros((days_in_year,num_years,self.lat_len,self.lon_len)),
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

    def _fit_trendline(self,day):
        """Fits trendline to time series of individual day of year and returns coefficients"""

        year_len = len(day.year)
        doy_vec = np.arange(1,year_len+1)
        day = day.stack(ll=('lat','lon'))
        coefs = np.polyfit(doy_vec,day,deg=1)
        coefs = coefs.reshape((2,self.lat_len,self.lon_len))

        return coefs

    def _fit_all_trendlines(self):
        """Fits trendline to all individual days of the year and stores all coefficients in xarray. Shape: (doy, coef, lat, lon)"""

        days_in_year = 365
        num_coefs = 2
        self._coefs = xr.DataArray(np.zeros((days_in_year,num_coefs,self.lat_len,self.lon_len)),
                                dims=['doy','coef','lat','lon'],
                                coords={'doy':np.arange(1,days_in_year+1),
                                        'coef':['slope','intercept'],
                                        'lat':self.data.lat.values,
                                        'lon':self.data.lon.values})

        for day in self.data.doy.values:
            self._coefs.loc[day,:] = self.fit_trendline(self.data.loc[day,:])

    def _remove_seasonality(self,coef):
        """Fits sinusoid using projection onto Fourier Basis and returns fit"""

        nbases = 3
        time_len = len(coef.doy)
        doy_vec = np.arange(1,time_len+1)
        t_basis = (doy_vec - 0.5)/365
        nt = len(t_basis)
        bases = np.empty((nbases, nt), dtype=complex)
        for counter in range(nbases):
            bases[counter, :] = np.exp(2*(counter + 1)*np.pi*1j*t_basis)

        coef -= coef.reduce(np.mean, dim='doy')
        fourier_coef = 2/nt*(np.sum(bases[..., np.newaxis, np.newaxis]*coef.values[np.newaxis, ...], axis=1))
        rec = np.real(np.conj(fourier_coef[:, np.newaxis, ...])*bases[..., np.newaxis, np.newaxis])
        fit = np.sum(rec, axis=0)
        return fit

    def _remove_coef_seasonality(self):
        """Removes the seasonality within the coefficients without removing the mean"""

        slope = self._coefs.loc[:,'slope',:,:].copy(deep=True)
        intercept = self._coefs.loc[:,'intercept',:,:].copy(deep=True)
        self._coefs.loc[:,'slope',:,:] = self.remove_seasonality(slope) + self._coefs.loc[:,'slope',:,:].mean(axis=0).values
        self._coefs.loc[:,'intercept',:,:] = self.remove_seasonality(intercept) + self._coefs.loc[:,'intercept',:,:].mean(axis=0).values

    def _remove_daily_trends(self):
        """Fits trendline to each day of year according to deseasonalized coefficents and subtracts it away from data"""

        year_vec = np.arange(1,len(self.data.year)+1).reshape((1,len(self.data.year),1,1))
        self.data -= self._coefs.loc[:,'slope',:,:].expand_dims(dim='year',axis=1) * year_vec
        self.data = self.data.transpose('year','doy','lat','lon') - self._coefs.loc[:,'intercept',:,:]
        self.data = self.data.transpose('doy','year','lat','lon')

    def _reshape_sequential(self):
        """Reshapes data into serialized time series according to date. Shape: (time, lat, lon)"""

        seq_len = len(self.data.doy) * len(self.data.year)

        data = xr.DataArray(np.zeros((seq_len,self.lat_len,self.lon_len)),
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

    def _add_dimension(self, dim_name: str=None):
        """Adds dimension to data. Used for channel"""

        self.data = self.data.expand_dims(dim=dim_name,axis=3)

    def _save_sst(self):
        """Saves SST data to specified outpath"""

        self.data.to_netcdf(self.sst_outpath)

    def _run_detrending(self):
        """Pipeline for running the entire detrending process"""

        self._reshape_daily()
        self._fit_all_trendlines()
        self._remove_coef_seasonality()
        self._remove_daily_trends()
        self._reshape_sequential()
        self._add_dimension('channel')
        self._save_sst()

    def _encode_target(self):
        """Encodes the target according to target levels. Binary encoding if 1 level is set or one-hot encoding if multiple"""

        if len(self.target_levels) == 1:
            self.target[np.where(self.target < np.percentile(self.target,self.target_levels[0]))] = 0
            self.target[np.where(self.target > 0)] = 1

        else:
            self.target_levels.append(0)
            target_onehot = xr.DataArray(np.zeros((len(self.target),len(self.target_levels))),
                                         dims = ['time','lowerbound'],
                                         coords={'time': self.target_dates,
                                                 'lowerbound': self.target_levels})
            target_onehot[np.where(self.target >= np.percentile(self.target,self.target_levels[0]))[0],0] = 1
            for i in range(1,len(self.target_levels)):
                target_onehot[np.where(np.logical_and(self.target < np.percentile(self.target,self.target_levels[i-1]),self.target >= np.percentile(self.target,self.target_levels[i])))[0],i] = 1

            self.target = target_onehot

    def _save_target(self):
        """Saves target to specified outpath"""

        self.target.to_netcdf(self.target_outpath)

    def load_sst(self):
        """Loads detrended SST data.

        Also assigns the attributes that will be necessary for the generator come model training.

        Attributes
        ----------
        sst_inpath: str
            file path to detrended SST data
        data: xarray.core.dataarray.DataArray
            detrended SST data
        lat_len: int
            number of latitude indices
        lon_len: int
            number of longitude indices
        channel: int
            number of channels (1)
        data_dates: pandas.core.indexes.datetimes.DatetimeIndex
            dates corresponding to each entry of data
        """

        self.data = xr.open_dataarray(self.sst_inpath)
        self.lat_len = len(self.data.lat)
        self.lon_len = len(self.data.lon)
        self.channel = len(self.data.channel)
        self.data_dates = pd.to_datetime(self.data.time.values)

    def load_target(self):
        """Loads encoded target.

        Also assigns attributes that will be necessary for the generator come model training.

        Attributes
        ----------
        target_inpath: str
            file path to encoded target
        target: xarray.core.dataarray.DataArray
            encoded target values
        target_dates: pandas.core.indexes.datetimes.DatetimeIndex
            dates corresponding to each entry of target
        """

        self.target = xr.open_dataarray(self.target_inpath)
        self.target_dates = pd.to_datetime(self.target.time.values)

    def prepare_sst(self):
        """Facilitates preparation of SST data according to download_raw_sst and combine_raw_sst.

        If download_raw_sst is True, data is downloaded, combined, and detrended. If combine_raw_sst is
        True, data is combined and detrended. Otherwise, data is assumed to be detrended and is simply
        loaded.

        Attributes
        ----------
        download_raw_sst: bool
            True if user needs to download raw SST data from NOAA
        combine_raw_sst: bool
            True if user has not performed detrending on raw SST data
        """

        if self.download_raw_sst:
            self.download_sst()
            self._combine_sst()
            self._remove_leap_sst()
            self._run_detrending()
        elif self.combine_raw_sst:
            self._combine_sst()
            self._remove_leap_sst()
            self._run_detrending()
        else:
            self.load_sst()

    def prepare_target(self):
        """Facilitates preparation of target data according to load_raw_t95.

        If load_raw_t95 is True, raw t95 data is loaded and encoded. Otherwise, target
        is assumed to be encoded already and is simply loaded.

        Attributes
        ----------
        load_raw_t95: bool
            True if user has not already encoded target
        """
        if self.load_raw_t95:
            self._load_t95()
            self._remove_leap_t95()
            self._encode_target()
            self._save_target()
        else:
            self.load_target()

    def prepare(self):
        """Runs prepare_sst and prepare_target.

        Run this method with appropriate boolean attributes to ensure data and target
        are properly prepared. See individual method docstrings for more help.
        """
        
        self.prepare_sst()
        self.prepare_target()
