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

class SSTPreparer:

  def __init__(self):
      pass

  def load_data(self):
      pass

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
