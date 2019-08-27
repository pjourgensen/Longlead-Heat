# Gene Selection for Acute Leukemias

#### -- Project Status: [Active]

## Intro/Objective
The purpose of this project is to develop a forecasting model for heatwave events for the intended use of informing agricultural planning and community preparation. The data that will be used for developing this model is Sea Surface Temperature (SST) data that has been collated by the NOAA. SST data is a valuable factor in climate modeling and weather forecasting and has had demonstrated success in similar models[1]. In this project, I plan to apply neural networks to this data in an effort to predict heatwave events in the Eastern US.

### Methods Used
* Inferential Statistics
* Time Series Analysis
* Detrending
* Fourier Transform
* CNN
* LSTM
* Algorithm Development
* Data Visualization
* Predictive Modeling

### Technologies
* Python, jupyter
* Pandas, Numpy
* Tensorflow, Keras
* Seaborn, Matplotlib

## Project Description
* Data  
   * Data consists of daily sea surface temperature grids (720x1440) for the beginning of 1982 through the end of 2015
   * Target data consists of the spatial 95th percentile of temperature anomalies across the eastern US
   * Neither data set contains missing values
* Detrending and Deseasonalizing Data
   * Removed leap dates and created time series for each individual day of the year (365 time series' of 34 timesteps)
   * Fit linear trend to each of the time series to determine slope and intercept for each day of the year
   * Fit sinusoid to slope and intercept time series' using projection onto Fourier Basis to assess seasonality of coefficients
   * Used projected coefficient values to fit and remove linear trend from each day of the year, capturing long term trend and annual seasonality
* Target Encoding
   * One-hot encoded target data for ternary classification by categorizing data based on percentiles
   * Base model trained on separations at 67th and 33rd percentiles
* Model
   * Built multilayer ConvLSTM model with keras to capture spatiotemporal dependencies
   * Leveraged Google Cloud Engine for training

## Getting Started

1. Clone this repo.
2. Download Raw SST data into "Data" folder by running SSTPreparer.download_sst(). Note: each year requires ~450 Mb.    
3. Download the requirements.
4. Update configs as needed. Refer to docstrings for more details.
5. In "train.py", update parameters as needed. Refer to docstrings for more details.
6. Run "train.py"

## For more detail and discussion:
* [Blog Post](https://pjourgensen.github.io/longlead_pred.html)

