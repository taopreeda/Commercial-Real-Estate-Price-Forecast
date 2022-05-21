# Commercial Real Estate Price Prediction

## Model Objective

- To forecast CPI (Commercial Price Index) from Q4/2017 to Q3/2019 in 10 US major cities: Atlanta, Boston, Chicago, Dallas, Houston, Los Angeles, Miami, New York, Philadelphia, Washington DC.

## Dataset Summary
- Historical quarterly price index from Q2/2001 to Q3/2017 in 10 US cities
- 30-Year Fixed Rate Mortgage Average
- Freddit Mac Housing Price Index
- Zillow Median Income
- Zillow Home Value Index
- Zillow Rent Value Index
- ACS population data
- Zip to CBSA mapping
- Market code to description

## Overview
There are 4+1 steps in this model.
### Step 1: Data preparation
- Data processing: Join tables, clean and test data, add macro variables, define new variables if needed.
### Step 1b: Data visualization
- Visualize data to find insights.

<p align="center">Historical Price Index in 10 US cities\
![alt text](https://github.com/taopreeda/Commercial-Real-Estate-Price-Forecast/blob/main/cpi.png)</p>\
### Step 2: Model calibration and testing
- Run a model to get coefficient estimates and performance testing between historical and estimate Commercial Price Index (CPI)
### Step 3: Forecast macro variables
- Predict macro drivers which will be used in step 4 to forecast CPI
### Step 4: Forecast CPI
- Forecast CPI from 4Q/2017 to 3Q/2019


> More details in each step can be found in the relevant folder.