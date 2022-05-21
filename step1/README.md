### Step 1: Data preparation
- Data processing: Join tables, Clean and test data, Add macro variables, Define new variables if needed.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
%matplotlib inline
```

# Load Data


```python
cpi = pd.read_csv('Q2_cpi.csv', index_col='period', parse_dates = True)
frm30yr = pd.read_csv('Q2_30_Year_FRM.csv', index_col='period', parse_dates = True)
fmhpi = pd.read_csv('Q2_fmhpi.csv', index_col='period', parse_dates = True)
income = pd.read_csv('Q2_zillow_mi_market.csv', index_col='period', parse_dates = True)
hi = pd.read_csv('Q2_zillow_hi_market.csv', index_col='period', parse_dates = True)
hi_zip = pd.read_csv('Q2_zillow_hi_zip.csv', index_col='period', parse_dates = True)
ri = pd.read_csv('Q2_zillow_ri_market.csv', index_col='period', parse_dates = True)
ri_zip = pd.read_csv('Q2_zillow_ri_zip.csv', index_col='period', parse_dates = True)
mkt_pop = pd.read_csv('Q2_market_pop.csv', index_col='period', parse_dates = True)
zip_pop = pd.read_csv('Q2_zipcode_pop.csv')
zip_to_mkt = pd.read_csv('Q2_zip_to_market_corr.csv')
mkt_to_name = pd.read_csv('Q2_market_to_name.csv')
```


```python
cpi.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Atlanta-Sandy Springs-Marietta, GA Metro Area</th>
      <th>Boston-Cambridge-Newton, MA-NH Metro Area</th>
      <th>Chicago-Joliet-Naperville, IL-IN-WI Metro Area</th>
      <th>Dallas-Fort Worth-Arlington, TX Metro Area</th>
      <th>Houston-Sugar Land-Baytown, TX Metro Area</th>
      <th>Los Angeles-Long Beach-Anaheim, CA Metro Area</th>
      <th>Miami-Fort Lauderdale-Miami Beach, FL Metropolitan Statistical Area</th>
      <th>New York-Newark-Jersey City, NY-NJ-PA Metro Area</th>
      <th>Philadelphia-Camden-Wilmington, PA-NJ-DE-MD Metro Area</th>
      <th>Washington-Arlington-Alexandria, DC-VA-MD-WV Metro Area</th>
    </tr>
    <tr>
      <th>period</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001-04-01</th>
      <td>67.572311</td>
      <td>76.393851</td>
      <td>66.301575</td>
      <td>70.113257</td>
      <td>69.958842</td>
      <td>60.952183</td>
      <td>60.805031</td>
      <td>63.128420</td>
      <td>70.965044</td>
      <td>62.413111</td>
    </tr>
    <tr>
      <th>2001-07-01</th>
      <td>67.292649</td>
      <td>75.527751</td>
      <td>66.736398</td>
      <td>70.157962</td>
      <td>69.702285</td>
      <td>61.427964</td>
      <td>60.807779</td>
      <td>63.365518</td>
      <td>71.602410</td>
      <td>63.129960</td>
    </tr>
    <tr>
      <th>2001-10-01</th>
      <td>67.071931</td>
      <td>76.010220</td>
      <td>68.067210</td>
      <td>69.743978</td>
      <td>69.587393</td>
      <td>61.192562</td>
      <td>61.471180</td>
      <td>63.758934</td>
      <td>71.997636</td>
      <td>63.962213</td>
    </tr>
    <tr>
      <th>2002-01-01</th>
      <td>67.183085</td>
      <td>75.452801</td>
      <td>68.453589</td>
      <td>68.723861</td>
      <td>69.882963</td>
      <td>61.356233</td>
      <td>62.187711</td>
      <td>63.926603</td>
      <td>71.868553</td>
      <td>64.298790</td>
    </tr>
    <tr>
      <th>2002-04-01</th>
      <td>67.483460</td>
      <td>74.662812</td>
      <td>68.191444</td>
      <td>67.676960</td>
      <td>70.122951</td>
      <td>61.038046</td>
      <td>64.228132</td>
      <td>64.575764</td>
      <td>72.455305</td>
      <td>63.792568</td>
    </tr>
  </tbody>
</table>
</div>




```python
cities_cbsa = ['12060', '14460', '16980', '19100', '26420', '31080', '33100', '35620', '37980', '47900']
```


```python
idx = pd.DatetimeIndex(start=cpi.index.min(),end=cpi.index.max(),freq='1D')
frm30yr = frm30yr.reindex(idx).interpolate(method='linear')
frm30yr = frm30yr.reindex(cpi.index)
```


```python
mkt_nm = dict(zip(mkt_to_name.name,mkt_to_name.cbsa))

cpi = cpi.rename(columns=mkt_nm)

cpi.columns = cpi.columns.map(str)
```


```python
# rent to home value
rv = pd.DataFrame(ri.values/hi['2010-11':].values, columns=ri.columns, index=ri.index)
```

## Check stationary


```python
from statsmodels.tsa.stattools import adfuller

def stationary(df):
    result = adfuller(df.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
```


```python
ds = [cpi['12060'], fmhpi['12060'], income['12060'], hi['12060'], ri['12060'], mkt_pop['12060'], rv['12060'], frm30yr['mortgage30us']]
for i in ds:
    print(stationary(i))
```

    ADF Statistic: -2.768178
    p-value: 0.062965
    None
    ADF Statistic: -1.545556
    p-value: 0.510867
    None
    ADF Statistic: 1.224590
    p-value: 0.996151
    None
    ADF Statistic: -3.284241
    p-value: 0.015598
    None
    ADF Statistic: 1.045081
    p-value: 0.994720
    None
    ADF Statistic: 0.000000
    p-value: 0.958532
    None
    ADF Statistic: -1.530782
    p-value: 0.518233
    None
    ADF Statistic: -1.747013
    p-value: 0.407067
    None


    /Users/preeda/anaconda/lib/python3.6/site-packages/statsmodels/regression/linear_model.py:1353: RuntimeWarning: divide by zero encountered in double_scalars
      return np.dot(wresid, wresid) / self.df_resid


Essentially, most of them are non-stationary. Need to do some transformation.

## Data Transformation


```python
# interpolate income and population data
income = income.resample('MS').interpolate('linear')
mkt_pop = mkt_pop.resample('MS').interpolate('linear')
```


```python
# Make it quarterly and interpolate. Extrapolate can be improved
income = income.reindex(cpi.index)
income = income.interpolate(method='linear', fill_value='extrapolate')
mkt_pop = mkt_pop.reindex(cpi.index)
mkt_pop = mkt_pop.interpolate(method='linear', fill_value='extrapolate')
```


```python
cpr = cpi.pct_change()
fmhpr = fmhpi.pct_change()
income_gr = income.pct_change()
hi_gr = hi.pct_change()
ri_gr = ri.pct_change()
pop_gr = mkt_pop.pct_change()
rv_chg = rv.diff()
```


```python
frm30yr_chg = frm30yr.diff()
frm30yr_chg.columns = ['mortgage30us_chg']
```

Check stationary again.


```python
ds = [cpr['12060'], fmhpr['12060'], income_gr['12060'], hi_gr['12060'], ri_gr['12060'], pop_gr['12060'], rv_chg['12060']]
for i in ds:
    print(stationary(i))
```

    ADF Statistic: -2.668548
    p-value: 0.079641
    None
    ADF Statistic: -1.988320
    p-value: 0.291718
    None
    ADF Statistic: -2.778754
    p-value: 0.061373
    None
    ADF Statistic: -1.723173
    p-value: 0.419214
    None
    ADF Statistic: -4.655148
    p-value: 0.000102
    None
    ADF Statistic: -2.217787
    p-value: 0.199853
    None
    ADF Statistic: -2.287657
    p-value: 0.175980
    None


Better but there're still some issues. Due to time constraints, I'll continue to use this data to do the modeling.

### Save raw data used for step 3 Forecast macro variables


```python
frm30yr.to_csv('frm30yr.csv')
mkt_pop.to_csv('mkt_pop.csv')
```

# Normalize data


```python
def normalize(df,meannm,distnm):
    mean = df.mean().to_frame(name=meannm)
    dist = (df.max() - df.min()).to_frame(name=distnm)
    return (df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))), mean, dist)
```


```python
cpr, mean_cpr, dist_cpr = normalize(cpr,'mean_cpr','dist_cpr')
fmhpr, mean_fmhpr, dist_fmhpr = normalize(fmhpr,'mean_fmhpr','dist_fmhpr')
income_gr, mean_income, dist_income = normalize(income_gr,'mean_income','dist_income')
hi_gr, mean_hi, dist_hi = normalize(hi_gr,'mean_hi','dist_hi')
ri_gr, mean_ri, dist_ri = normalize(ri_gr,'mean_ri','dist_ri')
pop_gr, mean_pop, dist_pop = normalize(pop_gr,'mean_pop','dist_pop')
rv, mean_rv, dist_rv = normalize(rv,'mean_rv','dist_rv')
frm30yr, mean_frm30yr, dist_frm30yr = normalize(frm30yr,'mean_frm30yr','dist_frm30yr')
rv_chg, mean_rv_chg, dist_rv_chg = normalize(rv_chg,'mean_rv_chg','dist_rv_chg')
frm30yr_chg, mean_frm30yr_chg, dist_frm30yr_chg = normalize(frm30yr_chg,'mean_frm30yr_chg','dist_frm30yr_chg')
```


```python
mean_all = pd.concat([mean_cpr, mean_fmhpr, mean_income, mean_hi, mean_ri, mean_pop, mean_rv, mean_rv_chg], axis=1)
mean_all['mean_frm30yr'] = mean_frm30yr.iloc[0][0]
mean_all['mean_frm30yr_chg'] = mean_frm30yr_chg.iloc[0][0]
dist_all = pd.concat([dist_cpr, dist_fmhpr, dist_income, dist_hi, dist_ri, dist_pop, dist_rv, dist_rv_chg], axis=1)
dist_all['dist_frm30yr'] = dist_frm30yr.iloc[0][0]
dist_all['dist_frm30yr_chg'] = dist_frm30yr_chg.iloc[0][0]
```


```python
mean_all.index.name = 'cbsa'
dist_all.index.name = 'cbsa'
```


```python
ds = [cpr['12060'], fmhpr['12060'], income_gr['12060'], hi_gr['12060'], ri_gr['12060'], pop_gr['12060'], rv_chg['12060']]
for i in ds:
    print(stationary(i))
```

    ADF Statistic: -2.668548
    p-value: 0.079641
    None
    ADF Statistic: -1.988320
    p-value: 0.291718
    None
    ADF Statistic: -2.778754
    p-value: 0.061373
    None
    ADF Statistic: -1.723173
    p-value: 0.419214
    None
    ADF Statistic: -4.655148
    p-value: 0.000102
    None
    ADF Statistic: -2.217787
    p-value: 0.199853
    None
    ADF Statistic: -2.287657
    p-value: 0.175980
    None


### Add lag


```python
# Create lag1 and lag2 quarter
def lag_func(df):
    df_l1 = df.shift(1)
    df_l2 = df.shift(2)
    return (df_l1, df_l2)
```


```python
cpr_l1, cpr_l2 = lag_func(cpr)
frm30yr_l1, frm30yr_l2 = lag_func(frm30yr)
fmhpr_l1, fmhpr_l2 = lag_func(fmhpr)
income_gr_l1, income_gr_l2 = lag_func(income_gr)
hi_gr_l1, hi_gr_l2 = lag_func(hi_gr)
ri_gr_l1, ri_gr_l2 = lag_func(ri_gr)
pop_gr_l1, pop_gr_l2 = lag_func(pop_gr)
rv_l1, rv_l2 = lag_func(rv)
frm30yr_chg_l1, frm30yr_chg_l2 = lag_func(frm30yr_chg)
rv_chg_l1, rv_chg_l2 = lag_func(rv_chg)
```


```python
frm30yr_l1.columns = ['mortgage30us_l1']
frm30yr_l2.columns = ['mortgage30us_l2']
frm30yr_chg_l1.columns = ['mortgage30us_chg_l1']
frm30yr_chg_l2.columns = ['mortgage30us_chg_l2']
```

# Convert Data Wide to Long


```python
def w_to_l(df,id1,col1,col2):
    df.reset_index(inplace=True)
    df = pd.melt(df, id_vars=[id1],value_vars=df.columns[1:])
    df.columns = [id1,col1,col2]
    df[col1] = pd.to_numeric(df[col1])
    return df
```


```python
cpi = w_to_l(cpi,'period','cbsa','cpi')
cpr = w_to_l(cpr,'period','cbsa','ret')
cpr_l1 = w_to_l(cpr_l1,'period','cbsa','return_l1')
cpr_l2 = w_to_l(cpr_l2,'period','cbsa','return_l2')
fmhpr = w_to_l(fmhpr,'period','cbsa','fmhpr')
fmhpr_l1 = w_to_l(fmhpr_l1,'period','cbsa','fmhpr_l1')
fmhpr_l2 = w_to_l(fmhpr_l2,'period','cbsa','fmhpr_l2')
income_gr = w_to_l(income_gr,'period','cbsa','income_gr')
income_gr_l1 = w_to_l(income_gr_l1,'period','cbsa','income_gr_l1')
income_gr_l2 = w_to_l(income_gr_l2,'period','cbsa','income_gr_l2')
hi_gr = w_to_l(hi_gr,'period','cbsa','hi_gr')
hi_gr_l1 = w_to_l(hi_gr_l1,'period','cbsa','hi_gr_l1')
hi_gr_l2 = w_to_l(hi_gr_l2,'period','cbsa','hi_gr_l2')
ri_gr = w_to_l(ri_gr,'period','cbsa','ri_gr')
ri_gr_l1 = w_to_l(ri_gr_l1,'period','cbsa','ri_gr_l1')
ri_gr_l2 = w_to_l(ri_gr_l2,'period','cbsa','ri_gr_l2')
pop_gr = w_to_l(pop_gr,'period','cbsa','pop_gr')
pop_gr_l1 = w_to_l(pop_gr_l1,'period','cbsa','pop_gr_l1')
pop_gr_l2 = w_to_l(pop_gr_l2,'period','cbsa','pop_gr_l2')
rv = w_to_l(rv,'period','cbsa','rv')
rv_l1 = w_to_l(rv_l1,'period','cbsa','rv_l1')
rv_l2 = w_to_l(rv_l2,'period','cbsa','rv_l2')
```


```python
frm30yr.reset_index(inplace=True)
frm30yr_l1.reset_index(inplace=True)
frm30yr_l2.reset_index(inplace=True)
frm30yr_chg.reset_index(inplace=True)
frm30yr_chg_l1.reset_index(inplace=True)
frm30yr_chg_l2.reset_index(inplace=True)
```


```python
cpr.dropna(inplace=True)
```


```python
cpr = cpr.merge(frm30yr, how='left', left_on=['period'], right_on=['period'])
cpr = cpr.merge(frm30yr_l1, how='left', left_on=['period'], right_on=['period'])
cpr = cpr.merge(frm30yr_l2, how='left', left_on=['period'], right_on=['period'])
cpr = cpr.merge(frm30yr_chg, how='left', left_on=['period'], right_on=['period'])
cpr = cpr.merge(frm30yr_chg_l1, how='left', left_on=['period'], right_on=['period'])
cpr = cpr.merge(frm30yr_chg_l2, how='left', left_on=['period'], right_on=['period'])
cpr = cpr.merge(cpr_l1, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(cpr_l2, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(fmhpr, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(fmhpr_l1, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(fmhpr_l2, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(income_gr, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(income_gr_l1, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(income_gr_l2, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(hi_gr, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(hi_gr_l1, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(hi_gr_l2, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(ri_gr, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(ri_gr_l1, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(ri_gr_l2, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(pop_gr, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(pop_gr_l1, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(pop_gr_l2, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(rv, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(rv_l1, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
cpr = cpr.merge(rv_l2, how='left', left_on=['period','cbsa'], right_on=['period','cbsa'])
```


```python
cpi.set_index(['period'], inplace=True)
cpr.set_index(['period'], inplace=True)
```

### Check Data


```python
cpr.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cbsa</th>
      <th>ret</th>
      <th>mortgage30us</th>
      <th>mortgage30us_l1</th>
      <th>mortgage30us_l2</th>
      <th>mortgage30us_chg</th>
      <th>mortgage30us_chg_l1</th>
      <th>mortgage30us_chg_l2</th>
      <th>return_l1</th>
      <th>return_l2</th>
      <th>...</th>
      <th>hi_gr_l2</th>
      <th>ri_gr</th>
      <th>ri_gr_l1</th>
      <th>ri_gr_l2</th>
      <th>pop_gr</th>
      <th>pop_gr_l1</th>
      <th>pop_gr_l2</th>
      <th>rv</th>
      <th>rv_l1</th>
      <th>rv_l2</th>
    </tr>
    <tr>
      <th>period</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001-07-01</th>
      <td>12060</td>
      <td>-0.048960</td>
      <td>0.526050</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.123804</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2001-10-01</th>
      <td>12060</td>
      <td>-0.043616</td>
      <td>0.408866</td>
      <td>0.526050</td>
      <td>NaN</td>
      <td>-0.213258</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.048960</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.074680</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2002-01-01</th>
      <td>12060</td>
      <td>-0.012888</td>
      <td>0.530169</td>
      <td>0.408866</td>
      <td>0.526050</td>
      <td>0.276413</td>
      <td>-0.213258</td>
      <td>NaN</td>
      <td>-0.043616</td>
      <td>-0.048960</td>
      <td>...</td>
      <td>0.096688</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2002-04-01</th>
      <td>12060</td>
      <td>0.004624</td>
      <td>0.532789</td>
      <td>0.530169</td>
      <td>0.408866</td>
      <td>0.032730</td>
      <td>0.276413</td>
      <td>-0.213258</td>
      <td>-0.012888</td>
      <td>-0.043616</td>
      <td>...</td>
      <td>0.117814</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2002-07-01</th>
      <td>12060</td>
      <td>0.039185</td>
      <td>0.375545</td>
      <td>0.532789</td>
      <td>0.530169</td>
      <td>-0.295510</td>
      <td>0.032730</td>
      <td>0.276413</td>
      <td>0.004624</td>
      <td>-0.012888</td>
      <td>...</td>
      <td>0.160655</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
cpr.shape
```




    (650, 28)




```python
cols = ['income_gr','income_gr_l1', 'income_gr_l2','ri_gr', 'ri_gr_l1', 'ri_gr_l2','rv', 'rv_l1', 'rv_l2']
cpr.drop(cols, axis=1, inplace=True)
```


```python
cpr.isnull().sum()
```




    cbsa                     0
    ret                      0
    mortgage30us             0
    mortgage30us_l1         10
    mortgage30us_l2         20
    mortgage30us_chg        10
    mortgage30us_chg_l1     20
    mortgage30us_chg_l2     30
    return_l1               10
    return_l2               20
    fmhpr                    0
    fmhpr_l1                 0
    fmhpr_l2                 0
    hi_gr                    0
    hi_gr_l1                 0
    hi_gr_l2                 0
    pop_gr                 150
    pop_gr_l1              160
    pop_gr_l2              170
    dtype: int64




```python
cpr.dropna(inplace=True)
```


```python
cpr.shape
```




    (480, 19)




```python
cpr.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cbsa</th>
      <th>ret</th>
      <th>mortgage30us</th>
      <th>mortgage30us_l1</th>
      <th>mortgage30us_l2</th>
      <th>mortgage30us_chg</th>
      <th>mortgage30us_chg_l1</th>
      <th>mortgage30us_chg_l2</th>
      <th>return_l1</th>
      <th>return_l2</th>
      <th>fmhpr</th>
      <th>fmhpr_l1</th>
      <th>fmhpr_l2</th>
      <th>hi_gr</th>
      <th>hi_gr_l1</th>
      <th>hi_gr_l2</th>
      <th>pop_gr</th>
      <th>pop_gr_l1</th>
      <th>pop_gr_l2</th>
    </tr>
    <tr>
      <th>period</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2005-10-01</th>
      <td>12060</td>
      <td>0.137064</td>
      <td>0.210812</td>
      <td>0.109352</td>
      <td>0.235522</td>
      <td>0.235671</td>
      <td>-0.231707</td>
      <td>0.150728</td>
      <td>0.029383</td>
      <td>0.051247</td>
      <td>0.040290</td>
      <td>0.023973</td>
      <td>0.024726</td>
      <td>0.118606</td>
      <td>0.098996</td>
      <td>0.099571</td>
      <td>0.485892</td>
      <td>0.495601</td>
      <td>0.505618</td>
    </tr>
    <tr>
      <th>2006-01-01</th>
      <td>12060</td>
      <td>0.179827</td>
      <td>0.285691</td>
      <td>0.210812</td>
      <td>0.109352</td>
      <td>0.181092</td>
      <td>0.235671</td>
      <td>-0.231707</td>
      <td>0.137064</td>
      <td>0.029383</td>
      <td>-0.074971</td>
      <td>-0.048408</td>
      <td>0.047618</td>
      <td>0.096583</td>
      <td>0.097139</td>
      <td>0.117867</td>
      <td>0.476477</td>
      <td>0.485892</td>
      <td>0.495601</td>
    </tr>
    <tr>
      <th>2006-04-01</th>
      <td>12060</td>
      <td>0.160063</td>
      <td>0.326874</td>
      <td>0.285691</td>
      <td>0.210812</td>
      <td>0.111908</td>
      <td>0.181092</td>
      <td>0.235671</td>
      <td>0.179827</td>
      <td>0.137064</td>
      <td>0.130991</td>
      <td>0.102689</td>
      <td>0.033318</td>
      <td>0.075392</td>
      <td>0.075791</td>
      <td>0.076194</td>
      <td>0.133119</td>
      <td>0.476477</td>
      <td>0.485892</td>
    </tr>
    <tr>
      <th>2006-07-01</th>
      <td>12060</td>
      <td>0.294379</td>
      <td>0.434324</td>
      <td>0.326874</td>
      <td>0.285691</td>
      <td>0.247970</td>
      <td>0.111908</td>
      <td>0.181092</td>
      <td>0.160063</td>
      <td>0.179827</td>
      <td>-0.036117</td>
      <td>0.070970</td>
      <td>0.134314</td>
      <td>0.093641</td>
      <td>0.074536</td>
      <td>0.094633</td>
      <td>0.131336</td>
      <td>0.133119</td>
      <td>0.476477</td>
    </tr>
    <tr>
      <th>2006-10-01</th>
      <td>12060</td>
      <td>0.178872</td>
      <td>0.309277</td>
      <td>0.434324</td>
      <td>0.326874</td>
      <td>-0.229401</td>
      <td>0.247970</td>
      <td>0.111908</td>
      <td>0.294379</td>
      <td>0.160063</td>
      <td>-0.038141</td>
      <td>-0.025233</td>
      <td>-0.073418</td>
      <td>0.034253</td>
      <td>0.092589</td>
      <td>0.093113</td>
      <td>0.129577</td>
      <td>0.131336</td>
      <td>0.133119</td>
    </tr>
  </tbody>
</table>
</div>




```python
cpr.corr()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cbsa</th>
      <th>ret</th>
      <th>mortgage30us</th>
      <th>mortgage30us_l1</th>
      <th>mortgage30us_l2</th>
      <th>mortgage30us_chg</th>
      <th>mortgage30us_chg_l1</th>
      <th>mortgage30us_chg_l2</th>
      <th>return_l1</th>
      <th>return_l2</th>
      <th>fmhpr</th>
      <th>fmhpr_l1</th>
      <th>fmhpr_l2</th>
      <th>hi_gr</th>
      <th>hi_gr_l1</th>
      <th>hi_gr_l2</th>
      <th>pop_gr</th>
      <th>pop_gr_l1</th>
      <th>pop_gr_l2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cbsa</th>
      <td>1.000000e+00</td>
      <td>-0.040757</td>
      <td>-3.104879e-17</td>
      <td>-3.525410e-17</td>
      <td>5.034293e-18</td>
      <td>2.624937e-18</td>
      <td>1.037545e-17</td>
      <td>0.000000</td>
      <td>-0.027319</td>
      <td>-0.013477</td>
      <td>-0.111540</td>
      <td>-0.104456</td>
      <td>-0.111582</td>
      <td>-0.120019</td>
      <td>-0.112296</td>
      <td>-0.106991</td>
      <td>-0.002172</td>
      <td>0.003576</td>
      <td>0.008609</td>
    </tr>
    <tr>
      <th>ret</th>
      <td>-4.075655e-02</td>
      <td>1.000000</td>
      <td>-1.202622e-01</td>
      <td>-1.956894e-01</td>
      <td>-2.771698e-01</td>
      <td>2.124903e-01</td>
      <td>2.350100e-01</td>
      <td>0.190960</td>
      <td>0.827045</td>
      <td>0.691240</td>
      <td>0.320601</td>
      <td>0.334657</td>
      <td>0.379748</td>
      <td>0.540109</td>
      <td>0.565761</td>
      <td>0.572889</td>
      <td>0.110954</td>
      <td>0.106885</td>
      <td>0.146577</td>
    </tr>
    <tr>
      <th>mortgage30us</th>
      <td>-3.104879e-17</td>
      <td>-0.120262</td>
      <td>1.000000e+00</td>
      <td>9.376492e-01</td>
      <td>8.784959e-01</td>
      <td>1.833078e-01</td>
      <td>1.422138e-01</td>
      <td>0.049472</td>
      <td>-0.044324</td>
      <td>0.015334</td>
      <td>-0.358215</td>
      <td>-0.311517</td>
      <td>-0.286601</td>
      <td>-0.425100</td>
      <td>-0.403690</td>
      <td>-0.364537</td>
      <td>0.195034</td>
      <td>0.228436</td>
      <td>0.302894</td>
    </tr>
    <tr>
      <th>mortgage30us_l1</th>
      <td>-3.525410e-17</td>
      <td>-0.195689</td>
      <td>9.376492e-01</td>
      <td>1.000000e+00</td>
      <td>9.366267e-01</td>
      <td>-1.698151e-01</td>
      <td>1.524791e-01</td>
      <td>0.147723</td>
      <td>-0.095671</td>
      <td>-0.025485</td>
      <td>-0.448960</td>
      <td>-0.418239</td>
      <td>-0.390821</td>
      <td>-0.496522</td>
      <td>-0.472377</td>
      <td>-0.435346</td>
      <td>0.177129</td>
      <td>0.207280</td>
      <td>0.237797</td>
    </tr>
    <tr>
      <th>mortgage30us_l2</th>
      <td>5.034293e-18</td>
      <td>-0.277170</td>
      <td>8.784959e-01</td>
      <td>9.366267e-01</td>
      <td>1.000000e+00</td>
      <td>-1.582915e-01</td>
      <td>-2.034166e-01</td>
      <td>0.163726</td>
      <td>-0.160606</td>
      <td>-0.071316</td>
      <td>-0.497005</td>
      <td>-0.474366</td>
      <td>-0.451637</td>
      <td>-0.565385</td>
      <td>-0.540085</td>
      <td>-0.500228</td>
      <td>0.171130</td>
      <td>0.215037</td>
      <td>0.240460</td>
    </tr>
    <tr>
      <th>mortgage30us_chg</th>
      <td>2.624937e-18</td>
      <td>0.212490</td>
      <td>1.833078e-01</td>
      <td>-1.698151e-01</td>
      <td>-1.582915e-01</td>
      <td>1.000000e+00</td>
      <td>-2.804247e-02</td>
      <td>-0.277533</td>
      <td>0.144914</td>
      <td>0.115556</td>
      <td>0.254157</td>
      <td>0.299667</td>
      <td>0.292766</td>
      <td>0.199038</td>
      <td>0.191455</td>
      <td>0.197727</td>
      <td>0.051999</td>
      <td>0.061424</td>
      <td>0.186221</td>
    </tr>
    <tr>
      <th>mortgage30us_chg_l1</th>
      <td>1.037545e-17</td>
      <td>0.235010</td>
      <td>1.422138e-01</td>
      <td>1.524791e-01</td>
      <td>-2.034166e-01</td>
      <td>-2.804247e-02</td>
      <td>1.000000e+00</td>
      <td>-0.049031</td>
      <td>0.185704</td>
      <td>0.129961</td>
      <td>0.147348</td>
      <td>0.169338</td>
      <td>0.181846</td>
      <td>0.207330</td>
      <td>0.203432</td>
      <td>0.194488</td>
      <td>0.012262</td>
      <td>-0.027335</td>
      <td>-0.013767</td>
    </tr>
    <tr>
      <th>mortgage30us_chg_l2</th>
      <td>0.000000e+00</td>
      <td>0.190960</td>
      <td>4.947219e-02</td>
      <td>1.477226e-01</td>
      <td>1.637256e-01</td>
      <td>-2.775333e-01</td>
      <td>-4.903122e-02</td>
      <td>1.000000</td>
      <td>0.249987</td>
      <td>0.197797</td>
      <td>-0.164011</td>
      <td>-0.127291</td>
      <td>0.018175</td>
      <td>0.131199</td>
      <td>0.180251</td>
      <td>0.209579</td>
      <td>-0.060759</td>
      <td>0.045310</td>
      <td>0.005376</td>
    </tr>
    <tr>
      <th>return_l1</th>
      <td>-2.731944e-02</td>
      <td>0.827045</td>
      <td>-4.432381e-02</td>
      <td>-9.567066e-02</td>
      <td>-1.606060e-01</td>
      <td>1.449136e-01</td>
      <td>1.857041e-01</td>
      <td>0.249987</td>
      <td>1.000000</td>
      <td>0.828103</td>
      <td>0.277960</td>
      <td>0.274657</td>
      <td>0.306687</td>
      <td>0.474337</td>
      <td>0.507030</td>
      <td>0.523838</td>
      <td>0.122562</td>
      <td>0.163060</td>
      <td>0.156806</td>
    </tr>
    <tr>
      <th>return_l2</th>
      <td>-1.347720e-02</td>
      <td>0.691240</td>
      <td>1.533448e-02</td>
      <td>-2.548544e-02</td>
      <td>-7.131559e-02</td>
      <td>1.155561e-01</td>
      <td>1.299609e-01</td>
      <td>0.197797</td>
      <td>0.828103</td>
      <td>1.000000</td>
      <td>0.222322</td>
      <td>0.243635</td>
      <td>0.276161</td>
      <td>0.406804</td>
      <td>0.435975</td>
      <td>0.452572</td>
      <td>0.098438</td>
      <td>0.158645</td>
      <td>0.195765</td>
    </tr>
    <tr>
      <th>fmhpr</th>
      <td>-1.115395e-01</td>
      <td>0.320601</td>
      <td>-3.582145e-01</td>
      <td>-4.489599e-01</td>
      <td>-4.970052e-01</td>
      <td>2.541569e-01</td>
      <td>1.473485e-01</td>
      <td>-0.164011</td>
      <td>0.277960</td>
      <td>0.222322</td>
      <td>1.000000</td>
      <td>0.906297</td>
      <td>0.678993</td>
      <td>0.605890</td>
      <td>0.555268</td>
      <td>0.516907</td>
      <td>-0.094701</td>
      <td>-0.005676</td>
      <td>0.049825</td>
    </tr>
    <tr>
      <th>fmhpr_l1</th>
      <td>-1.044557e-01</td>
      <td>0.334657</td>
      <td>-3.115174e-01</td>
      <td>-4.182391e-01</td>
      <td>-4.743657e-01</td>
      <td>2.996674e-01</td>
      <td>1.693382e-01</td>
      <td>-0.127291</td>
      <td>0.274657</td>
      <td>0.243635</td>
      <td>0.906297</td>
      <td>1.000000</td>
      <td>0.895566</td>
      <td>0.599998</td>
      <td>0.562539</td>
      <td>0.533338</td>
      <td>-0.078850</td>
      <td>-0.013706</td>
      <td>0.072419</td>
    </tr>
    <tr>
      <th>fmhpr_l2</th>
      <td>-1.115819e-01</td>
      <td>0.379748</td>
      <td>-2.866013e-01</td>
      <td>-3.908214e-01</td>
      <td>-4.516372e-01</td>
      <td>2.927659e-01</td>
      <td>1.818458e-01</td>
      <td>0.018175</td>
      <td>0.306687</td>
      <td>0.276161</td>
      <td>0.678993</td>
      <td>0.895566</td>
      <td>1.000000</td>
      <td>0.630724</td>
      <td>0.621352</td>
      <td>0.610916</td>
      <td>-0.047078</td>
      <td>-0.024124</td>
      <td>0.069889</td>
    </tr>
    <tr>
      <th>hi_gr</th>
      <td>-1.200189e-01</td>
      <td>0.540109</td>
      <td>-4.251003e-01</td>
      <td>-4.965215e-01</td>
      <td>-5.653854e-01</td>
      <td>1.990377e-01</td>
      <td>2.073302e-01</td>
      <td>0.131199</td>
      <td>0.474337</td>
      <td>0.406804</td>
      <td>0.605890</td>
      <td>0.599998</td>
      <td>0.630724</td>
      <td>1.000000</td>
      <td>0.967553</td>
      <td>0.904507</td>
      <td>-0.052751</td>
      <td>0.009331</td>
      <td>0.045151</td>
    </tr>
    <tr>
      <th>hi_gr_l1</th>
      <td>-1.122956e-01</td>
      <td>0.565761</td>
      <td>-4.036898e-01</td>
      <td>-4.723772e-01</td>
      <td>-5.400845e-01</td>
      <td>1.914548e-01</td>
      <td>2.034324e-01</td>
      <td>0.180251</td>
      <td>0.507030</td>
      <td>0.435975</td>
      <td>0.555268</td>
      <td>0.562539</td>
      <td>0.621352</td>
      <td>0.967553</td>
      <td>1.000000</td>
      <td>0.966685</td>
      <td>-0.032396</td>
      <td>0.019692</td>
      <td>0.037816</td>
    </tr>
    <tr>
      <th>hi_gr_l2</th>
      <td>-1.069913e-01</td>
      <td>0.572889</td>
      <td>-3.645372e-01</td>
      <td>-4.353460e-01</td>
      <td>-5.002282e-01</td>
      <td>1.977273e-01</td>
      <td>1.944883e-01</td>
      <td>0.209579</td>
      <td>0.523838</td>
      <td>0.452572</td>
      <td>0.516907</td>
      <td>0.533338</td>
      <td>0.610916</td>
      <td>0.904507</td>
      <td>0.966685</td>
      <td>1.000000</td>
      <td>-0.020843</td>
      <td>0.009816</td>
      <td>0.041159</td>
    </tr>
    <tr>
      <th>pop_gr</th>
      <td>-2.172362e-03</td>
      <td>0.110954</td>
      <td>1.950338e-01</td>
      <td>1.771286e-01</td>
      <td>1.711304e-01</td>
      <td>5.199854e-02</td>
      <td>1.226248e-02</td>
      <td>-0.060759</td>
      <td>0.122562</td>
      <td>0.098438</td>
      <td>-0.094701</td>
      <td>-0.078850</td>
      <td>-0.047078</td>
      <td>-0.052751</td>
      <td>-0.032396</td>
      <td>-0.020843</td>
      <td>1.000000</td>
      <td>0.788689</td>
      <td>0.606990</td>
    </tr>
    <tr>
      <th>pop_gr_l1</th>
      <td>3.576258e-03</td>
      <td>0.106885</td>
      <td>2.284356e-01</td>
      <td>2.072801e-01</td>
      <td>2.150370e-01</td>
      <td>6.142385e-02</td>
      <td>-2.733467e-02</td>
      <td>0.045310</td>
      <td>0.163060</td>
      <td>0.158645</td>
      <td>-0.005676</td>
      <td>-0.013706</td>
      <td>-0.024124</td>
      <td>0.009331</td>
      <td>0.019692</td>
      <td>0.009816</td>
      <td>0.788689</td>
      <td>1.000000</td>
      <td>0.812887</td>
    </tr>
    <tr>
      <th>pop_gr_l2</th>
      <td>8.608928e-03</td>
      <td>0.146577</td>
      <td>3.028942e-01</td>
      <td>2.377968e-01</td>
      <td>2.404599e-01</td>
      <td>1.862214e-01</td>
      <td>-1.376739e-02</td>
      <td>0.005376</td>
      <td>0.156806</td>
      <td>0.195765</td>
      <td>0.049825</td>
      <td>0.072419</td>
      <td>0.069889</td>
      <td>0.045151</td>
      <td>0.037816</td>
      <td>0.041159</td>
      <td>0.606990</td>
      <td>0.812887</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Final dataset


```python
cpr.to_csv('cpr_final.csv')
cpi.to_csv('cpi_final.csv')
mean_all.to_csv('mean_all.csv')
dist_all.to_csv('dist_all.csv')
```


```python
cpr.describe().transpose()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cbsa</th>
      <td>480.0</td>
      <td>27470.000000</td>
      <td>11088.217298</td>
      <td>12060.000000</td>
      <td>16980.000000</td>
      <td>28750.000000</td>
      <td>35620.000000</td>
      <td>47900.000000</td>
    </tr>
    <tr>
      <th>ret</th>
      <td>480.0</td>
      <td>-0.018687</td>
      <td>0.202388</td>
      <td>-0.703909</td>
      <td>-0.101883</td>
      <td>0.004099</td>
      <td>0.111199</td>
      <td>0.475699</td>
    </tr>
    <tr>
      <th>mortgage30us</th>
      <td>480.0</td>
      <td>-0.093392</td>
      <td>0.270243</td>
      <td>-0.467211</td>
      <td>-0.302198</td>
      <td>-0.173126</td>
      <td>0.199300</td>
      <td>0.434324</td>
    </tr>
    <tr>
      <th>mortgage30us_l1</th>
      <td>480.0</td>
      <td>-0.084313</td>
      <td>0.269579</td>
      <td>-0.467211</td>
      <td>-0.297892</td>
      <td>-0.157963</td>
      <td>0.199300</td>
      <td>0.434324</td>
    </tr>
    <tr>
      <th>mortgage30us_l2</th>
      <td>480.0</td>
      <td>-0.073963</td>
      <td>0.272116</td>
      <td>-0.467211</td>
      <td>-0.297892</td>
      <td>-0.150475</td>
      <td>0.216990</td>
      <td>0.434324</td>
    </tr>
    <tr>
      <th>mortgage30us_chg</th>
      <td>480.0</td>
      <td>0.008708</td>
      <td>0.195707</td>
      <td>-0.516035</td>
      <td>-0.123511</td>
      <td>-0.030176</td>
      <td>0.133432</td>
      <td>0.483965</td>
    </tr>
    <tr>
      <th>mortgage30us_chg_l1</th>
      <td>480.0</td>
      <td>0.006098</td>
      <td>0.198051</td>
      <td>-0.516035</td>
      <td>-0.125625</td>
      <td>-0.030176</td>
      <td>0.133432</td>
      <td>0.483965</td>
    </tr>
    <tr>
      <th>mortgage30us_chg_l2</th>
      <td>480.0</td>
      <td>0.010238</td>
      <td>0.198954</td>
      <td>-0.516035</td>
      <td>-0.125625</td>
      <td>-0.014417</td>
      <td>0.142945</td>
      <td>0.483965</td>
    </tr>
    <tr>
      <th>return_l1</th>
      <td>480.0</td>
      <td>-0.015818</td>
      <td>0.204392</td>
      <td>-0.703909</td>
      <td>-0.101883</td>
      <td>0.008345</td>
      <td>0.113015</td>
      <td>0.475699</td>
    </tr>
    <tr>
      <th>return_l2</th>
      <td>480.0</td>
      <td>-0.014573</td>
      <td>0.204748</td>
      <td>-0.703909</td>
      <td>-0.101883</td>
      <td>0.011754</td>
      <td>0.116365</td>
      <td>0.475699</td>
    </tr>
    <tr>
      <th>fmhpr</th>
      <td>480.0</td>
      <td>-0.048571</td>
      <td>0.196049</td>
      <td>-0.577166</td>
      <td>-0.173539</td>
      <td>-0.047906</td>
      <td>0.071510</td>
      <td>0.496618</td>
    </tr>
    <tr>
      <th>fmhpr_l1</th>
      <td>480.0</td>
      <td>-0.046248</td>
      <td>0.202293</td>
      <td>-0.585001</td>
      <td>-0.183946</td>
      <td>-0.026688</td>
      <td>0.098662</td>
      <td>0.391584</td>
    </tr>
    <tr>
      <th>fmhpr_l2</th>
      <td>480.0</td>
      <td>-0.044193</td>
      <td>0.195304</td>
      <td>-0.628931</td>
      <td>-0.168111</td>
      <td>-0.042762</td>
      <td>0.076656</td>
      <td>0.509450</td>
    </tr>
    <tr>
      <th>hi_gr</th>
      <td>480.0</td>
      <td>-0.071842</td>
      <td>0.226222</td>
      <td>-0.628398</td>
      <td>-0.236671</td>
      <td>-0.049314</td>
      <td>0.084942</td>
      <td>0.579983</td>
    </tr>
    <tr>
      <th>hi_gr_l1</th>
      <td>480.0</td>
      <td>-0.072171</td>
      <td>0.223957</td>
      <td>-0.620309</td>
      <td>-0.229264</td>
      <td>-0.049571</td>
      <td>0.086760</td>
      <td>0.526463</td>
    </tr>
    <tr>
      <th>hi_gr_l2</th>
      <td>480.0</td>
      <td>-0.072231</td>
      <td>0.226296</td>
      <td>-0.638758</td>
      <td>-0.241208</td>
      <td>-0.045033</td>
      <td>0.081703</td>
      <td>0.561531</td>
    </tr>
    <tr>
      <th>pop_gr</th>
      <td>480.0</td>
      <td>-0.025527</td>
      <td>0.225459</td>
      <td>-0.560511</td>
      <td>-0.155509</td>
      <td>-0.017818</td>
      <td>0.073519</td>
      <td>0.803660</td>
    </tr>
    <tr>
      <th>pop_gr_l1</th>
      <td>480.0</td>
      <td>-0.008310</td>
      <td>0.241158</td>
      <td>-0.560511</td>
      <td>-0.148335</td>
      <td>-0.013102</td>
      <td>0.081756</td>
      <td>0.811545</td>
    </tr>
    <tr>
      <th>pop_gr_l2</th>
      <td>480.0</td>
      <td>0.009073</td>
      <td>0.255137</td>
      <td>-0.560511</td>
      <td>-0.142120</td>
      <td>-0.009678</td>
      <td>0.097848</td>
      <td>0.819558</td>
    </tr>
  </tbody>
</table>
</div>


