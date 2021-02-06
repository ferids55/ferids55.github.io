---
layout: post
title: Churn Analyis
color: skyblue
thumbnail: "assets/img/thumbnails/customer-churn.jpeg"
tags: [Classification, Machine Learning]
---

Hopefully you will find enough information about how to set images in your blog here.
This is an example of a post which includes a feature image specified in the front matter of the post. 
The feature image spans the full-width of the page, and is shown with the title on permalink pages:

# CUSTOMER CHURN ANALYSIS

## Import Library


```python
# Library for data analysis
import numpy as np
import pandas as pd

# Library for visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Library for machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Setting parameter kernel
plt.rcParams['figure.figsize'] = [10,6]
sns.set_style('darkgrid')
```

## Data Preparation

### Import Raw Data


```python
# Read file as dataframe
df = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/data_retail.csv', sep = ';')
# Print first five rows
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>no</th>
      <th>Row_Num</th>
      <th>Customer_ID</th>
      <th>Product</th>
      <th>First_Transaction</th>
      <th>Last_Transaction</th>
      <th>Average_Transaction_Amount</th>
      <th>Count_Transaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>29531</td>
      <td>Jaket</td>
      <td>1466304274396</td>
      <td>1538718482608</td>
      <td>1467681</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>29531</td>
      <td>Sepatu</td>
      <td>1406077331494</td>
      <td>1545735761270</td>
      <td>1269337</td>
      <td>41</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>141526</td>
      <td>Tas</td>
      <td>1493349147000</td>
      <td>1548322802000</td>
      <td>310915</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>141526</td>
      <td>Jaket</td>
      <td>1493362372547</td>
      <td>1547643603911</td>
      <td>722632</td>
      <td>27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>37545</td>
      <td>Sepatu</td>
      <td>1429178498531</td>
      <td>1542891221530</td>
      <td>1775036</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Info dataset:')
df.info()
```

    Info dataset:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 8 columns):
     #   Column                      Non-Null Count   Dtype 
    ---  ------                      --------------   ----- 
     0   no                          100000 non-null  int64 
     1   Row_Num                     100000 non-null  int64 
     2   Customer_ID                 100000 non-null  int64 
     3   Product                     100000 non-null  object
     4   First_Transaction           100000 non-null  int64 
     5   Last_Transaction            100000 non-null  int64 
     6   Average_Transaction_Amount  100000 non-null  int64 
     7   Count_Transaction           100000 non-null  int64 
    dtypes: int64(7), object(1)
    memory usage: 6.1+ MB


### Data Wrangling


```python
# Change format column First_Transaction and Last Transaction
df['First_Transaction'] = pd.to_datetime(df['First_Transaction']/1000, unit='s', origin='1970-01-01')
df['Last_Transaction'] = pd.to_datetime(df['Last_Transaction']/1000, unit='s', origin='1970-01-01')
# View new format
df[['First_Transaction','Last_Transaction']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>First_Transaction</th>
      <th>Last_Transaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-06-19 02:44:34.395999908</td>
      <td>2018-10-05 05:48:02.608000040</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-07-23 01:02:11.493999958</td>
      <td>2018-12-25 11:02:41.269999981</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-04-28 03:12:27.000000000</td>
      <td>2019-01-24 09:40:02.000000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-04-28 06:52:52.546999931</td>
      <td>2019-01-16 13:00:03.911000013</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-04-16 10:01:38.530999899</td>
      <td>2018-11-22 12:53:41.529999970</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check Last Transaction update
print(max(df['Last_Transaction']))
```

    2019-02-01 23:57:57.286000013



```python
# Create new column is_Churn
df.loc[df['Last_Transaction'] <= '2018-08-01', 'is_Churn'] = True 
df.loc[df['Last_Transaction'] > '2018-08-01', 'is_Churn'] = False 
```


```python
# Remove unnecessary columns
del df['no']
del df['Row_Num']
```

### Clean Dataset


```python
# View first five rows after cleaning data
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer_ID</th>
      <th>Product</th>
      <th>First_Transaction</th>
      <th>Last_Transaction</th>
      <th>Average_Transaction_Amount</th>
      <th>Count_Transaction</th>
      <th>is_Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29531</td>
      <td>Jaket</td>
      <td>2016-06-19 02:44:34.395999908</td>
      <td>2018-10-05 05:48:02.608000040</td>
      <td>1467681</td>
      <td>22</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29531</td>
      <td>Sepatu</td>
      <td>2014-07-23 01:02:11.493999958</td>
      <td>2018-12-25 11:02:41.269999981</td>
      <td>1269337</td>
      <td>41</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>141526</td>
      <td>Tas</td>
      <td>2017-04-28 03:12:27.000000000</td>
      <td>2019-01-24 09:40:02.000000000</td>
      <td>310915</td>
      <td>30</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>141526</td>
      <td>Jaket</td>
      <td>2017-04-28 06:52:52.546999931</td>
      <td>2019-01-16 13:00:03.911000013</td>
      <td>722632</td>
      <td>27</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37545</td>
      <td>Sepatu</td>
      <td>2015-04-16 10:01:38.530999899</td>
      <td>2018-11-22 12:53:41.529999970</td>
      <td>1775036</td>
      <td>25</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# View info dataset
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 7 columns):
     #   Column                      Non-Null Count   Dtype         
    ---  ------                      --------------   -----         
     0   Customer_ID                 100000 non-null  int64         
     1   Product                     100000 non-null  object        
     2   First_Transaction           100000 non-null  datetime64[ns]
     3   Last_Transaction            100000 non-null  datetime64[ns]
     4   Average_Transaction_Amount  100000 non-null  int64         
     5   Count_Transaction           100000 non-null  int64         
     6   is_Churn                    100000 non-null  object        
    dtypes: datetime64[ns](2), int64(3), object(2)
    memory usage: 5.3+ MB


## Data Visualization

### Customer Acquisition By Year


```python
# Create new column Year First Transaction
df['Year_First_Transaction'] = df['First_Transaction'].dt.year
# Create new column Last Transaction
df['Year_Last_Transaction'] = df['Last_Transaction'].dt.year
# Grouping Number of Customers by Year
cust_by_year = df.groupby(['Year_First_Transaction'])['Customer_ID'].count()
cust_by_year
```




    Year_First_Transaction
    2013     1007
    2014     4954
    2015    11235
    2016    17656
    2017    31828
    2018    30327
    2019     2993
    Name: Customer_ID, dtype: int64




```python
# Plotting number of customers by year
sns.barplot(x = cust_by_year.index, y = cust_by_year, palette='Set2')
plt.title('Graph of Customer Acquisition', fontsize=15, pad=15)
plt.xlabel('Year Transaction', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
```




    Text(0, 0.5, 'Number of Customers')




![png](/assets/img/churn-analysis/output_18_1.png)


### Transaction by Year


```python
# Grouping total Count Transaction by Year
trans_by_year = df.groupby(['Year_First_Transaction'])['Count_Transaction'].sum()
trans_by_year
```




    Year_First_Transaction
    2013     23154
    2014    165494
    2015    297445
    2016    278707
    2017    299199
    2018     99989
    2019      5862
    Name: Count_Transaction, dtype: int64




```python
# Plotting total transactions by year
sns.barplot(x = trans_by_year.index, y = trans_by_year, palette='Set2')
plt.title('Graph of Transaction Customer', fontsize=15, pad=15)
plt.xlabel('Year Transaction', fontsize=12)
plt.ylabel('Total Transactions', fontsize=12)
```




    Text(0, 0.5, 'Total Transactions')




![png](/assets/img/churn-analysis/output_21_1.png)


### Average Transaction Amount by Year


```python
# Grouping Average Transaction Amount by Product and Year
avg_trans_amount = df.groupby(['Product', 'Year_First_Transaction'])['Average_Transaction_Amount'].mean().reset_index()
avg_trans_amount
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product</th>
      <th>Year_First_Transaction</th>
      <th>Average_Transaction_Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Baju</td>
      <td>2017</td>
      <td>1.490890e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Baju</td>
      <td>2018</td>
      <td>1.570201e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Baju</td>
      <td>2019</td>
      <td>1.383645e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jaket</td>
      <td>2014</td>
      <td>1.467937e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jaket</td>
      <td>2015</td>
      <td>1.296265e+06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Jaket</td>
      <td>2016</td>
      <td>1.317344e+06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Jaket</td>
      <td>2017</td>
      <td>1.369034e+06</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jaket</td>
      <td>2018</td>
      <td>1.419074e+06</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Jaket</td>
      <td>2019</td>
      <td>1.447536e+06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sepatu</td>
      <td>2013</td>
      <td>1.396499e+06</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sepatu</td>
      <td>2014</td>
      <td>1.427063e+06</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sepatu</td>
      <td>2015</td>
      <td>1.428235e+06</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sepatu</td>
      <td>2016</td>
      <td>1.425938e+06</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sepatu</td>
      <td>2017</td>
      <td>1.407275e+06</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sepatu</td>
      <td>2018</td>
      <td>1.346824e+06</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Sepatu</td>
      <td>2019</td>
      <td>1.338180e+06</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Tas</td>
      <td>2017</td>
      <td>1.109583e+06</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Tas</td>
      <td>2018</td>
      <td>1.337614e+06</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Tas</td>
      <td>2019</td>
      <td>1.287529e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting by Product
sns.pointplot(data = avg_trans_amount, x='Year_First_Transaction', 
              y='Average_Transaction_Amount', hue='Product')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.01))
plt.title('Average Transaction Amount by Year', fontsize=15, pad=15)
plt.xlabel('Year Transaction', fontsize=12)
plt.ylabel('Average Transaction Amount', fontsize=12)
```




    Text(0, 0.5, 'Average Transaction Amount')




![png](/assets/img/churn-analysis/output_24_1.png)


### Churn Customer Proportion


```python
# Create pivot table
churn_prop = df.pivot_table(index='is_Churn', 
                        columns='Product',
                        values='Customer_ID', 
                        aggfunc='count', 
                        fill_value=0)
churn_prop
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Product</th>
      <th>Baju</th>
      <th>Jaket</th>
      <th>Sepatu</th>
      <th>Tas</th>
    </tr>
    <tr>
      <th>is_Churn</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>1268</td>
      <td>11123</td>
      <td>16064</td>
      <td>4976</td>
    </tr>
    <tr>
      <th>True</th>
      <td>2144</td>
      <td>23827</td>
      <td>33090</td>
      <td>7508</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting churn proportion using pie plot
churn_prop.plot.pie(subplots=True, layout=(2,2), autopct='%1.0f%%', 
                    legend=False, title='Churn Proportion by Product')
plt.tight_layout()
```


![png](/assets/img/churn-analysis/output_27_0.png)


### Distribution Count Transaction


```python
# Create function distribution count transaction
def distribution_count_trans(row):
    if row['Count_Transaction'] == 1:
        val = '1'
    elif (row['Count_Transaction'] > 1 and row['Count_Transaction'] <= 3):
        val ='2 - 3'
    elif (row['Count_Transaction'] > 3 and row['Count_Transaction'] <= 6):
        val ='4 - 6'
    elif (row['Count_Transaction'] > 6 and row['Count_Transaction'] <= 10):
        val ='7 - 10'
    else:
        val ='> 10'
    return val
```


```python
# Apply function to new column
df['Count_Transaction_Group'] = df.apply(distribution_count_trans, axis=1)
# Grouping Number of Customers by Count Transaaction Group
count_trans_group = df.groupby(['Count_Transaction_Group'])['Customer_ID'].count()
count_trans_group
```




    Count_Transaction_Group
    1         49255
    2 - 3     14272
    4 - 6     12126
    7 - 10     2890
    > 10      21457
    Name: Customer_ID, dtype: int64




```python
# Plotting count transaction group 
sns.barplot(x = count_trans_group.index, y = count_trans_group, palette='Set2')
plt.title('Customer Distribution by Count Transaction Group', fontsize=15, pad=15)
plt.xlabel('Transaction Group', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
```




    Text(0, 0.5, 'Number of Customers')




![png](/assets/img/churn-analysis/output_31_1.png)


### Distribution Average Transaction


```python
# Create function distribution average transaction
def distribution_avg_trans(row):
    if (row['Average_Transaction_Amount'] >= 100000 and row['Average_Transaction_Amount'] <= 200000):
        val ='100.000 - 250.000'
    elif (row['Average_Transaction_Amount'] > 250000 and row['Average_Transaction_Amount'] <= 500000):
        val ='>250.000 - 500.000'
    elif (row['Average_Transaction_Amount'] > 500000 and row['Average_Transaction_Amount'] <= 750000):
        val ='>500.000 - 750.000'
    elif (row['Average_Transaction_Amount'] > 750000 and row['Average_Transaction_Amount'] <= 1000000):
        val ='>750.000 - 1.000.000'
    elif (row['Average_Transaction_Amount'] > 1000000 and row['Average_Transaction_Amount'] <= 2500000):
        val ='>1.000.000 - 2.500.000'
    elif (row['Average_Transaction_Amount'] > 2500000 and row['Average_Transaction_Amount'] <= 5000000):
        val ='>2.500.000 - 5.000.000'
    elif (row['Average_Transaction_Amount'] > 5000000 and row['Average_Transaction_Amount'] <= 10000000):
        val ='>5.000.000 - 10.000.000'
    else:
        val ='>10.000.000'
    return val
```


```python
# Apply function to new column
df['Average_Transaction_Amount_Group'] = df.apply(distribution_avg_trans, axis=1)
# Grouping Number of Customers by Average Transaction Amount Group
avg_trans_group = df.groupby(['Average_Transaction_Amount_Group'])['Customer_ID'].count()
avg_trans_group
```




    Average_Transaction_Amount_Group
    100.000 - 250.000           4912
    >1.000.000 - 2.500.000     32819
    >10.000.000                 3227
    >2.500.000 - 5.000.000      9027
    >250.000 - 500.000         18857
    >5.000.000 - 10.000.000     3689
    >500.000 - 750.000         15171
    >750.000 - 1.000.000       12298
    Name: Customer_ID, dtype: int64




```python
# Plotting average transaction amount group
orders = ['100.000 - 250.000', '>250.000 - 500.000', '>500.000 - 750.000',
          '>750.000 - 1.000.000', '>1.000.000 - 2.500.000', '>2.500.000 - 5.000.000',
          '>5.000.000 - 10.000.000', '>10.000.000']
sns.barplot(x = avg_trans_group, y = avg_trans_group.index, order=orders, palette='Set2')
plt.title('Customer Distribution by Average Transaction Amount Group', fontsize=15, pad=15)
plt.xlabel('Number of Customers', fontsize=12)
plt.ylabel('Average Transaction Amount Group', fontsize=12)
```




    Text(0, 0.5, 'Average Transaction Amount Group')




![png](/assets/img/churn-analysis/output_35_1.png)


## Data Modelling

### Feature Engineering


```python
# Check datatypes
df.dtypes
```




    Customer_ID                                  int64
    Product                                     object
    First_Transaction                   datetime64[ns]
    Last_Transaction                    datetime64[ns]
    Average_Transaction_Amount                   int64
    Count_Transaction                            int64
    is_Churn                                    object
    Year_First_Transaction                       int64
    Year_Last_Transaction                        int64
    Count_Transaction_Group                     object
    Average_Transaction_Amount_Group            object
    dtype: object




```python
# Copy old dataframe
df_model = df.copy()
# Create new column from First Transaction
df_model['Month_First'] = df_model['First_Transaction'].dt.month
df_model['Day_First'] = df_model['First_Transaction'].dt.day
# Create new column from Last Transaction
df_model['Month_Last'] = df_model['Last_Transaction'].dt.month
df_model['Day_Last'] = df_model['Last_Transaction'].dt.day
```


```python
# Encoding categorical data
df_model['Count_Transaction_Group'].replace({'1':0, '2 - 3':1, '4 - 6':2, '7 - 10':3,
                                             '> 10':4}, inplace=True)
amounts = {'100.000 - 250.000':0, '>250.000 - 500.000':1, '>500.000 - 750.000':2,
          '>750.000 - 1.000.000':3, '>1.000.000 - 2.500.000':4, '>2.500.000 - 5.000.000':5,
          '>5.000.000 - 10.000.000':6, '>10.000.000':7}
df_model['Average_Transaction_Amount_Group'].replace(amounts, inplace=True)
df_model['is_Churn'].replace({False:0, True:1}, inplace=True)
```


```python
# Perform one hot encoding
df_encode = pd.get_dummies(df_model['Product'])
# Selected columns
remove_cols = ['Product', 'First_Transaction', 'Last_Transaction', 'Count_Transaction', 'Average_Transaction_Amount']
df_fix = df_model.drop(remove_cols, axis=1)
# Concatenate dataframe
df_model_fix = pd.concat([df_fix, df_encode], axis=1)

# Check new first five rows
df_model_fix.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer_ID</th>
      <th>is_Churn</th>
      <th>Year_First_Transaction</th>
      <th>Year_Last_Transaction</th>
      <th>Count_Transaction_Group</th>
      <th>Average_Transaction_Amount_Group</th>
      <th>Month_First</th>
      <th>Day_First</th>
      <th>Month_Last</th>
      <th>Day_Last</th>
      <th>Baju</th>
      <th>Jaket</th>
      <th>Sepatu</th>
      <th>Tas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29531</td>
      <td>0</td>
      <td>2016</td>
      <td>2018</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>19</td>
      <td>10</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29531</td>
      <td>0</td>
      <td>2014</td>
      <td>2018</td>
      <td>4</td>
      <td>4</td>
      <td>7</td>
      <td>23</td>
      <td>12</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>141526</td>
      <td>0</td>
      <td>2017</td>
      <td>2019</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>28</td>
      <td>1</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>141526</td>
      <td>0</td>
      <td>2017</td>
      <td>2019</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>28</td>
      <td>1</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37545</td>
      <td>0</td>
      <td>2015</td>
      <td>2018</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>16</td>
      <td>11</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check new data type
df_model_fix.dtypes
```




    Customer_ID                         int64
    is_Churn                            int64
    Year_First_Transaction              int64
    Year_Last_Transaction               int64
    Count_Transaction_Group             int64
    Average_Transaction_Amount_Group    int64
    Month_First                         int64
    Day_First                           int64
    Month_Last                          int64
    Day_Last                            int64
    Baju                                uint8
    Jaket                               uint8
    Sepatu                              uint8
    Tas                                 uint8
    dtype: object




```python
# View Total Proportion Churn
df_model_fix['is_Churn'].value_counts(normalize=True)
```




    1    0.66569
    0    0.33431
    Name: is_Churn, dtype: float64



### Split Train and Test Data


```python
# Choose feature columns
X = df_model_fix.drop(['is_Churn'], axis=1)
# Choose target columns
y = df_model_fix['is_Churn']

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

### Model Selection


```python
# Create function choose model
def choose_model(model_name):
  # Train model
  model = model_name.fit(X_train, y_train)
  # Scoring Model
  model_score = model.score(X_test, y_test)
  print(f'Score Accuracy is {round(model_score*100, 2)}%')
  # Validating Model
  valid_score = np.mean(cross_val_score(model_name, X_train, y_train, cv=5))
  print(f'Score Validation is {round(valid_score*100, 2)}%')
```


```python
# Choose model
choose_model(LogisticRegression())
```

    Score Accuracy is 72.19%
    Score Validation is 74.93%



```python
# Choose model
choose_model(xgb.XGBClassifier())
```

    Score Accuracy is 100.0%
    Score Validation is 100.0%


### Model Evaluation


```python
# Function for plot confusion matrix and classification report
def check_matrix_and_reports(model_plot):
  # Churn Prediction 
  train = model_plot.fit(X_train, y_train)
  preds = train.predict(X_test)

  # Create confusion matrix
  class_names = ['Not Churn', 'Churn'] 
  cnf_matrix = confusion_matrix(y_test, preds)
  # Use subplotting
  fig, ax = plt.subplots(figsize=(8,4))
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  # Plotting using heatmap
  sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
  plt.title('Confusion matrix', fontsize=15, pad=15)
  plt.xlabel('Actual')
  plt.ylabel('Predicted')
  
  # Create classification report
  reports = classification_report(y_test, preds, target_names=class_names)
  print('Classification Report\n', reports)
```


```python
# Apply function with Logistic Regression
check_matrix_and_reports(LogisticRegression())
```

    Classification Report
                   precision    recall  f1-score   support
    
       Not Churn       0.70      0.31      0.43      8399
           Churn       0.73      0.93      0.82     16601
    
        accuracy                           0.72     25000
       macro avg       0.71      0.62      0.62     25000
    weighted avg       0.72      0.72      0.69     25000
    



![png](/assets/img/churn-analysis/output_52_1.png)



```python
# Apply function with XGBoost Classifier
check_matrix_and_reports(xgb.XGBClassifier())
```

    Classification Report
                   precision    recall  f1-score   support
    
       Not Churn       1.00      1.00      1.00      8399
           Churn       1.00      1.00      1.00     16601
    
        accuracy                           1.00     25000
       macro avg       1.00      1.00      1.00     25000
    weighted avg       1.00      1.00      1.00     25000
    



![png](/assets/img/churn-analysis/output_53_1.png)



```python
# Predict probability of churn
model1 = LogisticRegression().fit(X_train, y_train)
log_pred = model1.predict_proba(X_test)[:,1]
model2 = xgb.XGBClassifier().fit(X_train, y_train)
xgb_pred = model2.predict_proba(X_test)[:,1]

# ROC Chart components
fallout_log, sensitivity_log, thresholds_log = roc_curve(y_test, log_pred)
fallout_xgb, sensitivity_xgb, thresholds_xgb = roc_curve(y_test, xgb_pred)

# ROC Chart with both
plt.plot(fallout_log, sensitivity_log, color = 'blue', label='%s' % 'Logistic Regression')
plt.plot(fallout_xgb, sensitivity_xgb, color = 'green', label='%s' % 'XGBoost')
plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart on Probability of Churn")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f2d2a3b0f90>




![png](/assets/img/churn-analysis/output_54_1.png)



```python
# Print the logistic regression AUC with formatting
print("Logistic Regression AUC Score: %0.2f" % roc_auc_score(y_test, log_pred))

# Print the xgboost classifier AUC with formatting
print("XGBoost Classifier AUC Score: %0.2f" % roc_auc_score(y_test, xgb_pred))
```

    Logistic Regression AUC Score: 0.58
    XGBoost Classifier AUC Score: 1.00

