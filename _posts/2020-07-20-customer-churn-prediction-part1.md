---
layout: post
title: Customer Churn Prediction (Part 1)
color: green
thumbnail: "assets/img/thumbnails/customer-churn.jpeg"
tags: [Preprocessing, Cleansing, Data Preparation]
---

**DQLab Telco** merupakan perusahaan Telco yang sudah mempunyai banyak cabang tersebar dimana-mana. Sejak berdiri pada tahun 2019, DQLab Telco konsisten untuk memperhatikan customer experience nya sehingga tidak akan di tinggalkan pelanggan. Walaupun baru berumur 1 tahun lebih sedikit, DQLab Telco sudah mempunyai banyak pelanggan yang beralih langganan ke kompetitor. Pihak management ingin mengurangi jumlah pelanggan yang beralih (**churn**) dengan menggunakan machine learning.

Pada Part 1 ini akan dilakukan **data wrangling (data cleansing)** sebelum dilakukan pemodelan agar model prediksi menjadi lebih akurat dengan kualitas data yang lebih terjamin.

## **Dataset**

Dataset yang digunakan berasal dari [DQLab Telco](https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/dqlab_telco.csv). Asumsikan dataset ini merupakan data yang terakhir diperbarui oleh DQLab Telco pada bulan Juni 2020.

Untuk detail datanya adalah sebagai berikut:

* UpdatedAt : Periode of Data taken
* customerID : Customer ID
* gender : Whether the customer is a male or a female (Male, Female)
* SeniorCitizen : Whether the customer is a senior citizen or not (1, 0)
* Partner : Whether the customer has a partner or not (Yes, No)
* Dependents : Whether the customer has dependents or not (Yes, No)
* tenure : Number of months the customer has stayed with the company
* PhoneService : Whether the customer has a phone service or not (Yes, No)
* MultipleLines : Whether the customer has multiple lines or not (Yes, No, No phone service)
* InternetService : Customer’s internet service provider (DSL, Fiber optic, No)
* OnlineSecurity : Whether the customer has online security or not (Yes, No, No internet service)
* OnlineBackup : Whether the customer has online backup or not (Yes, No, No internet service)
* DeviceProtection : Whether the customer has device protection or not (Yes, No, No internet service)
* TechSupport : Whether the customer has tech support or not (Yes, No, No internet service)
* StreamingTV : Whether the customer has streaming TV or not (Yes, No, No internet service)
* StreamingMovies : Whether the customer has streaming movies or not (Yes, No, No internet service)
* Contract : The contract term of the customer (Month-to-month, One year, Two year)
* PaperlessBilling : Whether the customer has paperless billing or not (Yes, No)
* PaymentMethod : The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
* MonthlyCharges : The amount charged to the customer monthly
* TotalCharges : The total amount charged to the customer
* Churn : Whether the customer churned or not (Yes or No)



## **Import Library**

Library yang digunakan dalam pembahasan ini meliputi:
- **pandas** untuk analisis dan manipulasi data
- **matplotlib** untuk visualisasi data


```python
# Load library 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

pd.options.display.max_columns = 50
```

## **Data Cleansing**

Langkah yang akan dilakukan antara lain,
1. Mencari ID pelanggan (nomor telepon) yang valid.
2. Mengatasi data-data yang masih kosong atau *missing values*.
3. Mengatasi nilai-nilai pencilan (outlier) pada variabel numerik.
4. Menstandarisasi nilai dari variabel kategorik.


```python
# Load dataset
df_load = pd.read_csv('data/data_dqlab_telco.csv')

# Print first five rows
df_load.head(5)
```




<div style="overflow-x:auto;">
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
      <th>UpdatedAt</th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>202006</td>
      <td>45759018157</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1.0</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>202006</td>
      <td>45557574145</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.50</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>202006</td>
      <td>45366876421</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>202006</td>
      <td>45779536532</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45.0</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>202006</td>
      <td>45923787906</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>
<br>



```python
# Print number of rows and number of columns
print(f'Dataset ini memiliki {df_load.shape[0]} baris dan {df_load.shape[1]} kolom')

# Print number of unique Customer ID
print('Jumlah ID Pelanggan yang unik adalah',df_load.customerID.nunique())
```

    Dataset ini memiliki 7113 baris dan 22 kolom
    Jumlah ID Pelanggan yang unik adalah 7017
    


```python
# View info data
df_load.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7113 entries, 0 to 7112
    Data columns (total 22 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   UpdatedAt         7113 non-null   int64  
     1   customerID        7113 non-null   object 
     2   gender            7113 non-null   object 
     3   SeniorCitizen     7113 non-null   int64  
     4   Partner           7113 non-null   object 
     5   Dependents        7113 non-null   object 
     6   tenure            7014 non-null   float64
     7   PhoneService      7113 non-null   object 
     8   MultipleLines     7113 non-null   object 
     9   InternetService   7113 non-null   object 
     10  OnlineSecurity    7113 non-null   object 
     11  OnlineBackup      7113 non-null   object 
     12  DeviceProtection  7113 non-null   object 
     13  TechSupport       7113 non-null   object 
     14  StreamingTV       7113 non-null   object 
     15  StreamingMovies   7113 non-null   object 
     16  Contract          7113 non-null   object 
     17  PaperlessBilling  7113 non-null   object 
     18  PaymentMethod     7113 non-null   object 
     19  MonthlyCharges    7087 non-null   float64
     20  TotalCharges      7098 non-null   float64
     21  Churn             7070 non-null   object 
    dtypes: float64(3), int64(2), object(17)
    memory usage: 1.2+ MB
    

### **Mencari Validitas ID Pelanggan**

**Memfilter ID Pelanggan dengan Format Tertentu**

Format ID Pelanggan (Phone Number) yang benar, yaitu:

* Panjang karakter adalah 11-12.
* Terdiri dari angka saja, tidak diperbolehkan ada karakter selain angka.
* Diawali dengan angka 45 pada 2 digit pertama.




```python
# Create new column to check validity Customer ID
df_load['valid_id'] = df_load['customerID'].astype('str').str.match(r'(45\d{9,10})')

# Get rows of valid data
df_load = (df_load[df_load['valid_id'] == True]).drop('valid_id', axis=1)

# Total valid Customer ID with filtering
print('Hasil jumlah ID Pelanggan yang terfilter adalah', df_load['customerID'].count())
```

    Hasil jumlah ID Pelanggan yang terfilter adalah 7006
    

**Memfilter Duplikasi ID pelanggan**

Memastikan bahwa tidak ada ID pelanggan yang duplikat. Biasanya duplikasi ID number ini tipenya:

* Duplikasi dikarenakan inserting melebihi satu kali dengan nilai yang sama tiap kolomnya.
* Duplikasi dikarenakan inserting beda periode pengambilan data.




```python
# Drop duplicate rows
df_load.drop_duplicates()

# Drop duplicate rows Customer ID by periods
df_load = df_load.sort_values('UpdatedAt', ascending=False).drop_duplicates(['customerID'])

# Total valid Customer ID without duplicated
print('Hasil jumlah ID Pelanggan yang sudah dihilangkan duplikasinya adalah',df_load['customerID'].count())
```

    Hasil jumlah ID Pelanggan yang sudah dihilangkan duplikasinya adalah 6993
    

**Validitas dari ID Pelanggan** sangat diperlukan untuk memastikan bahwa data yang kita ambil sudah benar. Berdasarkan hasil tersbut, terdapat perbedaan jumlah ID Pelanggan dari data pertama kali di load sampai dengan hasil validitas. Dimana jumlah baris data ketika pertama kali di load ada sebanyak 7113 baris, lalu setelah di cek validitas dari ID pelanggan, maka terfilter 6993 baris data.

### **Mengecek dan Menangani Missing Values**


Karena tujuan project ini adalah memprediksi **churn** maka data pada kolom `Churn` tidak boleh kosong. Jika ditemukan missing values maka akan dilakukan penghapusan data.


```python
# Missing values kolom Churn
print('Total missing values data dari kolom Churn', df_load['Churn'].isnull().sum())

# Dropping all Rows with spesific column
df_load.dropna(subset=['Churn'], inplace=True)
print('Total data setelah penghapusan missing values adalah', df_load.shape[0])
```

    Total missing values data dari kolom Churn 43
    Total data setelah penghapusan missing values adalah 6950
    

**Mengatasi Missing Values dengan Pengisian Nilai tertentu**

Selain dengan menghapus rows dari data, menangani missing values bisa menggunakan nilai tertentu. Pada project ini diasumsikan data modeller meminta pengisian missing values dengan kriteria berikut:

  *  Tenure pihak data modeller meminta setiap rows yang memiliki missing values untuk Lama berlangganan di isi dengan 11.
  *  Variable yang bersifat numerik selain Tenure di isi dengan median dari masing-masing variable tersebut.





```python
print('Status Missing Values :',df_load.isnull().values.any())
print('\nJumlah Missing Values masing-masing kolom, adalah:')
print(df_load.isnull().sum().sort_values(ascending=False))

# Handling missing values Tenure fill with 11
df_load['tenure'].fillna(11, inplace=True)

# Handling missing values num vars (except Tenure)
for col_name in list(['MonthlyCharges','TotalCharges']):
    median_data = df_load[col_name].median()
    df_load[col_name].fillna(median_data, inplace=True)

print('\nJumlah Missing Values setelah di imputer datanya, adalah:')
print(df_load.isnull().sum().sort_values(ascending=False))
```

    Status Missing Values : True
    
    Jumlah Missing Values masing-masing kolom, adalah:
    tenure              99
    MonthlyCharges      26
    TotalCharges        15
    Churn                0
    InternetService      0
    customerID           0
    gender               0
    SeniorCitizen        0
    Partner              0
    Dependents           0
    PhoneService         0
    MultipleLines        0
    OnlineSecurity       0
    OnlineBackup         0
    DeviceProtection     0
    TechSupport          0
    StreamingTV          0
    StreamingMovies      0
    Contract             0
    PaperlessBilling     0
    PaymentMethod        0
    UpdatedAt            0
    dtype: int64
    
    Jumlah Missing Values setelah di imputer datanya, adalah:
    Churn               0
    TotalCharges        0
    customerID          0
    gender              0
    SeniorCitizen       0
    Partner             0
    Dependents          0
    tenure              0
    PhoneService        0
    MultipleLines       0
    InternetService     0
    OnlineSecurity      0
    OnlineBackup        0
    DeviceProtection    0
    TechSupport         0
    StreamingTV         0
    StreamingMovies     0
    Contract            0
    PaperlessBilling    0
    PaymentMethod       0
    MonthlyCharges      0
    UpdatedAt           0
    dtype: int64
    

Setelah kita analisis lebih lanjut, ternyata masih ada *missing values* dari data yang kita sudah cek validitas ID Pelanggannya. Missing values terdapat pada kolom **Churn, tenure, MonthlyCharges, dan TotalCharges**. Setelah kita tangani dengan cara penghapusan rows dan pengisian rows dengan nilai tertentu, terbukti sudah tidak ada missing values lagi pada data, terbukti dari jumlah missing values masing-masing variable yang bernilai 0. Selanjutnya kita akan melakukan penanganan pencilan (outlier)

### **Mendeteksi dan Mengatasi Outlier**

**Mendeteksi Outlier**

Mendeteksi pencilan dari suatu nilai (outlier) salah satunya bisa menggunakan **Box Plot**. **Box Plot** merupakan ringkasan distribusi sampel yang disajikan secara grafis yang bisa menggambarkan bentuk distribusi data (skewness), ukuran tendensi sentral dan ukuran penyebaran (keragaman) data pengamatan. Outlier biasanya ditemukan pada variabel bertipe numerik.



```python
print('\nPersebaran data sebelum ditangani Outlier: ')
print(df_load.describe())

# Creating Box Plot
fig, ax = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches(15, 6)
ax[0].boxplot(df_load['tenure'])
ax[0].set_xlabel('tenure')
ax[1].boxplot(df_load['MonthlyCharges'])
ax[1].set_xlabel('MonthlyCharges')
ax[2].boxplot(df_load['TotalCharges'])
ax[2].set_xlabel('TotalCharges')
plt.show()
```

    
    Persebaran data sebelum ditangani Outlier: 
           UpdatedAt  SeniorCitizen       tenure  MonthlyCharges  TotalCharges
    count     6950.0    6950.000000  6950.000000     6950.000000   6950.000000
    mean    202006.0       0.162302    32.477266       65.783741   2305.083460
    std          0.0       0.368754    25.188910       50.457871   2578.651143
    min     202006.0       0.000000     0.000000        0.000000     19.000000
    25%     202006.0       0.000000     9.000000       36.462500    406.975000
    50%     202006.0       0.000000    29.000000       70.450000   1400.850000
    75%     202006.0       0.000000    55.000000       89.850000   3799.837500
    max     202006.0       1.000000   500.000000     2311.000000  80000.000000
    


    
![png](/assets/img/customer-churn-prediction/output_25_1.png)
    


Dari ketiga boxplot dengan variable **tenure, MonthlyCharges, dan TotalCharges** terlihat jelas terdapat adanya outlier. Hal ini bisa di identifikasi dari adanya titik titik yang berada jauh dari gambar boxplotnya. Jika kita lihat persebaran datanya dari kolom max nya juga ada nilai yang sangat tinggi sekali. Kemudian nilai outlier tersebut ditangani dengan cara merubah nilainya ke nilai Maximum & Minimum dari interquartile range (IQR).

**Mengatasi Outlier**

Setelah kita mengetahui variable mana saja yang terdapat pencilan (Outlier), selanjutnya kita akan atasi outlier dengan menggunakan **metode interquartile range (IQR)**. 

Tentukan:

  -  Nilai Minimum dan Maximum data di tolerir
  -  Ubah Nilai yg di luar range Minumum & Maximum ke dalam nilai Minimum dan Maximum



```python
# Handling with IQR
Q1 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.25)
Q3 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.75)

IQR = Q3 - Q1
maximum = Q3 + (1.5*IQR)
print('Nilai Maximum dari masing-masing Variable adalah: ')
print(maximum)
minimum = Q1 - (1.5*IQR)
print('\nNilai Minimum dari masing-masing Variable adalah: ')
print(minimum)

more_than = (df_load > maximum)
lower_than = (df_load < minimum)
df_load = df_load.mask(more_than, maximum, axis=1)
df_load = df_load.mask(lower_than, minimum, axis=1)

print('\nPersebaran data setelah ditangani Outlier: ')
print(df_load[['tenure','MonthlyCharges','TotalCharges']].describe())
```

    Nilai Maximum dari masing-masing Variable adalah: 
    tenure             124.00000
    MonthlyCharges     169.93125
    TotalCharges      8889.13125
    dtype: float64
    
    Nilai Minimum dari masing-masing Variable adalah: 
    tenure             -60.00000
    MonthlyCharges     -43.61875
    TotalCharges     -4682.31875
    dtype: float64
    
    Persebaran data setelah ditangani Outlier: 
                tenure  MonthlyCharges  TotalCharges
    count  6950.000000     6950.000000   6950.000000
    mean     32.423165       64.992201   2286.058750
    std      24.581073       30.032040   2265.702553
    min       0.000000        0.000000     19.000000
    25%       9.000000       36.462500    406.975000
    50%      29.000000       70.450000   1400.850000
    75%      55.000000       89.850000   3799.837500
    max     124.000000      169.931250   8889.131250
    

Setelah di tangani outliernya, dan dilihat persebaran datanya, terlihat sudah tidak ada lagi nilai yang outlier.

### **Standarisasi Nilai dari Variabel Kategorikal**

**Mengecek Nilai Tidak Standard**

Selanjutnya kita akan mendeteksi apakah ada nilai-nilai dari variable kategorik yang tidak standard. Hal ini biasanya terjadi dikarenakan kesalahan input data. Perbedaan istilah menjadi salah satu faktor yang sering terjadi, untuk itu dibutuhkan standarisasi dari data yang sudah terinput.



```python
# Check standardization
for col_name in df_load.select_dtypes(include='object').columns:
    print('Unique Values Count Before Standardized Variable',col_name)
    print(df_load[col_name].value_counts())
    print('\n')
```

    Unique Values Count Before Standardized Variable customerID
    45386549500    1
    45577435073    1
    45612256114    1
    45896795719    1
    45513613179    1
                  ..
    45996422740    1
    45740109343    1
    45954614082    1
    45795432264    1
    45077845221    1
    Name: customerID, Length: 6950, dtype: int64
    
    
    Unique Values Count Before Standardized Variable gender
    Male         3499
    Female       3431
    Wanita         14
    Laki-Laki       6
    Name: gender, dtype: int64
    
    
    Unique Values Count Before Standardized Variable Partner
    No     3591
    Yes    3359
    Name: Partner, dtype: int64
    
    
    Unique Values Count Before Standardized Variable Dependents
    No     4870
    Yes    2060
    Iya      20
    Name: Dependents, dtype: int64
    
    
    Unique Values Count Before Standardized Variable PhoneService
    Yes    6281
    No      669
    Name: PhoneService, dtype: int64
    
    
    Unique Values Count Before Standardized Variable MultipleLines
    No                  3346
    Yes                 2935
    No phone service     669
    Name: MultipleLines, dtype: int64
    
    
    Unique Values Count Before Standardized Variable InternetService
    Fiber optic    3057
    DSL            2388
    No             1505
    Name: InternetService, dtype: int64
    
    
    Unique Values Count Before Standardized Variable OnlineSecurity
    No                     3454
    Yes                    1991
    No internet service    1505
    Name: OnlineSecurity, dtype: int64
    
    
    Unique Values Count Before Standardized Variable OnlineBackup
    No                     3045
    Yes                    2400
    No internet service    1505
    Name: OnlineBackup, dtype: int64
    
    
    Unique Values Count Before Standardized Variable DeviceProtection
    No                     3054
    Yes                    2391
    No internet service    1505
    Name: DeviceProtection, dtype: int64
    
    
    Unique Values Count Before Standardized Variable TechSupport
    No                     3431
    Yes                    2014
    No internet service    1505
    Name: TechSupport, dtype: int64
    
    
    Unique Values Count Before Standardized Variable StreamingTV
    No                     2774
    Yes                    2671
    No internet service    1505
    Name: StreamingTV, dtype: int64
    
    
    Unique Values Count Before Standardized Variable StreamingMovies
    No                     2747
    Yes                    2698
    No internet service    1505
    Name: StreamingMovies, dtype: int64
    
    
    Unique Values Count Before Standardized Variable Contract
    Month-to-month    3823
    Two year          1670
    One year          1457
    Name: Contract, dtype: int64
    
    
    Unique Values Count Before Standardized Variable PaperlessBilling
    Yes    4114
    No     2836
    Name: PaperlessBilling, dtype: int64
    
    
    Unique Values Count Before Standardized Variable PaymentMethod
    Electronic check             2337
    Mailed check                 1594
    Bank transfer (automatic)    1519
    Credit card (automatic)      1500
    Name: PaymentMethod, dtype: int64
    
    
    Unique Values Count Before Standardized Variable Churn
    No       5114
    Yes      1827
    Churn       9
    Name: Churn, dtype: int64
    
    
    

Ketika kita amati lebih jauh dari jumlah unique value dari masing-masing variable kategorik, terlihat jelas bahwa ada beberapa variable yang tidak standar. Variable itu adalah :

  -  **gender** (Female, Male, Wanita, Laki-Laki), yang bisa di standarkan nilainya menjadi (Female, Male) karena mempunyai makna yang sama.
  - **Dependents** (Yes, No, Iya), yang bisa di standarkan nilainya menjadi (Yes, No) karena mempunyai makna yang sama.
  -  **Churn** (Yes, No, Churn), yang bisa di standarkan nilainya menjadi (Yes, No) karena mempunyai makna yang sama.

**Menstandarisasi Variable Kategorik**

Setelah kita mengetahui variable mana saja yang ada nilai tidak standar, maka kita standarkan dengan pola terbanyak nya, dengan syarat tanpa mengubah maknanya. Contoh : Iya -> Yes


```python
# Change values
df_load = df_load.replace(['Wanita','Laki-Laki','Churn','Iya'],['Female','Male','Yes','Yes'])

# Check values after standardization
for col_name in list(['gender','Dependents','Churn']):
    print('Unique Values Count After Standardized Variable',col_name)
    print(df_load[col_name].value_counts())
    print('\n')
```

    Unique Values Count After Standardized Variable gender
    Male      3505
    Female    3445
    Name: gender, dtype: int64
    
    
    Unique Values Count After Standardized Variable Dependents
    No     4870
    Yes    2080
    Name: Dependents, dtype: int64
    
    
    Unique Values Count After Standardized Variable Churn
    No     5114
    Yes    1836
    Name: Churn, dtype: int64
    
    
    

Terlihat bahwa nilai dari variabel sudah standar dan siap dianggap sudah bersih atau *clean*.

**Save Clean Dataset**

Setelah raw data dilakukan data cleansing, selanjutnya data ini dapat digunakan untuk pemodelan.


```python
# Print first five rows clean data
df_load.head()
```




<div style="overflow-x:auto;">
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
      <th>UpdatedAt</th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>202006</td>
      <td>45759018157</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1.0</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4727</th>
      <td>202006</td>
      <td>45315483266</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>60.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>20.50</td>
      <td>1198.80</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4738</th>
      <td>202006</td>
      <td>45236961615</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>5.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>104.10</td>
      <td>541.90</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4737</th>
      <td>202006</td>
      <td>45929827382</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>72.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>115.50</td>
      <td>8312.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4736</th>
      <td>202006</td>
      <td>45305082233</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>56.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Two year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>81.25</td>
      <td>4620.40</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>
<br>



```python
# Save clean data
df_load.reset_index(drop=True, inplace=True)
df_load.to_csv('data/data_cleansing_telco.csv', index=False)
```
