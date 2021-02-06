---
layout: post
title: Customer Segmentation
color: blue
thumbnail: "assets/img/thumbnails/customer-segmentation.jpg"
tags: [Customer Segmentation, Preprocessing, K-Prototypes]
---

Sebuah perusahaan ingin berusaha untuk mengenal lebih baik pelanggan nya. Tujuannya agar perusahaan dapat membuat strategi pemasaran yang lebih tepat dan juga efisien bagi tiap tiap pelanggan. Untuk menyelesaikan masalah tersebut terdapat salah satu metode yang bisa dilakukan yaitu melakukan **segmentasi pelanggan**. **Segmentasi Pelanggan** adalah mengelompokkan pelanggan-pelanggan yang ada berdasarkan kesamaan karakter dari pelanggan tersebut. Untuk melakukan hal tersebut kamu akan menggunakan teknik **clustering***.

**Clustering** adalah proses pembagian objek-objek ke dalam beberapa kelompok atau *cluster* berdasarkan tingkat kemiripan antara satu objek dengan yang lain. Salah satunya teknik *clustering* yang dapat digunakan adalah algoritma **K-Prototypes**, dimana  algoritma **K-Prototypes** merupakan gabungan dari **K-means** dan juga **K-modes**. **K-means** itu sendiri biasa nya hanya digunakan untuk data-data yang bersifat numerik. Sedangkan untuk yang bersifat kategorikal saja, kita bisa menggunakan **K-modes**. Untuk dokumentasi lebih lanjut mengenail algoritma K-Prototypes bisa anda lihat [**disini**](https://github.com/nicodv/kmodes).

## Dataset

Dataset yang digunakan dalam pembahasan ini berasal dari [https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/customer_segments.txt](https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/customer_segments.txt).

Data tersebut memiliki tujuh kolom dengan penjelasan sebagai berikut:

- Customer ID: Kode pelanggan dengan format campuran teks CUST- diikuti angka
- Nama Pelanggan: Nama dari pelanggan dengan format teks tentunya
- Jenis Kelamin: Jenis kelamin dari pelanggan, hanya terdapat dua isi data kategori yaitu Pria dan Wanita
- Umur: Umur dari pelanggan dalam format angka
- Profesi: Profesi dari pelanggan, juga bertipe teks kategori yang terdiri dari Wiraswasta, Pelajar, Professional, Ibu Rumah Tangga, dan Mahasiswa.
- Tipe Residen: Tipe tempat tinggal dari pelanggan kita, untuk dataset ini hanya ada dua kategori: Cluster dan Sector.
- Nilai Belanja Setahun: Merupakan total belanja yang sudah di keluarkan oleh pelanggan tersebut.

## Library

Masalah ini akan dapat di selesaikan dengan menggunakan bantuan library - library di bawah ini:

- Pandas, di gunakan untuk melakukan pemrosesan analisis data
- Matplotlib, di gunakan sebagai dasar untuk melakukan visualisasi data
- Seaborn, di gunakan di atas matplotlib untuk melakukan data visualisasi yang lebih menarik
- Scikit-Learn, digunakan untuk mempersiapkan data sebelum dilakukan permodelan
- Kmodes, digunakan untuk melakukan permodelan menggunakan algoritma K-Modes dan K-Prototypes.
- Pickle, digunakan untuk melakukan penyimpanan dari model yang akan di buat.

Library **kmodes** harus dinstall terlebih dahulu dengan cara berikut.


```python
# Install kmodes
!pip install kmodes
```

    Requirement already satisfied: kmodes in c:\users\feri\anaconda3\lib\site-packages (0.10.2)
    Requirement already satisfied: scikit-learn>=0.19.0 in c:\users\feri\anaconda3\lib\site-packages (from kmodes) (0.23.2)
    Requirement already satisfied: scipy>=0.13.3 in c:\users\feri\anaconda3\lib\site-packages (from kmodes) (1.5.2)
    Requirement already satisfied: joblib>=0.11 in c:\users\feri\anaconda3\lib\site-packages (from kmodes) (0.17.0)
    Requirement already satisfied: numpy>=1.10.4 in c:\users\feri\anaconda3\lib\site-packages (from kmodes) (1.19.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\feri\anaconda3\lib\site-packages (from scikit-learn>=0.19.0->kmodes) (2.1.0)
    


```python
# Load library
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import LabelEncoder  
  
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes  
  
import pickle  
from pathlib import Path 

import warnings
warnings.filterwarnings('ignore')
```

## Load Dataset

Melakukan pembacaan dataset ke dalam bentuk dataframe, kemudian melihat preview data dan informasi data yang terbaca.


```python
# Load file into dataframe 
df = pd.read_csv("data/customer_segments.txt", sep="\t")  
  
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
      <th>Customer_ID</th>
      <th>Nama Pelanggan</th>
      <th>Jenis Kelamin</th>
      <th>Umur</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>NilaiBelanjaSetahun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST-001</td>
      <td>Budi Anggara</td>
      <td>Pria</td>
      <td>58</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>9497927</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUST-002</td>
      <td>Shirley Ratuwati</td>
      <td>Wanita</td>
      <td>14</td>
      <td>Pelajar</td>
      <td>Cluster</td>
      <td>2722700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUST-003</td>
      <td>Agus Cahyono</td>
      <td>Pria</td>
      <td>48</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5286429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUST-004</td>
      <td>Antonius Winarta</td>
      <td>Pria</td>
      <td>53</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5204498</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUST-005</td>
      <td>Ibu Sri Wahyuni, IR</td>
      <td>Wanita</td>
      <td>41</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>10615206</td>
    </tr>
  </tbody>
</table>
</div>




```python
# View info data
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50 entries, 0 to 49
    Data columns (total 7 columns):
     #   Column               Non-Null Count  Dtype 
    ---  ------               --------------  ----- 
     0   Customer_ID          50 non-null     object
     1   Nama Pelanggan       50 non-null     object
     2   Jenis Kelamin        50 non-null     object
     3   Umur                 50 non-null     int64 
     4   Profesi              50 non-null     object
     5   Tipe Residen         50 non-null     object
     6   NilaiBelanjaSetahun  50 non-null     int64 
    dtypes: int64(2), object(5)
    memory usage: 2.9+ KB
    

Terlihat tidak ditemukan *missing values* dan *wrong format*.

## Exploratory Data Analysis

Proses eksplorasi data bisa berupa **univariate** maupun **multivariate**. **Univariate Analysis** melihat karakteristik tiap-tiap feature, misal nya dengan melihat statistik deskriptif, membuat histogram, kdeplot, count plot maupun boxplot. Sedangkan untuk **Multivariate Analysis**, kita melihat hubungan tiap variabel dengan variabel lainnya, misal kan dengan membuat korelasi matrix, melihat predictive power, cross tabulasi, dan lainnya. Disini kita akan melakukan *univariate analysis* terhadap kolom bertipe numerik dan kategorik.

**EDA Numerical Data**

Melakukan analisa distribusi data pada kolom bertipe numerik dengan menggunakan visualisasi **boxplot** dan **histogram**. Kolom yang bertipe numerik adalah `Umur` dan `NilaiBelanjaSetahun`.


```python
# Setting canvas
sns.set(style='white')
  
# Function plotting numerical data
def observasi_num(features):  
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    for i, kol in enumerate(features):
        sns.boxplot(df[kol], ax = axs[i][0])
        sns.distplot(df[kol], ax = axs[i][1])   
        axs[i][0].set_title('mean = %.2f\n median = %.2f\n std = %.2f'%
                         (df[kol].mean(), df[kol].median(), df[kol].std()))
    plt.tight_layout()
    plt.show()  

# Apply function
kolom_numerik = ['Umur','NilaiBelanjaSetahun'] 
observasi_num(kolom_numerik) 
```


    
![png](/assets/img/customer-segmentation/output_17_0.png)
    


**EDA Categorical Data**

Melakukan analisa distribusi data pada kolom bertipe kategorik dengan menggunakan visualisasi **count plot**. Kolom yang bertipe kategorik yaitu `Jenis Kelamin`, `Profesi` dan `Tipe Residen`.


```python
# Define categorical column
kolom_kategorikal = ['Jenis Kelamin','Profesi','Tipe Residen']  

# Setting canvas
fig, axs = plt.subplots(3,1,figsize=(7,10)) 

# Plotting categorical data 
for i, kol in enumerate(kolom_kategorikal):  
    # Plotting data
    sns.countplot(df[kol], order = df[kol].value_counts().index, ax = axs[i])  
    axs[i].set_title('\nCount Plot %s\n'%(kol), fontsize=15)  
      
    # Create annotate  
    for p in axs[i].patches:  
        axs[i].annotate(format(p.get_height(), '.0f'),  
                        (p.get_x() + p.get_width() / 2., p.get_height()),  
                        ha = 'center',  
                        va = 'center',  
                        xytext = (0, 10),  
                        textcoords = 'offset points') 
          
    # Setting Plot  
    sns.despine(right=True,top = True, left = True)  
    axs[i].axes.yaxis.set_visible(False)
    plt.tight_layout()

plt.show()
```


    
![png](/assets/img/customer-segmentation/output_19_0.png)
    


Berdasarkan hasil visualisasi diatas diketahui bahwa:

- Rata-rata dari umur pelanggan adalah 37.5 tahun.
- Rata-rata dari nilai belanja setahun pelanggan adalah 7,069,874.82.
- Jenis kelamin pelanggan di dominasi oleh wanita sebanyak 41 orang (82%) dan laki-laki sebanyak 9 orang (18%).
- Profesi terbanyak adalah Wiraswasta (40%) diikuti dengan Professional (36%) dan lainnya sebanyak (24%).
- Dari seluruh pelanggan 64% dari mereka tinggal di Cluster dan 36% nya tinggal di Sector.

## Preprocessing Data

Setiap machine learning model memiliki karakteristik yang berbeda-beda. Hal ini membuat kita harus mempersiapkan data yang dimiliki sebelum digunakan untuk melakukan permodelan sehingga dapat menyesuaikan dengan karakteristik yang dimiliki oleh tiap model dan mendapatkan hasil yang optimal.

Kita akan melakukan permodelan dengan menggunakan teknik unsupervised clustering. Algoritma yang akan di gunakan adalah **K-Prototypes**. Salah satu faktor utama dalam algoritma ini adalah perlu menggunakan data yang skala antar variabel nya setara. Selain itu kita juga perlu melakukan pengkodean kolom-kolom kategorikal yang di miliki menjadi numerik.

**Standardization**

Tujuannya adalah agar data pada kolom bertipe numerik yang memiliki skala besar tidak mendominasi bagaimana cluster akan di bentuk dan juga tiap kolom akan dianggap sama pentingnya oleh algoritma yang akan digunakan.


```python
# Define numerical columns 
kolom_numerik = ['Umur','NilaiBelanjaSetahun']  
  
# Statistics before standardization 
print('Statistik Sebelum Standardisasi\n')  
print(df[kolom_numerik].describe().round(1))  
  
# Standardization 
df_std = StandardScaler().fit_transform(df[kolom_numerik])  
df_std = pd.DataFrame(data=df_std, index=df.index, columns=df[kolom_numerik].columns)  
  
# Statistics after standardization
print('Statistik hasil standardisasi\n')  
print(df_std.describe().round(0)) 
# Print the result 
df_std.head()
```

    Statistik Sebelum Standardisasi
    
           Umur  NilaiBelanjaSetahun
    count  50.0                 50.0
    mean   37.5            7069874.8
    std    14.7            2590619.0
    min    14.0            2722700.0
    25%    25.0            5257529.8
    50%    35.0            5980077.0
    75%    49.8            9739615.0
    max    64.0           10884508.0
    Statistik hasil standardisasi
    
           Umur  NilaiBelanjaSetahun
    count  50.0                 50.0
    mean    0.0                 -0.0
    std     1.0                  1.0
    min    -2.0                 -2.0
    25%    -1.0                 -1.0
    50%    -0.0                 -0.0
    75%     1.0                  1.0
    max     2.0                  1.0
    




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
      <th>Umur</th>
      <th>NilaiBelanjaSetahun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.411245</td>
      <td>0.946763</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.617768</td>
      <td>-1.695081</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.722833</td>
      <td>-0.695414</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.067039</td>
      <td>-0.727361</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.240944</td>
      <td>1.382421</td>
    </tr>
  </tbody>
</table>
</div>



**Label Encoding**

Merubah data pada kolom bertipe kategorikal menjadi angka dengan menggunakan salah satu fungsi dari **sklearn** yaitu `LabelEncoder`. Sebagai contoh untuk kolom **Jenis Kelamin**, data bernilai "Pria" akan diubah menjadi angka 0 dan "Wanita" akan di rubah menjadi angka 1.


```python
# Define categorical columns  
kolom_kategorikal = ['Jenis Kelamin','Profesi','Tipe Residen']  
  
# Encoding categorical data 
df_encode = df[kolom_kategorikal].copy()   
for col in kolom_kategorikal:  
    df_encode[col] = LabelEncoder().fit_transform(df_encode[col])
      
# Print the result 
df_encode.head()
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
      <th>Jenis Kelamin</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**Joining Dataframe**

Hasil data dari proses **standardization** yaitu `df_std` dan proses **label encoding** yaitu `df_encode` akan digabung menjadi dataframe baru.


```python
# Concatenate dataframe
df_model = pd.concat([df_encode, df_std], axis=1)
 
# Print new rows data
df_model.head()
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
      <th>Jenis Kelamin</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>Umur</th>
      <th>NilaiBelanjaSetahun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1.411245</td>
      <td>0.946763</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>-1.617768</td>
      <td>-1.695081</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.722833</td>
      <td>-0.695414</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1.067039</td>
      <td>-0.727361</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.240944</td>
      <td>1.382421</td>
    </tr>
  </tbody>
</table>
</div>



Hasil diatas adalah preview data setelah dilakukan **preprocessing data.**

## K-Prototypes Clustering

**Elbow Plot**

Salah satu parameter penting yang harus dimasukkan pada algoritma **K-Prototypes** adalah jumlah cluster yang diinginkan. Oleh karena itu, kamu perlu mencari jumlah cluster yang optimal. Salah satu cara untuk mendapatkan nilai optimal tersebut adalah dengan menggunakan bantuan **Elbow Plot**. Caranya dengan memvisualisasikan total jarak seluruh data kita ke pusat cluster nya. Selanjutnya kita memilih titik siku dari pola yang terbentuk dan menjadikannya sebagai jumlah cluster optimal kita.


```python
# Looping cost values using K-Prototypes  
cost = {}  
for k in range(2,10):  
    kproto = KPrototypes (n_clusters = k, random_state=75)  
    kproto.fit_predict(df_model, categorical=[0,1,2])  
    cost[k]= kproto.cost_

# Create elbow plot
sns.set_style('darkgrid')
sns.pointplot(x=list(cost.keys()), y=list(cost.values()))
plt.title('Elbow Plot')
plt.xlabel('Number of Cluster')
plt.ylabel('Cost')
plt.show()
```


    
![png](/assets/img/customer-segmentation/output_32_0.png)
    


Terlihat pada **jumlah cluster = 5** adalah titik siku yang terbentuk dan akan digunakan sebagai jumlah cluster optimal untuk membuat model *clustering* kita. Model yang terbentuk akan disimpan yang nantinya akan digunakan kembali untuk memprediksi data baru.


```python
# Setting n_clusters = 5
kproto = KPrototypes(n_clusters=5, random_state = 75)  
kproto = kproto.fit(df_model, categorical=[0,1,2])  
  
# Save Model  
pickle.dump(kproto, open('data/cluster.pkl', 'wb'))  
```

Selanjutnya melakukan prediksi terhadap data yang sudah dilakukan **preprocessing** kemudian menggabungkan dengan data asli nya.


```python
# Predict cluster
clusters =  kproto.predict(df_model, categorical=[0,1,2])    
print('segmen pelanggan: {}\n'.format(clusters))    
    
# Add cluster data into raw data    
df_final = df.copy()    
df_final['Cluster'] = clusters
df_final.head()
```

    segmen pelanggan: [1 2 4 4 0 3 1 4 3 3 4 4 1 1 0 3 3 4 0 2 0 4 3 0 0 4 0 3 4 4 2 1 2 0 3 0 3
     1 3 2 3 0 3 0 3 0 4 1 3 1]
    
    




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
      <th>Nama Pelanggan</th>
      <th>Jenis Kelamin</th>
      <th>Umur</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>NilaiBelanjaSetahun</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST-001</td>
      <td>Budi Anggara</td>
      <td>Pria</td>
      <td>58</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>9497927</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUST-002</td>
      <td>Shirley Ratuwati</td>
      <td>Wanita</td>
      <td>14</td>
      <td>Pelajar</td>
      <td>Cluster</td>
      <td>2722700</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUST-003</td>
      <td>Agus Cahyono</td>
      <td>Pria</td>
      <td>48</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5286429</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUST-004</td>
      <td>Antonius Winarta</td>
      <td>Pria</td>
      <td>53</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5204498</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUST-005</td>
      <td>Ibu Sri Wahyuni, IR</td>
      <td>Wanita</td>
      <td>41</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>10615206</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Kita bisa melakukan segmentasi data pelanggan berdasarkan kelompok atau *cluster* nya


```python
# Segmentation customer data by cluster 
for i in range(len(df_final['Cluster'].unique())):  
    print('Pelanggan cluster: {}'.format(i))  
    display(df_final[df_final['Cluster']== i])
    print('\n')
```

    Pelanggan cluster: 0
    


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
      <th>Nama Pelanggan</th>
      <th>Jenis Kelamin</th>
      <th>Umur</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>NilaiBelanjaSetahun</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>CUST-005</td>
      <td>Ibu Sri Wahyuni, IR</td>
      <td>Wanita</td>
      <td>41</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>10615206</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CUST-015</td>
      <td>Shirley Ratuwati</td>
      <td>Wanita</td>
      <td>20</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>10365668</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CUST-019</td>
      <td>Mega Pranoto</td>
      <td>Wanita</td>
      <td>32</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>10884508</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CUST-021</td>
      <td>Lestari Fabianto</td>
      <td>Wanita</td>
      <td>38</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>9222070</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>CUST-024</td>
      <td>Putri Ginting</td>
      <td>Wanita</td>
      <td>39</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>10259572</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>CUST-025</td>
      <td>Julia Setiawan</td>
      <td>Wanita</td>
      <td>29</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>10721998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CUST-027</td>
      <td>Grace Mulyati</td>
      <td>Wanita</td>
      <td>35</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>9114159</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>CUST-034</td>
      <td>Deasy Arisandi</td>
      <td>Wanita</td>
      <td>21</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>9759822</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>CUST-036</td>
      <td>Ni Made Suasti</td>
      <td>Wanita</td>
      <td>30</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>9678994</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>CUST-042</td>
      <td>Yuliana Wati</td>
      <td>Wanita</td>
      <td>26</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>9880607</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>CUST-044</td>
      <td>Anna</td>
      <td>Wanita</td>
      <td>18</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>9339737</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>CUST-046</td>
      <td>Elfira Surya</td>
      <td>Wanita</td>
      <td>25</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>10099807</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Pelanggan cluster: 1
    


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
      <th>Nama Pelanggan</th>
      <th>Jenis Kelamin</th>
      <th>Umur</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>NilaiBelanjaSetahun</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST-001</td>
      <td>Budi Anggara</td>
      <td>Pria</td>
      <td>58</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>9497927</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CUST-007</td>
      <td>Cahyono, Agus</td>
      <td>Pria</td>
      <td>64</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>9837260</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CUST-013</td>
      <td>Cahaya Putri</td>
      <td>Wanita</td>
      <td>64</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>9333168</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CUST-014</td>
      <td>Mario Setiawan</td>
      <td>Pria</td>
      <td>60</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>9471615</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>CUST-032</td>
      <td>Chintya Winarni</td>
      <td>Wanita</td>
      <td>47</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>10663179</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>CUST-038</td>
      <td>Agatha Salim</td>
      <td>Wanita</td>
      <td>46</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>10477127</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>CUST-048</td>
      <td>Maria Hutagalung</td>
      <td>Wanita</td>
      <td>45</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>10390732</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>CUST-050</td>
      <td>Lianna Nugraha</td>
      <td>Wanita</td>
      <td>55</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>10569316</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Pelanggan cluster: 2
    


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
      <th>Nama Pelanggan</th>
      <th>Jenis Kelamin</th>
      <th>Umur</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>NilaiBelanjaSetahun</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>CUST-002</td>
      <td>Shirley Ratuwati</td>
      <td>Wanita</td>
      <td>14</td>
      <td>Pelajar</td>
      <td>Cluster</td>
      <td>2722700</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>CUST-020</td>
      <td>Irene Novianto</td>
      <td>Wanita</td>
      <td>16</td>
      <td>Pelajar</td>
      <td>Sector</td>
      <td>2896845</td>
      <td>2</td>
    </tr>
    <tr>
      <th>30</th>
      <td>CUST-031</td>
      <td>Eviana Handry</td>
      <td>Wanita</td>
      <td>19</td>
      <td>Mahasiswa</td>
      <td>Cluster</td>
      <td>3042773</td>
      <td>2</td>
    </tr>
    <tr>
      <th>32</th>
      <td>CUST-033</td>
      <td>Cecilia Kusnadi</td>
      <td>Wanita</td>
      <td>19</td>
      <td>Mahasiswa</td>
      <td>Cluster</td>
      <td>3047926</td>
      <td>2</td>
    </tr>
    <tr>
      <th>39</th>
      <td>CUST-040</td>
      <td>Irene Darmawan</td>
      <td>Wanita</td>
      <td>14</td>
      <td>Pelajar</td>
      <td>Sector</td>
      <td>2861855</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Pelanggan cluster: 3
    


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
      <th>Nama Pelanggan</th>
      <th>Jenis Kelamin</th>
      <th>Umur</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>NilaiBelanjaSetahun</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>CUST-006</td>
      <td>Rosalina Kurnia</td>
      <td>Wanita</td>
      <td>24</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5215541</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CUST-009</td>
      <td>Elisabeth Suryadinata</td>
      <td>Wanita</td>
      <td>29</td>
      <td>Professional</td>
      <td>Sector</td>
      <td>5993218</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CUST-010</td>
      <td>Mario Setiawan</td>
      <td>Pria</td>
      <td>33</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5257448</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CUST-016</td>
      <td>Bambang Rudi</td>
      <td>Pria</td>
      <td>35</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5262521</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CUST-017</td>
      <td>Yuni Sari</td>
      <td>Wanita</td>
      <td>32</td>
      <td>Ibu Rumah Tangga</td>
      <td>Cluster</td>
      <td>5677762</td>
      <td>3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CUST-023</td>
      <td>Denny Amiruddin</td>
      <td>Pria</td>
      <td>34</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5239290</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27</th>
      <td>CUST-028</td>
      <td>Adeline Huang</td>
      <td>Wanita</td>
      <td>40</td>
      <td>Ibu Rumah Tangga</td>
      <td>Cluster</td>
      <td>6631680</td>
      <td>3</td>
    </tr>
    <tr>
      <th>34</th>
      <td>CUST-035</td>
      <td>Ida Ayu</td>
      <td>Wanita</td>
      <td>39</td>
      <td>Professional</td>
      <td>Sector</td>
      <td>5962575</td>
      <td>3</td>
    </tr>
    <tr>
      <th>36</th>
      <td>CUST-037</td>
      <td>Felicia Tandiono</td>
      <td>Wanita</td>
      <td>25</td>
      <td>Professional</td>
      <td>Sector</td>
      <td>5972787</td>
      <td>3</td>
    </tr>
    <tr>
      <th>38</th>
      <td>CUST-039</td>
      <td>Gina Hidayat</td>
      <td>Wanita</td>
      <td>20</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5257775</td>
      <td>3</td>
    </tr>
    <tr>
      <th>40</th>
      <td>CUST-041</td>
      <td>Shinta Aritonang</td>
      <td>Wanita</td>
      <td>24</td>
      <td>Ibu Rumah Tangga</td>
      <td>Cluster</td>
      <td>6820976</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42</th>
      <td>CUST-043</td>
      <td>Yenna Sumadi</td>
      <td>Wanita</td>
      <td>31</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5268410</td>
      <td>3</td>
    </tr>
    <tr>
      <th>44</th>
      <td>CUST-045</td>
      <td>Rismawati Juni</td>
      <td>Wanita</td>
      <td>22</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5211041</td>
      <td>3</td>
    </tr>
    <tr>
      <th>48</th>
      <td>CUST-049</td>
      <td>Josephine Wahab</td>
      <td>Wanita</td>
      <td>33</td>
      <td>Ibu Rumah Tangga</td>
      <td>Sector</td>
      <td>4992585</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Pelanggan cluster: 4
    


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
      <th>Nama Pelanggan</th>
      <th>Jenis Kelamin</th>
      <th>Umur</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>NilaiBelanjaSetahun</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>CUST-003</td>
      <td>Agus Cahyono</td>
      <td>Pria</td>
      <td>48</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5286429</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUST-004</td>
      <td>Antonius Winarta</td>
      <td>Pria</td>
      <td>53</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5204498</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CUST-008</td>
      <td>Danang Santosa</td>
      <td>Pria</td>
      <td>52</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5223569</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CUST-011</td>
      <td>Maria Suryawan</td>
      <td>Wanita</td>
      <td>50</td>
      <td>Professional</td>
      <td>Sector</td>
      <td>5987367</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CUST-012</td>
      <td>Erliana Widjaja</td>
      <td>Wanita</td>
      <td>49</td>
      <td>Professional</td>
      <td>Sector</td>
      <td>5941914</td>
      <td>4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CUST-018</td>
      <td>Nelly Halim</td>
      <td>Wanita</td>
      <td>63</td>
      <td>Ibu Rumah Tangga</td>
      <td>Cluster</td>
      <td>5340690</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CUST-022</td>
      <td>Novita Purba</td>
      <td>Wanita</td>
      <td>52</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5298157</td>
      <td>4</td>
    </tr>
    <tr>
      <th>25</th>
      <td>CUST-026</td>
      <td>Christine Winarto</td>
      <td>Wanita</td>
      <td>55</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5269392</td>
      <td>4</td>
    </tr>
    <tr>
      <th>28</th>
      <td>CUST-029</td>
      <td>Tia Hartanti</td>
      <td>Wanita</td>
      <td>56</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5271845</td>
      <td>4</td>
    </tr>
    <tr>
      <th>29</th>
      <td>CUST-030</td>
      <td>Rosita Saragih</td>
      <td>Wanita</td>
      <td>46</td>
      <td>Ibu Rumah Tangga</td>
      <td>Sector</td>
      <td>5020976</td>
      <td>4</td>
    </tr>
    <tr>
      <th>46</th>
      <td>CUST-047</td>
      <td>Mira Kurnia</td>
      <td>Wanita</td>
      <td>55</td>
      <td>Ibu Rumah Tangga</td>
      <td>Cluster</td>
      <td>6130724</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    

Selain itu kita juga melakukan visualisasi untuk melihat persebaran data hasil *clustering* agar dapat mempermudah melakukan penamaan di tiap-tiap cluster.


```python
# Plotting distribution numerical data by cluster
for i in kolom_numerik:  
    plt.figure(figsize=(6,4))  
    ax = sns.boxplot(x = 'Cluster',y = i, data = df_final)  
    plt.title ('\nBox Plot {}\n'.format(i), fontsize=14, fontweight='bold')  
    plt.show() 
```


    
![png](/assets/img/customer-segmentation/output_40_0.png)
    



    
![png](/assets/img/customer-segmentation/output_40_1.png)
    



```python
# Plotting distribution categorical data by cluster
for i in kolom_kategorikal:  
    plt.figure(figsize=(6,4))  
    ax = sns.countplot(data = df_final, x = 'Cluster', hue = i )  
    plt.title('\nCount Plot {}\n'.format(i), fontsize=14, fontweight='bold')  
    ax.legend(loc="upper center")  
    for p in ax.patches:  
        ax.annotate(format(p.get_height(), '.0f'),  
                    (p.get_x() + p.get_width() / 2., p.get_height()),  
                     ha = 'center',  
                     va = 'center',  
                     xytext = (0, 10),  
                     textcoords = 'offset points')  
      
    sns.despine(right=True,top = True, left = True)  
    ax.axes.yaxis.set_visible(False)  
    plt.show() 
```


    
![png](/assets/img/customer-segmentation/output_41_0.png)
    



    
![png](/assets/img/customer-segmentation/output_41_1.png)
    



    
![png](/assets/img/customer-segmentation/output_41_2.png)
    


Berdasarkan hasil visualisasi tersebut maka data pelanggan akan dikelompokkan sebagai berikut:
- **Cluster 0: Diamond Young Entrepreneur**, isi cluster ini adalah para wiraswasta yang memiliki nilai transaksi rata-rata mendekati 10 juta. Selain itu isi dari cluster ini memiliki umur sekitar 18 - 41 tahun dengan rata-ratanya adalah 29 tahun.
- **Cluster 1: Diamond Senior Entrepreneur**, isi cluster ini adalah para wiraswata yang memiliki nilai transaksi rata-rata mendekati 10 juta. Isi dari cluster ini memiliki umur sekitar 45 - 64 tahun dengan rata-ratanya adalah 55 tahun.
- **Cluster 2: Silver Students**, isi cluster ini adalah para pelajar dan mahasiswa dengan rata-rata umur mereka adalah 16 tahun dan nilai belanja setahun mendekati 3 juta.
- **Cluster 3: Gold Young Member**, isi cluster ini adalah para professional dan ibu rumah tangga yang berusia muda dengan rentang umur sekitar 20 - 40 tahun dan dengan rata-rata 30 tahun dan nilai belanja setahun nya mendekati 6 juta.
- **Cluster 4: Gold Senior Member**, isi cluster ini adalah para professional dan ibu rumah tangga yang berusia tua dengan rentang umur 46 - 63 tahun dan dengan rata-rata 53 tahun dan nilai belanja setahun nya mendekati 6 juta.


```python
# Create segment cluster 
df_final['Segment'] = df_final['Cluster'].map({  
    0: 'Diamond Young Member',  
    1: 'Diamond Senior Member',  
    2: 'Silver Member',  
    3: 'Gold Young Member',  
    4: 'Gold Senior Member'  
})  

df_final.head()
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
      <th>Nama Pelanggan</th>
      <th>Jenis Kelamin</th>
      <th>Umur</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>NilaiBelanjaSetahun</th>
      <th>Cluster</th>
      <th>Segment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST-001</td>
      <td>Budi Anggara</td>
      <td>Pria</td>
      <td>58</td>
      <td>Wiraswasta</td>
      <td>Sector</td>
      <td>9497927</td>
      <td>1</td>
      <td>Diamond Senior Member</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUST-002</td>
      <td>Shirley Ratuwati</td>
      <td>Wanita</td>
      <td>14</td>
      <td>Pelajar</td>
      <td>Cluster</td>
      <td>2722700</td>
      <td>2</td>
      <td>Silver Member</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUST-003</td>
      <td>Agus Cahyono</td>
      <td>Pria</td>
      <td>48</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5286429</td>
      <td>4</td>
      <td>Gold Senior Member</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUST-004</td>
      <td>Antonius Winarta</td>
      <td>Pria</td>
      <td>53</td>
      <td>Professional</td>
      <td>Cluster</td>
      <td>5204498</td>
      <td>4</td>
      <td>Gold Senior Member</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUST-005</td>
      <td>Ibu Sri Wahyuni, IR</td>
      <td>Wanita</td>
      <td>41</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>10615206</td>
      <td>0</td>
      <td>Diamond Young Member</td>
    </tr>
  </tbody>
</table>
</div>



Hasil diatas adalah preview data setelah dilakukan penamaan segmen data pelanggan.

## Predicting New Customer

Berdasarkan model yang telah dibuat sebelumnya kita akan menguji jika terdapat data pelanggan baru yang masuk. Sama seperti sebelumnya data baru akan dilakukan *preprocessing* terlebih dahulu.

**Preprocessing New Data**

Melakukan proses standarisasi untuk kolom numerik dan pengkodean label untuk kolom kategorikal.


```python
# Create new data 
data = [{  
    'Customer_ID': 'CUST-100' ,  
    'Nama Pelanggan': 'Joko' ,  
    'Jenis Kelamin': 'Pria',  
    'Umur': 45,  
    'Profesi': 'Wiraswasta',  
    'Tipe Residen': 'Cluster' ,  
    'NilaiBelanjaSetahun': 8230000  
      
}]  
 
new_data = pd.DataFrame(data)  
new_data
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
      <th>Nama Pelanggan</th>
      <th>Jenis Kelamin</th>
      <th>Umur</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>NilaiBelanjaSetahun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST-100</td>
      <td>Joko</td>
      <td>Pria</td>
      <td>45</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>8230000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create function preprocessing data
def preprocessing_data(new_data):  
    # Encoding categorical data
    kolom_kategorikal = ['Jenis Kelamin', 'Profesi', 'Tipe Residen']
    encode_data = new_data[kolom_kategorikal].copy()  
  
    ## Jenis Kelamin   
    encode_data['Jenis Kelamin'] = encode_data['Jenis Kelamin'].map({  
        'Pria': 0,  
        'Wanita' : 1  
    })        
    ## Profesi  
    encode_data['Profesi'] = encode_data['Profesi'].map({  
        'Ibu Rumah Tangga': 0,  
        'Mahasiswa' : 1,  
        'Pelajar': 2,  
        'Professional': 3,  
        'Wiraswasta': 4  
    })    
    ## Tipe Residen  
    encode_data['Tipe Residen'] = encode_data['Tipe Residen'].map({  
        'Cluster': 0,  
        'Sector' : 1  
    })  
      
    # Standardization numerical data
    kolom_numerik = ['Umur','NilaiBelanjaSetahun']  
    scale_data = new_data[kolom_numerik].copy()  
      
    ## Umur  
    scale_data['Umur'] = (scale_data['Umur'] - 37.5)/14.7        
    ## Nilai Belanja Setahun  
    scale_data['NilaiBelanjaSetahun'] = (scale_data['NilaiBelanjaSetahun'] - 7069874.8)/2590619.0  
      
    # Joining categorical and numerical data
    model_data = pd.concat([encode_data, scale_data], axis=1)
     
    return model_data 
  
# Preprocessing new data 
fix_data = preprocessing_data(new_data)  
fix_data
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
      <th>Jenis Kelamin</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>Umur</th>
      <th>NilaiBelanjaSetahun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0.510204</td>
      <td>0.447818</td>
    </tr>
  </tbody>
</table>
</div>



Setelah melakukan *preprocessing* terhadap data baru, maka selanjutnya memprediksi data tersebut termasuk dalam cluster mana atau segmen apa.


```python
# Function clustering new data
def clustering_data(new_data):       
    # Open Model  
    model = pickle.load(open('data/cluster.pkl', 'rb'))        
    # Predict new data
    pred = model.predict(new_data, categorical=[0,1,2])  
      
    return pred  
```


```python
# Function segmentation new data
def segmentation(new_data, pred_cluster):     
    # Add cluster to new column
    new_data['Cluster'] = pred_cluster
      
    # Choosing Segment
    new_data['Segment'] = new_data['Cluster'].map({  
        0: 'Diamond Young Member',  
        1: 'Diamond Senior Member',  
        2: 'Silver Students',  
        3: 'Gold Young Member',  
        4: 'Gold Senior Member'  
    })  
      
    return new_data
```


```python
# Predicting new customer
pred_cluster = clustering_data(fix_data) 
result = segmentation(new_data, pred_cluster)  
result
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
      <th>Nama Pelanggan</th>
      <th>Jenis Kelamin</th>
      <th>Umur</th>
      <th>Profesi</th>
      <th>Tipe Residen</th>
      <th>NilaiBelanjaSetahun</th>
      <th>Cluster</th>
      <th>Segment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST-100</td>
      <td>Joko</td>
      <td>Pria</td>
      <td>45</td>
      <td>Wiraswasta</td>
      <td>Cluster</td>
      <td>8230000</td>
      <td>1</td>
      <td>Diamond Senior Member</td>
    </tr>
  </tbody>
</table>
</div>



Hasil diatas adalah prediksi cluster atau segmen data pelanggan baru.
