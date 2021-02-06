---
layout: post
title: Market Basket Analysis in Python
color: brown
thumbnail: "assets/img/thumbnails/market-basket-analysis.jpg"
tags: [Data Science, Apriori]
---

**DQLab Fashion** adalah sebuah toko fashion yang menjual berbagai produk seperti jeans, kemeja, kosmetik, dan lain-lain. Walaupun cukup berkembang, namun dengan semakin banyaknya kompetitor dan banyak produk yang stoknya masih banyak tentunya membuat khawatir Pak Agus, manajer **DQLab Fashion**. Salah satu solusi adalah membuat paket yang inovatif. Dimana produk yang sebelumnya tidak terlalu laku tapi punya pangsa pasar malah bisa dipaketkan dan laku.

**Tujuan** :
* Mendapatkan insight top 10 dan bottom 10 dari produk yang terjual.
* Mendapatkan daftar seluruh kombinasi paket produk dengan korelasi yang kuat.
* Mendapatkan daftar seluruh kombinasi paket produk dengan item tertentu.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid')

import warnings
warnings.filterwarnings('ignore')
```


```python
# import dataset
fashion = pd.read_csv('transaksi_dqlab_retail.tsv', sep='\t')
# print first five rows
fashion.head()
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
      <th>Kode Transaksi</th>
      <th>Nama Barang</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1</td>
      <td>Kaos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#1</td>
      <td>Shampo Biasa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#1</td>
      <td>Sepatu Sport merk Z</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#1</td>
      <td>Serum Vitamin</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#1</td>
      <td>Baju Renang Pria Dewasa</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check dimension
print(f"DQLab Fashion Data terdiri dari {fashion.shape[0]} baris dan {fashion.shape[1]} kolom")
```

    DQLab Fashion Data terdiri dari 33668 baris dan 2 kolom
    


```python
# chcek info data
fashion.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 33668 entries, 0 to 33667
    Data columns (total 2 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   Kode Transaksi  33668 non-null  object
     1   Nama Barang     33668 non-null  object
    dtypes: object(2)
    memory usage: 526.2+ KB
    

Tidak ditemukan *missing values* dan format data sudah sesuai.


```python
print("Jumlah transaksi unik adalah", fashion['Kode Transaksi'].nunique())
```

    Jumlah transaksi unik adalah 3450
    


```python
# cek jumlah produk dari transaksi
print("Jumlah produk unik adalah", fashion['Nama Barang'].nunique())
```

    Jumlah produk unik adalah 69
    


```python
print("Berikut ini adalah list produk yang telah terjual :")
fashion['Nama Barang'].unique()
```

    Berikut ini adalah list produk yang telah terjual :
    
    array(['Kaos', 'Shampo Biasa', 'Sepatu Sport merk Z', 'Serum Vitamin',
           'Baju Renang Pria Dewasa', 'Baju Renang Wanita Dewasa',
           'Baju Kaos Olahraga', 'Celana Jogger Casual', 'Dompet Card Holder',
           'Celana Jeans Sobek Wanita', 'Blouse Denim', 'Baju Batik Wanita',
           'Hair and Scalp', 'Minyak Rambut', 'Wedges Hitam',
           'Sepatu Sandal Anak', 'Tas Sekolah Anak Perempuan',
           'Baju Kemeja Putih', 'Dompet Flip Cover', 'Hair Tonic',
           'Baju Kaos Anak - Superheroes', 'Celana Pendek Casual',
           'Jeans Jumbo', 'Celana Pendek Jeans', 'Sepatu Sekolah Hitam W',
           'Tas Ransel Mini', 'Dompet Kulit Pria', 'Hair Dryer',
           'Flat Shoes Ballerina', 'Tas Sekolah Anak Laki-laki',
           'Cover Koper', 'Gembok Koper', 'Sweater Top Panjang', 'Tank Top',
           'Atasan Kaos Putih', 'Atasan Baju Belang', 'Shampo Anti Dandruff',
           'Tas Pinggang Wanita', 'Koper Fiber', 'Sepatu Sport merk Y',
           'Dompet STNK Gantungan', 'Cream Whitening', 'Celana Tactical ',
           'Woman Ripped Jeans ', 'Tas Travel', 'Tali Pinggang Anak',
           'Dompet Unisex', 'Tali Pinggang Gesper Pria', 'Tas Waist Bag',
           'Tas Tangan', 'Celana Pendek Green/Hijau', 'Obat Penumbuh Rambut',
           'Pelembab', 'Baju Kaos Anak - Karakter Kartun',
           'Sepatu Kulit Casual', 'Tas Makeup', 'Baju Renang Anak Perempuan',
           'Kuas Makeup ', 'Sunblock Cream', 'Tas Kulit Selempang',
           'Tas Multifungsi', 'Hair Dye', 'Celana Panjang Format Hitam',
           'Celana Jeans Sobek Pria', 'Mascara', 'Tali Ban Ikat Pinggang',
           'Stripe Pants', 'Tas Kosmetik', 'Baju Renang Pria Anak-anak'],
          dtype=object)



**Statistik Top 10**


```python
top_10 = fashion['Nama Barang'].value_counts()[:10]
plt.figure(figsize=(10,5))
ax1 = sns.barplot(x=top_10, y=top_10.index, palette='rocket')
for p1 in ax1.patches:
    ax1.annotate('{:.2f}'.format(p1.get_width()/fashion['Kode Transaksi'].nunique()*100), 
                 xy=(p1.get_width()+0.5, p1.get_y()+0.4),
                 ha='left', va='center', color= 'black')
ax1.set_title('Statistik Top 10 Produk', fontweight='bold', fontsize=15)
ax1.set_xlabel('Jumlah Transaksi')
plt.show()
```


    
![png](/assets/img/fashion-analysis/output_12_0.png)
    


**Statistik Bottom 10 Produk**


```python
bottom_10 = fashion['Nama Barang'].value_counts()[-10:]
plt.figure(figsize=(10,5))
ax2 = sns.barplot(x=bottom_10, y=bottom_10.index, palette='rocket')
for p2 in ax2.patches:
    ax2.annotate('{:.2f}'.format(p2.get_width()/fashion['Kode Transaksi'].nunique()*100), 
                 xy=(p2.get_width()+0.5, p2.get_y()+0.4),
                 ha='left', va='center', color= 'black')
ax2.set_title('Statistik Bottom 10 Produk', fontweight='bold', fontsize=15)
ax2.set_xlabel('Jumlah Transaksi')
plt.show()
```


    
![png](/assets/img/fashion-analysis/output_14_0.png)
    


Insights :
* **Shampo Biasa** telah terjual sekitar 60,14% dari seluruh transaksi.
* Hanya 0,26% **Celana Jeans Sobek Pria** yang terdapat pada keseluruhan transaksi.


```python
# grouping berdasarkan kode transaksi
data = fashion.groupby('Kode Transaksi')['Nama Barang'].unique()
data.head()
```




    Kode Transaksi
    #1       [Kaos, Shampo Biasa, Sepatu Sport merk Z, Seru...
    #10      [Jeans Jumbo, Celana Pendek Jeans, Kaos, Baju ...
    #100     [Hair Dryer, Shampo Biasa, Hair Tonic, Sepatu ...
    #1000    [Celana Jeans Sobek Wanita, Celana Pendek Jean...
    #1001    [Hair Dryer, Wedges Hitam, Sepatu Sport merk Z...
    Name: Nama Barang, dtype: object




```python
# cek total transaksi
list_transaksi = data.tolist()
len(list_transaksi)
```




    3450




```python
# Hitung banyak produk per transaksi dari semua transaksi
counts = [len(transaksi) for transaksi in list_transaksi]

# Hitung nilai tengah jumlah produk pada sebuah transaksi
print("Median number of items in a transaction is", int(np.median(counts)))

# Hitung nilai maksimum jumlah produk pada sebuah transaksi
print("Maximum number of items in a transaction is", np.max(counts))
```

    Median number of items in a transaction is 10
    Maximum number of items in a transaction is 22
    

## Encoding Transactions


```python
# encoding produk pada sebuah list transaksi
from mlxtend.preprocessing import TransactionEncoder

encoder = TransactionEncoder()
data_encode = encoder.fit(list_transaksi).transform(list_transaksi)
df_encode = pd.DataFrame(data_encode, columns=encoder.columns_)
df_encode
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
      <th>Atasan Baju Belang</th>
      <th>Atasan Kaos Putih</th>
      <th>Baju Batik Wanita</th>
      <th>Baju Kaos Anak - Karakter Kartun</th>
      <th>Baju Kaos Anak - Superheroes</th>
      <th>Baju Kaos Olahraga</th>
      <th>Baju Kemeja Putih</th>
      <th>Baju Renang Anak Perempuan</th>
      <th>Baju Renang Pria Anak-anak</th>
      <th>Baju Renang Pria Dewasa</th>
      <th>...</th>
      <th>Tas Multifungsi</th>
      <th>Tas Pinggang Wanita</th>
      <th>Tas Ransel Mini</th>
      <th>Tas Sekolah Anak Laki-laki</th>
      <th>Tas Sekolah Anak Perempuan</th>
      <th>Tas Tangan</th>
      <th>Tas Travel</th>
      <th>Tas Waist Bag</th>
      <th>Wedges Hitam</th>
      <th>Woman Ripped Jeans</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3445</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3446</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3447</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3448</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3449</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>3450 rows × 69 columns</p>
</div>
<br>



```python
# Hitung support metric pada tiap kolom
df_encode.mean(axis=0)
```




    Atasan Baju Belang                  0.016232
    Atasan Kaos Putih                   0.118841
    Baju Batik Wanita                   0.380290
    Baju Kaos Anak - Karakter Kartun    0.121159
    Baju Kaos Anak - Superheroes        0.115942
                                          ...   
    Tas Tangan                          0.037101
    Tas Travel                          0.220870
    Tas Waist Bag                       0.161159
    Wedges Hitam                        0.246087
    Woman Ripped Jeans                  0.034203
    Length: 69, dtype: float64




```python
# Plotting distribution of product counts
distribusi = df_encode.sum(axis=1).value_counts().sort_index()
plt.figure(figsize=(10,5))
ax3 = sns.barplot(x=distribusi.index, y=distribusi, palette='Set2')
for p3 in ax3.patches:
    ax3.annotate('{:.0f}'.format(p3.get_height()), (p3.get_x()+0.4, p3.get_height()),
                 ha='center', va='bottom', color= 'black')
ax3.set_title('Distribusi Jumlah Produk per Transaksi', fontweight='bold', fontsize=15)
ax3.set_xlabel('Jumlah Produk')
ax3.set_ylabel('Jumlah Transaksi')
plt.show()
```


    
![png](/assets/img/fashion-analysis/output_22_0.png)
    


**Kombinasi Produk Menarik** :
* Memiliki asosiasi atau hubungan erat.
* Kombinasi produk minimal 2 item, dan maksimum 3 item.
* Kombinasi produk itu muncul setidaknya 10 dari dari seluruh transaksi.
* Memiliki tingkat confidence minimal 50 persen.


```python
# setting transaksi berdasarkan kriteria menarik
df_transaksi = df_encode[df_encode.sum(axis=1) >= 2]
kriteria = 10 / len(list_transaksi)
kriteria
```




    0.002898550724637681



## Apriori Algorithm


```python
from mlxtend.frequent_patterns import apriori

frekuensi_itemset = apriori(df_transaksi, min_support=kriteria, max_len=3, use_colnames=True)
frekuensi_itemset
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
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.016274</td>
      <td>(Atasan Baju Belang)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.119151</td>
      <td>(Atasan Kaos Putih)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.380703</td>
      <td>(Baju Batik Wanita)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.121476</td>
      <td>(Baju Kaos Anak - Karakter Kartun)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.116245</td>
      <td>(Baju Kaos Anak - Superheroes)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16214</th>
      <td>0.006103</td>
      <td>(Tas Waist Bag, Tas Travel, Tas Sekolah Anak L...</td>
    </tr>
    <tr>
      <th>16215</th>
      <td>0.008718</td>
      <td>(Wedges Hitam, Tas Travel, Tas Sekolah Anak La...</td>
    </tr>
    <tr>
      <th>16216</th>
      <td>0.009300</td>
      <td>(Wedges Hitam, Tas Waist Bag, Tas Sekolah Anak...</td>
    </tr>
    <tr>
      <th>16217</th>
      <td>0.012206</td>
      <td>(Tas Waist Bag, Wedges Hitam, Tas Travel)</td>
    </tr>
    <tr>
      <th>16218</th>
      <td>0.003487</td>
      <td>(Woman Ripped Jeans , Wedges Hitam, Tas Travel)</td>
    </tr>
  </tbody>
</table>
<p>16219 rows × 2 columns</p>
</div>



## Association Rules


```python
from mlxtend.frequent_patterns import association_rules

# Recover association rules using metric lift
rules = association_rules(frekuensi_itemset, metric = 'lift')
rules
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Baju Batik Wanita)</td>
      <td>(Atasan Baju Belang)</td>
      <td>0.380703</td>
      <td>0.016274</td>
      <td>0.005522</td>
      <td>0.014504</td>
      <td>0.891208</td>
      <td>-0.000674</td>
      <td>0.998203</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Atasan Baju Belang)</td>
      <td>(Baju Batik Wanita)</td>
      <td>0.016274</td>
      <td>0.380703</td>
      <td>0.005522</td>
      <td>0.339286</td>
      <td>0.891208</td>
      <td>-0.000674</td>
      <td>0.937314</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Atasan Baju Belang)</td>
      <td>(Baju Kemeja Putih)</td>
      <td>0.016274</td>
      <td>0.364720</td>
      <td>0.004940</td>
      <td>0.303571</td>
      <td>0.832342</td>
      <td>-0.000995</td>
      <td>0.912198</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Baju Kemeja Putih)</td>
      <td>(Atasan Baju Belang)</td>
      <td>0.364720</td>
      <td>0.016274</td>
      <td>0.004940</td>
      <td>0.013546</td>
      <td>0.832342</td>
      <td>-0.000995</td>
      <td>0.997234</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Blouse Denim)</td>
      <td>(Atasan Baju Belang)</td>
      <td>0.182796</td>
      <td>0.016274</td>
      <td>0.003487</td>
      <td>0.019078</td>
      <td>1.172269</td>
      <td>0.000512</td>
      <td>1.002858</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>75439</th>
      <td>(Woman Ripped Jeans , Tas Travel)</td>
      <td>(Wedges Hitam)</td>
      <td>0.006684</td>
      <td>0.246731</td>
      <td>0.003487</td>
      <td>0.521739</td>
      <td>2.114611</td>
      <td>0.001838</td>
      <td>1.575018</td>
    </tr>
    <tr>
      <th>75440</th>
      <td>(Wedges Hitam, Tas Travel)</td>
      <td>(Woman Ripped Jeans )</td>
      <td>0.079628</td>
      <td>0.034292</td>
      <td>0.003487</td>
      <td>0.043796</td>
      <td>1.277125</td>
      <td>0.000757</td>
      <td>1.009939</td>
    </tr>
    <tr>
      <th>75441</th>
      <td>(Woman Ripped Jeans )</td>
      <td>(Wedges Hitam, Tas Travel)</td>
      <td>0.034292</td>
      <td>0.079628</td>
      <td>0.003487</td>
      <td>0.101695</td>
      <td>1.277125</td>
      <td>0.000757</td>
      <td>1.024565</td>
    </tr>
    <tr>
      <th>75442</th>
      <td>(Wedges Hitam)</td>
      <td>(Woman Ripped Jeans , Tas Travel)</td>
      <td>0.246731</td>
      <td>0.006684</td>
      <td>0.003487</td>
      <td>0.014134</td>
      <td>2.114611</td>
      <td>0.001838</td>
      <td>1.007557</td>
    </tr>
    <tr>
      <th>75443</th>
      <td>(Tas Travel)</td>
      <td>(Woman Ripped Jeans , Wedges Hitam)</td>
      <td>0.221447</td>
      <td>0.014821</td>
      <td>0.003487</td>
      <td>0.015748</td>
      <td>1.062529</td>
      <td>0.000205</td>
      <td>1.000942</td>
    </tr>
  </tbody>
</table>
<p>75444 rows × 9 columns</p>
</div>




```python
# pilih rules dengan nilai confindence 50% dan positif leverage
result = rules[(rules['confidence'] >= 0.5) & (rules['leverage'] > 0)] \
                .sort_values(by='lift', ascending=False)[:10]
result.reset_index(drop=True, inplace=True)
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Tas Makeup, Tas Pinggang Wanita)</td>
      <td>(Baju Renang Anak Perempuan)</td>
      <td>0.011915</td>
      <td>0.036036</td>
      <td>0.010462</td>
      <td>0.878049</td>
      <td>24.365854</td>
      <td>0.010033</td>
      <td>7.904505</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Tas Makeup, Tas Travel)</td>
      <td>(Baju Renang Anak Perempuan)</td>
      <td>0.012496</td>
      <td>0.036036</td>
      <td>0.010171</td>
      <td>0.813953</td>
      <td>22.587209</td>
      <td>0.009721</td>
      <td>5.181306</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Tas Makeup, Tas Ransel Mini)</td>
      <td>(Baju Renang Anak Perempuan)</td>
      <td>0.015402</td>
      <td>0.036036</td>
      <td>0.011334</td>
      <td>0.735849</td>
      <td>20.419811</td>
      <td>0.010779</td>
      <td>3.649292</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Tas Pinggang Wanita, Sunblock Cream)</td>
      <td>(Kuas Makeup )</td>
      <td>0.023540</td>
      <td>0.034292</td>
      <td>0.016274</td>
      <td>0.691358</td>
      <td>20.160703</td>
      <td>0.015467</td>
      <td>3.128893</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Tas Pinggang Wanita, Baju Renang Anak Perempuan)</td>
      <td>(Tas Makeup)</td>
      <td>0.013078</td>
      <td>0.040976</td>
      <td>0.010462</td>
      <td>0.800000</td>
      <td>19.523404</td>
      <td>0.009926</td>
      <td>4.795118</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(Tas Ransel Mini, Baju Renang Anak Perempuan)</td>
      <td>(Tas Makeup)</td>
      <td>0.014240</td>
      <td>0.040976</td>
      <td>0.011334</td>
      <td>0.795918</td>
      <td>19.423795</td>
      <td>0.010750</td>
      <td>4.699215</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(Baju Renang Anak Perempuan, Celana Pendek Gre...</td>
      <td>(Tas Makeup)</td>
      <td>0.013078</td>
      <td>0.040976</td>
      <td>0.010171</td>
      <td>0.777778</td>
      <td>18.981087</td>
      <td>0.009636</td>
      <td>4.315606</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(Tas Makeup, Tas Waist Bag)</td>
      <td>(Baju Renang Anak Perempuan)</td>
      <td>0.006393</td>
      <td>0.036036</td>
      <td>0.004359</td>
      <td>0.681818</td>
      <td>18.920455</td>
      <td>0.004129</td>
      <td>3.029601</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(Tas Makeup, Celana Pendek Green/Hijau)</td>
      <td>(Baju Renang Anak Perempuan)</td>
      <td>0.015112</td>
      <td>0.036036</td>
      <td>0.010171</td>
      <td>0.673077</td>
      <td>18.677885</td>
      <td>0.009627</td>
      <td>2.948596</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(Sunblock Cream, Dompet Flip Cover)</td>
      <td>(Kuas Makeup )</td>
      <td>0.025865</td>
      <td>0.034292</td>
      <td>0.016274</td>
      <td>0.629213</td>
      <td>18.348505</td>
      <td>0.015387</td>
      <td>2.604484</td>
    </tr>
  </tbody>
</table>
</div>



Insights :

- Kombinasi **Tas Makeup dan Tas Pinggang Wanita terhadap Baju Renang Anak Perempuan** menghasilkan asosiasi atau hubungan yang paling erat dimana nilai confidence cukup tinggi diikuti metric lift.

## Slow Moving Product

Slow Moving Product adalah produk yang memiliki tingkat penjualan relatif lama karena bisa dipengaruhi musiman.


```python
# Item slow moving berdasarkan top 3 rules
tas_makeup = rules[rules['consequents'] == {'Tas Makeup'}] \
             .sort_values(by='lift', ascending=False)[:3] 
baju_renang_pria = rules[rules['consequents'] == {'Baju Renang Pria Anak-anak'}] \
             .sort_values(by='lift', ascending=False)[:3]
```


```python
slow_moving = pd.concat([tas_makeup, baju_renang_pria])
slow_moving
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26581</th>
      <td>(Tas Pinggang Wanita, Baju Renang Anak Perempuan)</td>
      <td>(Tas Makeup)</td>
      <td>0.013078</td>
      <td>0.040976</td>
      <td>0.010462</td>
      <td>0.800000</td>
      <td>19.523404</td>
      <td>0.009926</td>
      <td>4.795118</td>
    </tr>
    <tr>
      <th>26587</th>
      <td>(Tas Ransel Mini, Baju Renang Anak Perempuan)</td>
      <td>(Tas Makeup)</td>
      <td>0.014240</td>
      <td>0.040976</td>
      <td>0.011334</td>
      <td>0.795918</td>
      <td>19.423795</td>
      <td>0.010750</td>
      <td>4.699215</td>
    </tr>
    <tr>
      <th>26158</th>
      <td>(Baju Renang Anak Perempuan, Celana Pendek Gre...</td>
      <td>(Tas Makeup)</td>
      <td>0.013078</td>
      <td>0.040976</td>
      <td>0.010171</td>
      <td>0.777778</td>
      <td>18.981087</td>
      <td>0.009636</td>
      <td>4.315606</td>
    </tr>
    <tr>
      <th>26714</th>
      <td>(Gembok Koper, Tas Waist Bag)</td>
      <td>(Baju Renang Pria Anak-anak)</td>
      <td>0.014821</td>
      <td>0.009300</td>
      <td>0.004069</td>
      <td>0.274510</td>
      <td>29.518382</td>
      <td>0.003931</td>
      <td>1.365560</td>
    </tr>
    <tr>
      <th>26690</th>
      <td>(Flat Shoes Ballerina, Gembok Koper)</td>
      <td>(Baju Renang Pria Anak-anak)</td>
      <td>0.021796</td>
      <td>0.009300</td>
      <td>0.004069</td>
      <td>0.186667</td>
      <td>20.072500</td>
      <td>0.003866</td>
      <td>1.218074</td>
    </tr>
    <tr>
      <th>26655</th>
      <td>(Jeans Jumbo, Celana Jeans Sobek Wanita)</td>
      <td>(Baju Renang Pria Anak-anak)</td>
      <td>0.045626</td>
      <td>0.009300</td>
      <td>0.005522</td>
      <td>0.121019</td>
      <td>13.013336</td>
      <td>0.005097</td>
      <td>1.127101</td>
    </tr>
  </tbody>
</table>
</div>



Insights :
* Terlihat pada item yang dipasangkan dengan **Tas Makeup** menghasilkan asosiasi atau hubungan yang cukup erat dimana nilai confidence yang tinggi diikuti metric lift. Selain itu dari ketiga rules yang terbentuk masing-masing terdapat produk **Baju Renang Anak Perempuan**. 
* Sedangkan item yang dipasangkan dengan **Baju Renang Pria Anak-anak** menghasilkan asosiasi atau hubungan yang tidak begitu erat dilihat dari nilai confidence yang rendah walaupun metric lift lebih tinggi.

## Kesimpulan
* Produk **Tas Makeup, Tas Pinggang Wanita, dan Baju Renang Anak Perempuan** dapat diletakkan di area atau rak yang berdekatan agar memudahkan pembeli mengambilnya.
* Produk **Tas Makeup, Tas Pinggang Wanita, dan Baju Renang Anak Perempuan** dapat dibuat promo diskon agar pembeli lebih tertarik membelinya.


```python

```
