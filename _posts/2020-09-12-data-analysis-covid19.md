---
layout: post
title: Data Analysis Covid-19
color: darkred
thumbnail: "assets/img/thumbnails/covid19.jpg"
tags: [Covid-19, Data Analysis]
---


**Covid-19** adalah pandemi yang sudah mewabah ke seluruh dunia. Sebagian besar negara-negara di dunia sudah terjangkit. Penanganan tiap-tiap negara pun berbeda, sesuai dengan kebijakan pemerintah. Hal ini mengakibatkan perbedaan trend kenaikan atau penurunan kasus covid berbeda-beda tiap negara.

Pada kesempatan kali ini, kita akan membahas salah satu project dari [DQLab](https://dqlab.id) mengenai **Analisis Data Covid-19 di Dunia dan ASEAN**.
Dataset yang digunakan adalah data covid-19 dari salah satu open API yang tersedia yaitu [https://covid19-api.org/](https://covid19-api.org/).

# Import Library
Library yang akan digunakan dalam kasus ini, yaitu
* *requests* untuk menunjuk API url yang akan diakses
* *json* untuk mengambil data dengan extensi json
* *numpy* untuk perhitungan numerik
* *pandas* untuk membuat dataframe dan manipulasi data
* *matplotlib* untuk memvisualisasi data dengan grafik




```python
# Import library yang digunakan
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
```

# Mengakses Data


## Membuat Fungsi get API
Tahap pertama adalah membuat fungsi python untuk mengambil data json pada url yang ditunjuk/dipilih. Dimana data sukses diambil dengan **status code 200**. Jika tidak artinya data gagal untuk diambil.


```python
# Fungsi untuk mengambil data dari API
def get_json(api_url):
	response = requests.get(api_url)
	if response.status_code == 200:
		return json.loads(response.content.decode('utf-8'))
	else:
		return None
```

## Memanggil API Covid19
Tahap kedua adalah mengambil data covid-19. Rekapitulasi data COVID-19 global berada di https://covid19-api.org/. Lalu kita akan mengambil data covid19 di seluruh negara pada tanggal tertentu. Untuk kasus ini, record data yang diambil adalah **'2020-09-12'**, kemudian masukkan hasil respon api ke dalam variabel.

Untuk mendapatkan dataframe covid-19, kita gunakan fungsi bawaan dari pandas yaitu **pd.io.json.json_normalize** dan panggil *function* yang sudah dibuat sebelumnya, yaitu **get_json()**. Lalu kita bisa menampilkan beberapa data awal sebagai sampel.


```python
# Mengambil record data covid-19 pada tanggal tertentu
record_date = '2020-09-12'
covid_url = 'https://covid19-api.org/api/status?date='+record_date

# Membuat dataframe covid-19
df_covid_worldwide = pd.io.json.json_normalize(get_json(covid_url))
df_covid_worldwide.head()
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
      <th>country</th>
      <th>last_update</th>
      <th>cases</th>
      <th>deaths</th>
      <th>recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US</td>
      <td>2020-09-11T23:32:03</td>
      <td>6440541</td>
      <td>192886</td>
      <td>2417878</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IN</td>
      <td>2020-09-11T23:32:03</td>
      <td>4562414</td>
      <td>76271</td>
      <td>3542663</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BR</td>
      <td>2020-09-11T23:32:03</td>
      <td>4238446</td>
      <td>129522</td>
      <td>3683047</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RU</td>
      <td>2020-09-11T23:32:03</td>
      <td>1048257</td>
      <td>18309</td>
      <td>865646</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PE</td>
      <td>2020-09-11T23:32:03</td>
      <td>710067</td>
      <td>30344</td>
      <td>544745</td>
    </tr>
  </tbody>
</table>
</div>



## Merubah Format date
Terlihat data kolom **last_update** belum menggunakan format yang sesuai. Untuk itu kita akan ubah menjadi format nya menggunakan fungsi **pd.to_datetime** dengan format **YYYY-mm-dd HH:MM:SS**. Kemudian ubah bentuk *datetime* untuk mengambil *date only* dengan fungsi date() setiap baris nya.


```python
# Merubah format data tanggal
df_covid_worldwide['last_update'] = pd.to_datetime(df_covid_worldwide['last_update'], 
                                                   format='%Y-%m-%d %H:%M:%S')
df_covid_worldwide['last_update'] = df_covid_worldwide['last_update'].apply(lambda x: x.date())
```


```python
# Tampilan data setelah diformat
df_covid_worldwide.head()
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
      <th>country</th>
      <th>last_update</th>
      <th>cases</th>
      <th>deaths</th>
      <th>recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US</td>
      <td>2020-09-11</td>
      <td>6440541</td>
      <td>192886</td>
      <td>2417878</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IN</td>
      <td>2020-09-11</td>
      <td>4562414</td>
      <td>76271</td>
      <td>3542663</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BR</td>
      <td>2020-09-11</td>
      <td>4238446</td>
      <td>129522</td>
      <td>3683047</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RU</td>
      <td>2020-09-11</td>
      <td>1048257</td>
      <td>18309</td>
      <td>865646</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PE</td>
      <td>2020-09-11</td>
      <td>710067</td>
      <td>30344</td>
      <td>544745</td>
    </tr>
  </tbody>
</table>
</div>



## Mengambil Data Countries
Tahap berikutnya adalah mengambil data countries. Data tersebut berasal dari url https://covid19-api.org/api/countries. Lalu kita akan ambil hanya kolom **nama negara dan kode negara** saja untuk dijadikan dataframe.


```python
# Mengambil data countries dari API
countries_url = 'https://covid19-api.org/api/countries'
df_countries = pd.io.json.json_normalize(get_json(countries_url))

# Ambil kolom yang digunakan
df_countries = df_countries.rename(columns={'alpha2': 'country'})[['name','country']]

df_countries.head()
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
      <th>name</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>AF</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>DZ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>AD</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>AO</td>
    </tr>
  </tbody>
</table>
</div>



# Analisis COVID-19 di Dunia

## Merge Covid19 Data dan Countries
Pertama kita akan gabungkan 2 *dataframe* yang telah dibuat sebelumnya yaitu **df_covid_worldwide** dan **df_countries** menggunakan fungsi **merge** berdasarkan kode negara atau kolom **country**. Lalu tampilkan 5 data teratas hasil *merge dataframe*.


```python
# Merge dataframe
df_covid_denormalized = pd.merge(df_covid_worldwide, df_countries, on='country')

# Menampilkan 5 data teratas
df_covid_denormalized.head()
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
      <th>country</th>
      <th>last_update</th>
      <th>cases</th>
      <th>deaths</th>
      <th>recovered</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US</td>
      <td>2020-09-11</td>
      <td>6440541</td>
      <td>192886</td>
      <td>2417878</td>
      <td>United States of America</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IN</td>
      <td>2020-09-11</td>
      <td>4562414</td>
      <td>76271</td>
      <td>3542663</td>
      <td>India</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BR</td>
      <td>2020-09-11</td>
      <td>4238446</td>
      <td>129522</td>
      <td>3683047</td>
      <td>Brazil</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RU</td>
      <td>2020-09-11</td>
      <td>1048257</td>
      <td>18309</td>
      <td>865646</td>
      <td>Russian Federation</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PE</td>
      <td>2020-09-11</td>
      <td>710067</td>
      <td>30344</td>
      <td>544745</td>
      <td>Peru</td>
    </tr>
  </tbody>
</table>
</div>



Terlihat negara **United States of Amerika** memiliki kasus covid-19 terbanyak yaitu 6.440.541 kasus pada update terkini.

## Menghitung Fatality Ratio
Selanjutnya kita akan mencari rasio tingkat kematian dari total kasus yang terjadi pada tiap negara. Caranya adalah menambahkan kolom baru bernama **fatality_ratio**, yang merupakan pembagian antara kolom **deaths** dan **cases**.


```python
# Mencari fatality ratio tiap negara
df_covid_denormalized['fatality_ratio'] = df_covid_denormalized['deaths']/df_covid_denormalized['cases']

df_covid_denormalized.head()
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
      <th>country</th>
      <th>last_update</th>
      <th>cases</th>
      <th>deaths</th>
      <th>recovered</th>
      <th>name</th>
      <th>fatality_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US</td>
      <td>2020-09-11</td>
      <td>6440541</td>
      <td>192886</td>
      <td>2417878</td>
      <td>United States of America</td>
      <td>0.029949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IN</td>
      <td>2020-09-11</td>
      <td>4562414</td>
      <td>76271</td>
      <td>3542663</td>
      <td>India</td>
      <td>0.016717</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BR</td>
      <td>2020-09-11</td>
      <td>4238446</td>
      <td>129522</td>
      <td>3683047</td>
      <td>Brazil</td>
      <td>0.030559</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RU</td>
      <td>2020-09-11</td>
      <td>1048257</td>
      <td>18309</td>
      <td>865646</td>
      <td>Russian Federation</td>
      <td>0.017466</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PE</td>
      <td>2020-09-11</td>
      <td>710067</td>
      <td>30344</td>
      <td>544745</td>
      <td>Peru</td>
      <td>0.042734</td>
    </tr>
  </tbody>
</table>
</div>



## Top 20 Negara dengan Fatality Ratio Tertinggi
Kita akan ambil **20 negara teratas** dengan kolom fatality_ratio tertinggi dengan mengurutkan data secara descending menggunakan fungsi **sort_values**.


```python
# Menampilkan 20 Negara dengan fatality ratio tertinggi
df_fatality_rate = df_covid_denormalized.sort_values(by='fatality_ratio', ascending=False)

df_top_20_fatality_rate = df_fatality_rate.head(20)
df_top_20_fatality_rate
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
      <th>country</th>
      <th>last_update</th>
      <th>cases</th>
      <th>deaths</th>
      <th>recovered</th>
      <th>name</th>
      <th>fatality_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>143</th>
      <td>YE</td>
      <td>2020-09-11</td>
      <td>2007</td>
      <td>582</td>
      <td>1211</td>
      <td>Yemen</td>
      <td>0.289985</td>
    </tr>
    <tr>
      <th>18</th>
      <td>IT</td>
      <td>2020-09-11</td>
      <td>284796</td>
      <td>35597</td>
      <td>212432</td>
      <td>Italy</td>
      <td>0.124991</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GB</td>
      <td>2020-09-11</td>
      <td>364085</td>
      <td>41703</td>
      <td>1848</td>
      <td>United Kingdom of Great Britain and Northern I...</td>
      <td>0.114542</td>
    </tr>
    <tr>
      <th>35</th>
      <td>BE</td>
      <td>2020-09-11</td>
      <td>90568</td>
      <td>9917</td>
      <td>18659</td>
      <td>Belgium</td>
      <td>0.109498</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MX</td>
      <td>2020-09-11</td>
      <td>652364</td>
      <td>69649</td>
      <td>541804</td>
      <td>Mexico</td>
      <td>0.106764</td>
    </tr>
    <tr>
      <th>191</th>
      <td>EH</td>
      <td>2020-09-11</td>
      <td>10</td>
      <td>1</td>
      <td>8</td>
      <td>Western Sahara</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>EC</td>
      <td>2020-09-11</td>
      <td>113206</td>
      <td>10749</td>
      <td>91242</td>
      <td>Ecuador</td>
      <td>0.094951</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FR</td>
      <td>2020-09-11</td>
      <td>401794</td>
      <td>30899</td>
      <td>89823</td>
      <td>France</td>
      <td>0.076903</td>
    </tr>
    <tr>
      <th>39</th>
      <td>NL</td>
      <td>2020-09-11</td>
      <td>83369</td>
      <td>6290</td>
      <td>1846</td>
      <td>Netherlands</td>
      <td>0.075448</td>
    </tr>
    <tr>
      <th>157</th>
      <td>TD</td>
      <td>2020-09-11</td>
      <td>1081</td>
      <td>79</td>
      <td>934</td>
      <td>Chad</td>
      <td>0.073080</td>
    </tr>
    <tr>
      <th>38</th>
      <td>SE</td>
      <td>2020-09-11</td>
      <td>86505</td>
      <td>5846</td>
      <td>0</td>
      <td>Sweden</td>
      <td>0.067580</td>
    </tr>
    <tr>
      <th>25</th>
      <td>CA</td>
      <td>2020-09-11</td>
      <td>137531</td>
      <td>9214</td>
      <td>121523</td>
      <td>Canada</td>
      <td>0.066996</td>
    </tr>
    <tr>
      <th>181</th>
      <td>FJ</td>
      <td>2020-09-11</td>
      <td>32</td>
      <td>2</td>
      <td>24</td>
      <td>Fiji</td>
      <td>0.062500</td>
    </tr>
    <tr>
      <th>154</th>
      <td>LR</td>
      <td>2020-09-11</td>
      <td>1315</td>
      <td>82</td>
      <td>1199</td>
      <td>Liberia</td>
      <td>0.062357</td>
    </tr>
    <tr>
      <th>84</th>
      <td>SD</td>
      <td>2020-09-11</td>
      <td>13470</td>
      <td>834</td>
      <td>6731</td>
      <td>Sudan</td>
      <td>0.061915</td>
    </tr>
    <tr>
      <th>155</th>
      <td>NE</td>
      <td>2020-09-11</td>
      <td>1178</td>
      <td>69</td>
      <td>1100</td>
      <td>Niger</td>
      <td>0.058574</td>
    </tr>
    <tr>
      <th>68</th>
      <td>IE</td>
      <td>2020-09-11</td>
      <td>30571</td>
      <td>1781</td>
      <td>23364</td>
      <td>Ireland</td>
      <td>0.058258</td>
    </tr>
    <tr>
      <th>160</th>
      <td>SM</td>
      <td>2020-09-11</td>
      <td>722</td>
      <td>42</td>
      <td>662</td>
      <td>San Marino</td>
      <td>0.058172</td>
    </tr>
    <tr>
      <th>26</th>
      <td>BO</td>
      <td>2020-09-11</td>
      <td>124205</td>
      <td>7193</td>
      <td>79483</td>
      <td>Plurinational State of Bolivia</td>
      <td>0.057912</td>
    </tr>
    <tr>
      <th>91</th>
      <td>HU</td>
      <td>2020-09-11</td>
      <td>10909</td>
      <td>631</td>
      <td>4014</td>
      <td>Hungary</td>
      <td>0.057842</td>
    </tr>
  </tbody>
</table>
</div>



## Visualisasi Top 20 Negara dengan Fatality Rate Tertinggi
Untuk memvisualisasikan negara-negara dengan kasus fatality rate tertinggi akibat covid-19 ini dapat dilakukan dengan menggunakan bar chart. Dimana sumbu x adalah kolom **name** (nama negara), dan sumbu y adalah kolom **fatality_ratio**, lalu gunakan fungsi plt.barh(x, y) dari library matplotlib agar nama negara terlihat jelas.


```python
# Membuat grafik Top 20 Negara
plt.figure(figsize=(13, 8))

top_20 = df_top_20_fatality_rate.sort_values(by='fatality_ratio')
x = top_20['name']
y = top_20['fatality_ratio']
plt.barh(x,y)

# Labeling axis
plt.xlabel('Fatality Rate')
plt.ylabel('Country Name')
plt.title('Top 20 Highest Fatality Rate Countries')
# plt.xticks(rotation=90)
# plt.tight_layout()

plt.show()
```


    
![png](/assets/img/data-analysis-covid19/output_22_0.png)
    


Dari grafik tersebut dapat disimpulkan **Negara Yemen** memiliki fatality ratio tertinggi yaitu **0.289985**, walaupun memiliki total kasus lebih sedikit dibandingkan Italy di peringkat kedua.

# Analisis COVID-19 di ASEAN

## Menggabungkan Dataframe
Kali ini kita akan membandingkan kasus covid-19 di negara ASEAN, yaitu Indonesia, Thailand, Singapore, Malaysia, Philippines, Vietnam, Brunei, Myanmar (Burma), Cambodia, Laos. Berikut adalah kode negara ASEAN yang akan kita masukkan sebagai list.

- ID -> Indonesia
- TH -> Thailand
- SG -> Singapore
- MY -> Malaysia
- PH -> Philippines
- VN -> Vietnam
- BN -> Brunei
- MM -> Myanmar
- KH -> Cambodia
- LA -> Laos

Untuk itu, API country akan dipanggil berkali-kali sebanyak negara yang ada di list tersebut menggunakan perulanagan/looping. Tahapan mengambil data API masih sama seperti di awal.


```python
# Membuat list kode negara ASEAN
countries = ['ID','TH','SG','MY','PH','VN','BN','MM','KH','LA']

# Looping untk mengambil data covid-19 tiap negara
i = 0
for country in countries:
	covid_timeline_url = 'https://covid19-api.org/api/timeline/'+country
	df_covid_timeline = pd.io.json.json_normalize(get_json(covid_timeline_url))
	df_covid_timeline['last_update'] = pd.to_datetime(df_covid_timeline['last_update'], format='%Y-%m-%dT%H:%M:%S')
	df_covid_timeline['last_update'] = df_covid_timeline['last_update'].apply(lambda x: x.date())
	if i==0:
		df_covid_timeline_merged = df_covid_timeline
	else:
		df_covid_timeline_merged = df_covid_timeline.append(df_covid_timeline_merged, ignore_index=True)
	i=i+1

df_covid_timeline_merged.head()
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
      <th>country</th>
      <th>last_update</th>
      <th>cases</th>
      <th>deaths</th>
      <th>recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LA</td>
      <td>2020-09-12</td>
      <td>23</td>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LA</td>
      <td>2020-09-11</td>
      <td>23</td>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LA</td>
      <td>2020-09-10</td>
      <td>22</td>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LA</td>
      <td>2020-09-09</td>
      <td>22</td>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LA</td>
      <td>2020-09-08</td>
      <td>22</td>
      <td>0</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



## Merge Data Covid19 Dengan Data Country
Tahap berikutnya adalah menggabungkan 2 dataframe yaitu **df_covid_timeline_merged** dengan **df_countries** dengan kolom country (kode negara) sebagai pivot.


```python
# Merge Data Covid-19 dan Country
df_covid_timeline_denormalized = pd.merge(df_covid_timeline_merged, df_countries, on='country')

df_covid_timeline_denormalized.head()
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
      <th>country</th>
      <th>last_update</th>
      <th>cases</th>
      <th>deaths</th>
      <th>recovered</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LA</td>
      <td>2020-09-12</td>
      <td>23</td>
      <td>0</td>
      <td>21</td>
      <td>Lao People's Democratic Republic</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LA</td>
      <td>2020-09-11</td>
      <td>23</td>
      <td>0</td>
      <td>21</td>
      <td>Lao People's Democratic Republic</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LA</td>
      <td>2020-09-10</td>
      <td>22</td>
      <td>0</td>
      <td>21</td>
      <td>Lao People's Democratic Republic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LA</td>
      <td>2020-09-09</td>
      <td>22</td>
      <td>0</td>
      <td>21</td>
      <td>Lao People's Democratic Republic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LA</td>
      <td>2020-09-08</td>
      <td>22</td>
      <td>0</td>
      <td>21</td>
      <td>Lao People's Democratic Republic</td>
    </tr>
  </tbody>
</table>
</div>



## Pertumbuhan Kasus Covid-19 di ASEAN
Untuk melihat pertumbuhan kasus covid-19, kita akan memfilter kolom last_update dari data covid-19 sehingga hanya data dari **tanggal 1 Maret 2020** ke atas yang diambil menggunakan fungsi **datetime.date** dari library datetime dengan format (YYYY, mm, dd).


```python
import datetime

# Filter pertumbuhan covid-19 pada tanggal tertentu
df_covid_timeline_denormalized = df_covid_timeline_denormalized[(df_covid_timeline_denormalized['last_update'] >= datetime.date(2020, 3, 1))]

```

## Visualisasi Kasus Covid-19 di ASEAN
Untuk menggambarkan pertumbuhan kasus covid-19, kita akan gunakan line chart. Dimana sumbu x adalah tanggal (**last_update**) pada tiap-tiap negara dan y adalah jumlah kasus (**cases**) pada tiap-tiap negara.


```python
# List negara ASEAN
countries = ['ID','TH','SG','MY','PH','VN','BN','MM','KH','LA']

# Membuat perulangan line chart tiap negara

plt.figure(figsize=(12,7))
for country in countries:
	country_data = df_covid_timeline_denormalized['country']==country
	x = df_covid_timeline_denormalized[country_data]['last_update']
	y = df_covid_timeline_denormalized[country_data]['cases']
	plt.plot(x, y, label = country)

plt.legend()
plt.xlabel('Record Date')
plt.ylabel('Total Cases')
plt.title('Asean Covid19 Cases Comparison')

plt.show()
```


    
![png](/assets/img/data-analysis-covid19/output_32_0.png)
    


Dari grafik di atas dapat disimpulkan **Negara Philippines** memiliki jumlah kasus terbanyak di ASEAN yaitu **diatas 250.000 kasus**, disusul Indonesia di peringkat 2, dan Singapore di peringkat 3.
