---
layout: post
title: Movies Recommendation (Part 1)
color: magenta
thumbnail: "assets/img/thumbnails/movies.jpg"
tags: [Preprocessing, Movies Recommendation, Simple Recommender]
---

**Simple Recommender System** adalah sistem rekomendasi yang hanya menggunakan urutan sebagai dasar perhitungannya. Sebagai contoh dalam menentukan rekomendasi film terbaik, kita akan menggunakan urutan berdasarkan mungkin vote terbanyak, rating tertinggi, penjualan film paling tinggi, atau apapun yang lain. Dalam pembahasan ini, kita akan **membuat sistem rekomendasi film** menggunakan kombinasi antara rata-rata rating, jumlah vote, dan membentuk metric baru dari metric yang sudah ada, kemudian kita akan melakukan sorting untuk metric ini dari yang tertinggi ke terendah.

## **Simple Recommender using Weighted Rating**

**Simple Recommender** menawarkan rekomendasi yang umum untuk semua user berdasarkan popularitas film dan terkadang genre. Ide awal di balik sistem rekomendasi ini adalah sebagai berikut.
  -  Film-film yang lebih populer akan memiliki kemungkinan yang lebih besar untuk disukai juga oleh rata-rata penonton.
  -  Model ini tidak memberikan rekomendasi yang personal untuk setiap tipe user. 
  -  Implementasi model ini pun juga bisa dibilang cukup mudah, yang perlu kita lakukan hanyalah mengurutkan film-film tersebut berdasarkan rating dan popularitas dan menunjukkan film teratas dari list film tersebut.

Sebagai tambahan, kita dapat menambahkan genre untuk mendapatkan film teratas untuk genre spesifik tersebut

## **Formula dari IMDB dengan Weighted Rating**

$Weighted  Rating = (v / (v+m)). R +  (m / (v+m)).C$

dimana,
- **v**: jumlah votes untuk film tersebut
- **m**: jumlah minimum votes yang dibutuhkan supaya dapat masuk dalam chart
- **R**: rata-rata rating dari film tersebut
- **C**: rata-rata jumlah votes dari seluruh semesta film

## **Dataset**

Dataset yang digunakan dalam pembahasan ini, yaitu:

-  `title.basic.tsv` yang berisi informasi umum mengenai film-film yang ada.   
    [https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/title.basics.tsv](https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/title.basics.tsv)
-  `title.ratings.tsv` yang berisi mengenail rating dan jumlah votes dari film-film yang ada.   
    [https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/title.ratings.tsv](https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/title.ratings.tsv)


## **Library**

Library yang dibutuhkan untuk pembahasan ini adalah:
- **numpy** untuk perhitungan array atau matriks
- **pandas** untuk manipulasi dan analisis data


```python
# Load library
import numpy as np
import pandas as pd
```

## **File Unloading**

Melakukan pembacaan file `title_basic.tsv` dan `title_rating.tsv` ke dalam bentuk dataframe.


```python
# Load file into dataframe
movie_df = pd.read_csv('data/title_basics.tsv', sep='\t')
rating_df = pd.read_csv('data/title_ratings.tsv', sep='\t')
```

## **Data Cleaning**

Adapun langkah-langkah yang dilakukan, seperti:
- Preview data awal
- Melihat informasi data
- Mengecek dan mengatasi data kosong atau *missing values*
- Mengecek dan mengatasi format data yang tidak sesuai

### **Table Movies**


```python
# Print first five rows
movie_df.head()
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
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>originalTitle</th>
      <th>isAdult</th>
      <th>startYear</th>
      <th>endYear</th>
      <th>runtimeMinutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0221078</td>
      <td>short</td>
      <td>Circle Dance, Ute Indians</td>
      <td>Circle Dance, Ute Indians</td>
      <td>0</td>
      <td>1898</td>
      <td>\N</td>
      <td>\N</td>
      <td>Documentary,Short</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt8862466</td>
      <td>tvEpisode</td>
      <td>¡El #TeamOsos va con todo al "Reality del amor"!</td>
      <td>¡El #TeamOsos va con todo al "Reality del amor"!</td>
      <td>0</td>
      <td>2018</td>
      <td>\N</td>
      <td>\N</td>
      <td>Comedy,Drama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt7157720</td>
      <td>tvEpisode</td>
      <td>Episode #3.41</td>
      <td>Episode #3.41</td>
      <td>0</td>
      <td>2016</td>
      <td>\N</td>
      <td>29</td>
      <td>Comedy,Game-Show</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt2974998</td>
      <td>tvEpisode</td>
      <td>Episode dated 16 May 1987</td>
      <td>Episode dated 16 May 1987</td>
      <td>0</td>
      <td>1987</td>
      <td>\N</td>
      <td>\N</td>
      <td>News</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt2903620</td>
      <td>tvEpisode</td>
      <td>Frances Bavier: Aunt Bee Retires</td>
      <td>Frances Bavier: Aunt Bee Retires</td>
      <td>0</td>
      <td>1973</td>
      <td>\N</td>
      <td>\N</td>
      <td>Documentary</td>
    </tr>
  </tbody>
</table>
</div>



Terlihat ada kolom dengan nilai `\\N` yang dimungkinkan kesalahan pembacaan data.


```python
# View info data
movie_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9025 entries, 0 to 9024
    Data columns (total 9 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   tconst          9025 non-null   object
     1   titleType       9025 non-null   object
     2   primaryTitle    9011 non-null   object
     3   originalTitle   9011 non-null   object
     4   isAdult         9025 non-null   int64 
     5   startYear       9025 non-null   object
     6   endYear         9025 non-null   object
     7   runtimeMinutes  9025 non-null   object
     8   genres          9014 non-null   object
    dtypes: int64(1), object(8)
    memory usage: 634.7+ KB
    

Kemudian mengecek jumlah data kosong atau *missing values* pada kolom yang ada.


```python
# Check missing values
movie_df.isnull().sum()
```




    tconst             0
    titleType          0
    primaryTitle      14
    originalTitle     14
    isAdult            0
    startYear          0
    endYear            0
    runtimeMinutes     0
    genres            11
    dtype: int64



Dari hasil pengecekan di atas diketahui bahwa kolom **primaryTitle, originalTitle, dan genres** terdapat *missing values*. Untuk mengatasi hal itu akan dilakukan **penghapusan baris** pada data yang kosong karena jumlahnya masih sedikit.


```python
# Get rows with missing values
movies_missing = movie_df.loc[(movie_df['primaryTitle'].isnull()) | (movie_df['originalTitle'].isnull()) | 
                              (movie_df['genres'].isnull())]

print('Jumlah data movies yang kosong yaitu', len(movies_missing))
movies_missing
```

    Jumlah data movies yang kosong yaitu 25
    




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
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>originalTitle</th>
      <th>isAdult</th>
      <th>startYear</th>
      <th>endYear</th>
      <th>runtimeMinutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9000</th>
      <td>tt10790040</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2019</td>
      <td>\N</td>
      <td>\N</td>
      <td>\N</td>
    </tr>
    <tr>
      <th>9001</th>
      <td>tt10891902</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2020</td>
      <td>\N</td>
      <td>\N</td>
      <td>Crime</td>
    </tr>
    <tr>
      <th>9002</th>
      <td>tt11737860</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2020</td>
      <td>\N</td>
      <td>\N</td>
      <td>Comedy,Drama,Romance</td>
    </tr>
    <tr>
      <th>9003</th>
      <td>tt11737862</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2020</td>
      <td>\N</td>
      <td>\N</td>
      <td>Comedy,Drama,Romance</td>
    </tr>
    <tr>
      <th>9004</th>
      <td>tt11737866</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2020</td>
      <td>\N</td>
      <td>\N</td>
      <td>Comedy,Drama,Romance</td>
    </tr>
    <tr>
      <th>9005</th>
      <td>tt11737872</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2020</td>
      <td>\N</td>
      <td>\N</td>
      <td>\N</td>
    </tr>
    <tr>
      <th>9006</th>
      <td>tt11737874</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2020</td>
      <td>\N</td>
      <td>\N</td>
      <td>Comedy,Drama,Romance</td>
    </tr>
    <tr>
      <th>9007</th>
      <td>tt1971246</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2011</td>
      <td>\N</td>
      <td>\N</td>
      <td>Biography</td>
    </tr>
    <tr>
      <th>9008</th>
      <td>tt2067043</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1965</td>
      <td>\N</td>
      <td>\N</td>
      <td>Music</td>
    </tr>
    <tr>
      <th>9009</th>
      <td>tt4404732</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2015</td>
      <td>\N</td>
      <td>\N</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>9010</th>
      <td>tt5773048</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2015</td>
      <td>\N</td>
      <td>\N</td>
      <td>Talk-Show</td>
    </tr>
    <tr>
      <th>9011</th>
      <td>tt8473688</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1987</td>
      <td>\N</td>
      <td>\N</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>9012</th>
      <td>tt8541336</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2018</td>
      <td>\N</td>
      <td>\N</td>
      <td>Reality-TV,Romance</td>
    </tr>
    <tr>
      <th>9013</th>
      <td>tt9824302</td>
      <td>tvEpisode</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2016</td>
      <td>\N</td>
      <td>\N</td>
      <td>Documentary</td>
    </tr>
    <tr>
      <th>9014</th>
      <td>tt10233364</td>
      <td>tvEpisode</td>
      <td>Rolling in the Deep Dish\tRolling in the Deep ...</td>
      <td>0</td>
      <td>2019</td>
      <td>\N</td>
      <td>\N</td>
      <td>Reality-TV</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9015</th>
      <td>tt10925142</td>
      <td>tvEpisode</td>
      <td>The IMDb Show on Location: Star Wars Galaxy's ...</td>
      <td>0</td>
      <td>2019</td>
      <td>\N</td>
      <td>\N</td>
      <td>Talk-Show</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9016</th>
      <td>tt10970874</td>
      <td>tvEpisode</td>
      <td>Die Bauhaus-Stadt Tel Aviv - Vorbild für die M...</td>
      <td>0</td>
      <td>2019</td>
      <td>\N</td>
      <td>\N</td>
      <td>\N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9017</th>
      <td>tt11670006</td>
      <td>tvEpisode</td>
      <td>...ein angenehmer Unbequemer...\t...ein angene...</td>
      <td>0</td>
      <td>1981</td>
      <td>\N</td>
      <td>\N</td>
      <td>Documentary</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9018</th>
      <td>tt11868642</td>
      <td>tvEpisode</td>
      <td>GGN Heavyweight Championship Lungs With Mike T...</td>
      <td>0</td>
      <td>2020</td>
      <td>\N</td>
      <td>\N</td>
      <td>Talk-Show</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9019</th>
      <td>tt2347742</td>
      <td>tvEpisode</td>
      <td>No sufras por la alergia esta primavera\tNo su...</td>
      <td>0</td>
      <td>2004</td>
      <td>\N</td>
      <td>\N</td>
      <td>\N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9020</th>
      <td>tt3984412</td>
      <td>tvEpisode</td>
      <td>I'm Not Going to Come Last, I'm Just Going to ...</td>
      <td>0</td>
      <td>2014</td>
      <td>\N</td>
      <td>\N</td>
      <td>Reality-TV</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9021</th>
      <td>tt8740950</td>
      <td>tvEpisode</td>
      <td>Weight Loss Resolution Restart - Ins &amp; Outs of...</td>
      <td>0</td>
      <td>2015</td>
      <td>\N</td>
      <td>\N</td>
      <td>Reality-TV</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9022</th>
      <td>tt9822816</td>
      <td>tvEpisode</td>
      <td>Zwischen Vertuschung und Aufklärung - Missbrau...</td>
      <td>0</td>
      <td>2019</td>
      <td>\N</td>
      <td>\N</td>
      <td>\N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9023</th>
      <td>tt9900062</td>
      <td>tvEpisode</td>
      <td>The Direction of Yuu's Love: Hings Aren't Goin...</td>
      <td>0</td>
      <td>1994</td>
      <td>\N</td>
      <td>\N</td>
      <td>Animation,Comedy,Drama</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9024</th>
      <td>tt9909210</td>
      <td>tvEpisode</td>
      <td>Politik und/oder Moral - Wie weit geht das Ver...</td>
      <td>0</td>
      <td>2005</td>
      <td>\N</td>
      <td>\N</td>
      <td>\N</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get data without missing values
movie_df = movie_df.loc[(movie_df['primaryTitle'].notnull()) & (movie_df['originalTitle'].notnull()) & 
                        (movie_df['genres'].notnull())]

# Print number of rows
print('Jumlah data movies tanpa missing values yaitu', len(movie_df))
```

    Jumlah data movies tanpa missing values yaitu 9000
    

Jika kita perhatikan pada kolom **startYear, endYear, runtimeMinutes, dan genres**, terdapat data dengan nilai `\\N` yang berarti **NULL** (kesalahan format). Hal selanjutnya yang akan kita lakukan adalah mengubah nilai dari `\\N` tersebut menjadi `np.nan` dan melakukan **formatting tipe data** kolom startYear, endYear, dan runtimeMinutes menjadi float64.


```python
# Change values'\\N' column startYear
movie_df['startYear'] = movie_df['startYear'].replace('\\N', np.nan)
movie_df['startYear'] = movie_df['startYear'].astype('float64')
# Change values '\\N' column endYear
movie_df['endYear'] = movie_df['endYear'].replace('\\N', np.nan)
movie_df['endYear'] = movie_df['endYear'].astype('float64')
# Change values '\\N' column runtimeMinutes
movie_df['runtimeMinutes'] = movie_df['runtimeMinutes'].replace('\\N', np.nan)
movie_df['runtimeMinutes'] = movie_df['runtimeMinutes'].astype('float64')

# View new first five rows
movie_df.head()
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
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>originalTitle</th>
      <th>isAdult</th>
      <th>startYear</th>
      <th>endYear</th>
      <th>runtimeMinutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0221078</td>
      <td>short</td>
      <td>Circle Dance, Ute Indians</td>
      <td>Circle Dance, Ute Indians</td>
      <td>0</td>
      <td>1898.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Documentary,Short</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt8862466</td>
      <td>tvEpisode</td>
      <td>¡El #TeamOsos va con todo al "Reality del amor"!</td>
      <td>¡El #TeamOsos va con todo al "Reality del amor"!</td>
      <td>0</td>
      <td>2018.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Comedy,Drama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt7157720</td>
      <td>tvEpisode</td>
      <td>Episode #3.41</td>
      <td>Episode #3.41</td>
      <td>0</td>
      <td>2016.0</td>
      <td>NaN</td>
      <td>29.0</td>
      <td>Comedy,Game-Show</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt2974998</td>
      <td>tvEpisode</td>
      <td>Episode dated 16 May 1987</td>
      <td>Episode dated 16 May 1987</td>
      <td>0</td>
      <td>1987.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>News</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt2903620</td>
      <td>tvEpisode</td>
      <td>Frances Bavier: Aunt Bee Retires</td>
      <td>Frances Bavier: Aunt Bee Retires</td>
      <td>0</td>
      <td>1973.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Documentary</td>
    </tr>
  </tbody>
</table>
</div>



Selanjutnya, kita akan membuat sebuah fungsi yang bernama `transform_to_list` untuk mengubah nilai genre menjadi list. 


```python
def transform_to_list(x):
    if ',' in x: 
        # Change values genres to list with split (comma)
        return x.split(',')
    else: 
        # Return self list
        return [x]
    
    if x == '\\N':
        # Return empty list
        return []

movie_df['genres'] = movie_df['genres'].apply(lambda x: transform_to_list(x))
movie_df['genres'].head()
```




    0    [Documentary, Short]
    1         [Comedy, Drama]
    2     [Comedy, Game-Show]
    3                  [News]
    4           [Documentary]
    Name: genres, dtype: object



### **Table Rating**


```python
# Print first five rows
rating_df.head()
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
      <th>tconst</th>
      <th>averageRating</th>
      <th>numVotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0000001</td>
      <td>5.6</td>
      <td>1608</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0000002</td>
      <td>6.0</td>
      <td>197</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0000003</td>
      <td>6.5</td>
      <td>1285</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0000004</td>
      <td>6.1</td>
      <td>121</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0000005</td>
      <td>6.1</td>
      <td>2050</td>
    </tr>
  </tbody>
</table>
</div>




```python
# View info data
rating_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1030009 entries, 0 to 1030008
    Data columns (total 3 columns):
     #   Column         Non-Null Count    Dtype  
    ---  ------         --------------    -----  
     0   tconst         1030009 non-null  object 
     1   averageRating  1030009 non-null  float64
     2   numVotes       1030009 non-null  int64  
    dtypes: float64(1), int64(1), object(1)
    memory usage: 23.6+ MB
    

Data rating sudah bersih karena tidak ditemukan data kosong atau *missing values* dan *wrong format*.

## **Merge Table Movie and Table Rating**

Melakukan penggabungan data dari kedua tabel berdasarkan kolom `tconst` dengan **inner join**.


```python
# Merge table movie and rating
movie_rating_df = pd.merge(movie_df, rating_df, on='tconst', how='inner')

# Print first five rows
movie_rating_df.head()
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
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>originalTitle</th>
      <th>isAdult</th>
      <th>startYear</th>
      <th>endYear</th>
      <th>runtimeMinutes</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>numVotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0043745</td>
      <td>short</td>
      <td>Lion Down</td>
      <td>Lion Down</td>
      <td>0</td>
      <td>1951.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>[Animation, Comedy, Family]</td>
      <td>7.1</td>
      <td>459</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0167491</td>
      <td>video</td>
      <td>Wicked Covergirls</td>
      <td>Wicked Covergirls</td>
      <td>1</td>
      <td>1998.0</td>
      <td>NaN</td>
      <td>85.0</td>
      <td>[Adult]</td>
      <td>5.7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt6574096</td>
      <td>tvEpisode</td>
      <td>Shadow Play - Part 2</td>
      <td>Shadow Play - Part 2</td>
      <td>0</td>
      <td>2017.0</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>[Adventure, Animation, Comedy]</td>
      <td>8.5</td>
      <td>240</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt6941700</td>
      <td>tvEpisode</td>
      <td>RuPaul Roast</td>
      <td>RuPaul Roast</td>
      <td>0</td>
      <td>2017.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[Reality-TV]</td>
      <td>8.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt7305674</td>
      <td>video</td>
      <td>UCLA Track &amp; Field Promo</td>
      <td>UCLA Track &amp; Field Promo</td>
      <td>0</td>
      <td>2017.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[Short, Sport]</td>
      <td>9.7</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# View info data
print(movie_rating_df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1376 entries, 0 to 1375
    Data columns (total 11 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   tconst          1376 non-null   object 
     1   titleType       1376 non-null   object 
     2   primaryTitle    1376 non-null   object 
     3   originalTitle   1376 non-null   object 
     4   isAdult         1376 non-null   int64  
     5   startYear       1376 non-null   float64
     6   endYear         26 non-null     float64
     7   runtimeMinutes  1004 non-null   float64
     8   genres          1376 non-null   object 
     9   averageRating   1376 non-null   float64
     10  numVotes        1376 non-null   int64  
    dtypes: float64(4), int64(2), object(5)
    memory usage: 129.0+ KB
    None
    

Terlihat masih ada data kosong pada kolom **endYear dan runtimeMinutes**. Disini kita hanya akan menghilangkan data yang tidak memiliki durasi saja.


```python
# Remove missing data
movie_rating_df = movie_rating_df.dropna(subset=['runtimeMinutes'])

# Print total data
print('Jumlah data baru yaitu', len(movie_rating_df))
```

    Jumlah data baru yaitu 1004
    

## **Building Simple Recommender System**

Sesuai rumus Weighted Rating, kita akan cari nilai-nilai berikut:

$Weighted  Rating = (v / (v+m)). R +  (m / (v+m)).C$

**Nilai C**

Hal pertama yang akan kita cari adalah nilai dari C yang merupakan rata-rata dari averageRating.


```python
C = movie_rating_df['averageRating'].mean()
print(C)
```

    6.829581673306767
    

**Nilai m**

Mari kita ambil contoh film dengan numVotes di atas 80% populasi, jadi populasi yang akan kita ambil hanya sebesar 20%. 


```python
m = movie_rating_df['numVotes'].quantile(0.8)
print(m)
```

    229.0
    

Selanjutnya kita harus membuat sebuah fungsi `imdb_weighted_rating` berdasarkan rumus Weighted Rating.


```python
# Function Weighted Rating
def imdb_weighted_rating(df, var=0.8):
    # Variabel IMDB Score
    v = df['numVotes']
    R = df['averageRating']
    C = df['averageRating'].mean()
    m = df['numVotes'].quantile(var)
    
    # Formula IMDB
    df['score'] = (v/(m+v))*R + (m/(m+v))*C
    return df['score']
    
imdb_weighted_rating(movie_rating_df)

# View data with IMDB score
movie_rating_df.head()
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
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>originalTitle</th>
      <th>isAdult</th>
      <th>startYear</th>
      <th>endYear</th>
      <th>runtimeMinutes</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>numVotes</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0043745</td>
      <td>short</td>
      <td>Lion Down</td>
      <td>Lion Down</td>
      <td>0</td>
      <td>1951.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>[Animation, Comedy, Family]</td>
      <td>7.1</td>
      <td>459</td>
      <td>7.009992</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0167491</td>
      <td>video</td>
      <td>Wicked Covergirls</td>
      <td>Wicked Covergirls</td>
      <td>1</td>
      <td>1998.0</td>
      <td>NaN</td>
      <td>85.0</td>
      <td>[Adult]</td>
      <td>5.7</td>
      <td>7</td>
      <td>6.796077</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt6574096</td>
      <td>tvEpisode</td>
      <td>Shadow Play - Part 2</td>
      <td>Shadow Play - Part 2</td>
      <td>0</td>
      <td>2017.0</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>[Adventure, Animation, Comedy]</td>
      <td>8.5</td>
      <td>240</td>
      <td>7.684380</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tt2262289</td>
      <td>movie</td>
      <td>The Pin</td>
      <td>The Pin</td>
      <td>0</td>
      <td>2013.0</td>
      <td>NaN</td>
      <td>85.0</td>
      <td>[Drama]</td>
      <td>7.7</td>
      <td>27</td>
      <td>6.921384</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tt0874027</td>
      <td>tvEpisode</td>
      <td>Episode #32.9</td>
      <td>Episode #32.9</td>
      <td>0</td>
      <td>2006.0</td>
      <td>NaN</td>
      <td>29.0</td>
      <td>[Comedy, Game-Show, News]</td>
      <td>8.0</td>
      <td>8</td>
      <td>6.869089</td>
    </tr>
  </tbody>
</table>
</div>
<br>


Dari tahap yang sudah kita lakukan sebelumnya, telah terdapat kolom tambahan **score**. Pertama kita akan filter numVotes yang lebih dari m kemudian diurutkan score dari tertinggi ke terendah untuk diambil nilai beberapa nilai teratas.


```python
# Create function recommender system
def simple_recommender(df, top=100):
    # Filtering and sorting
    df = df.loc[df['numVotes'] >= m]
    df = df.sort_values(by='score', ascending=False)
    
    # Get top 100
    df = df[:top]
    return df
    
# Get top 25 data movies    
simple_recommender(movie_rating_df, top=25)
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
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>originalTitle</th>
      <th>isAdult</th>
      <th>startYear</th>
      <th>endYear</th>
      <th>runtimeMinutes</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>numVotes</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>68</th>
      <td>tt4110822</td>
      <td>tvEpisode</td>
      <td>S.O.S. Part 2</td>
      <td>S.O.S. Part 2</td>
      <td>0</td>
      <td>2015.0</td>
      <td>NaN</td>
      <td>43.0</td>
      <td>[Action, Adventure, Drama]</td>
      <td>9.4</td>
      <td>3820</td>
      <td>9.254624</td>
    </tr>
    <tr>
      <th>236</th>
      <td>tt2200252</td>
      <td>video</td>
      <td>Attack of the Clones Review</td>
      <td>Attack of the Clones Review</td>
      <td>0</td>
      <td>2010.0</td>
      <td>NaN</td>
      <td>86.0</td>
      <td>[Comedy]</td>
      <td>9.3</td>
      <td>1411</td>
      <td>8.955045</td>
    </tr>
    <tr>
      <th>1181</th>
      <td>tt7697962</td>
      <td>tvEpisode</td>
      <td>Chapter Seventeen: The Missionaries</td>
      <td>Chapter Seventeen: The Missionaries</td>
      <td>0</td>
      <td>2019.0</td>
      <td>NaN</td>
      <td>54.0</td>
      <td>[Drama, Fantasy, Horror]</td>
      <td>9.2</td>
      <td>1536</td>
      <td>8.892450</td>
    </tr>
    <tr>
      <th>326</th>
      <td>tt7124590</td>
      <td>tvEpisode</td>
      <td>Chapter Thirty-Four: Judgment Night</td>
      <td>Chapter Thirty-Four: Judgment Night</td>
      <td>0</td>
      <td>2018.0</td>
      <td>NaN</td>
      <td>42.0</td>
      <td>[Crime, Drama, Mystery]</td>
      <td>9.1</td>
      <td>1859</td>
      <td>8.850993</td>
    </tr>
    <tr>
      <th>1045</th>
      <td>tt0533506</td>
      <td>tvEpisode</td>
      <td>The Prom</td>
      <td>The Prom</td>
      <td>0</td>
      <td>1999.0</td>
      <td>NaN</td>
      <td>60.0</td>
      <td>[Action, Drama, Fantasy]</td>
      <td>8.9</td>
      <td>2740</td>
      <td>8.740308</td>
    </tr>
    <tr>
      <th>71</th>
      <td>tt8399426</td>
      <td>tvEpisode</td>
      <td>Savages</td>
      <td>Savages</td>
      <td>0</td>
      <td>2018.0</td>
      <td>NaN</td>
      <td>58.0</td>
      <td>[Drama, Fantasy, Romance]</td>
      <td>9.0</td>
      <td>1428</td>
      <td>8.700045</td>
    </tr>
    <tr>
      <th>1234</th>
      <td>tt2843830</td>
      <td>tvEpisode</td>
      <td>VIII.</td>
      <td>VIII.</td>
      <td>0</td>
      <td>2014.0</td>
      <td>NaN</td>
      <td>57.0</td>
      <td>[Adventure, Drama]</td>
      <td>8.9</td>
      <td>1753</td>
      <td>8.660784</td>
    </tr>
    <tr>
      <th>1087</th>
      <td>tt4295140</td>
      <td>tvSeries</td>
      <td>Chef's Table</td>
      <td>Chef's Table</td>
      <td>0</td>
      <td>2015.0</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>[Documentary]</td>
      <td>8.6</td>
      <td>12056</td>
      <td>8.566998</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>tt2503932</td>
      <td>tvEpisode</td>
      <td>Trial and Error</td>
      <td>Trial and Error</td>
      <td>0</td>
      <td>2013.0</td>
      <td>NaN</td>
      <td>43.0</td>
      <td>[Drama, Fantasy, Horror]</td>
      <td>8.6</td>
      <td>2495</td>
      <td>8.451165</td>
    </tr>
    <tr>
      <th>448</th>
      <td>tt0337566</td>
      <td>video</td>
      <td>AC/DC: Live at Donington</td>
      <td>AC/DC: Live at Donington</td>
      <td>0</td>
      <td>1992.0</td>
      <td>NaN</td>
      <td>120.0</td>
      <td>[Documentary, Music]</td>
      <td>8.5</td>
      <td>1343</td>
      <td>8.256663</td>
    </tr>
    <tr>
      <th>624</th>
      <td>tt0620159</td>
      <td>tvEpisode</td>
      <td>Strike Out</td>
      <td>Strike Out</td>
      <td>0</td>
      <td>2000.0</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>[Comedy]</td>
      <td>8.7</td>
      <td>401</td>
      <td>8.020118</td>
    </tr>
    <tr>
      <th>1281</th>
      <td>tt3166390</td>
      <td>tvEpisode</td>
      <td>Looking for a Plus-One</td>
      <td>Looking for a Plus-One</td>
      <td>0</td>
      <td>2014.0</td>
      <td>NaN</td>
      <td>28.0</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>8.7</td>
      <td>396</td>
      <td>8.014679</td>
    </tr>
    <tr>
      <th>314</th>
      <td>tt0954759</td>
      <td>tvEpisode</td>
      <td>Ben Franklin</td>
      <td>Ben Franklin</td>
      <td>0</td>
      <td>2007.0</td>
      <td>NaN</td>
      <td>21.0</td>
      <td>[Comedy]</td>
      <td>8.1</td>
      <td>2766</td>
      <td>8.002863</td>
    </tr>
    <tr>
      <th>189</th>
      <td>tt5661506</td>
      <td>video</td>
      <td>Florence + the Machine: The Odyssey</td>
      <td>Florence + the Machine: The Odyssey</td>
      <td>0</td>
      <td>2016.0</td>
      <td>NaN</td>
      <td>49.0</td>
      <td>[Music]</td>
      <td>8.8</td>
      <td>330</td>
      <td>7.992798</td>
    </tr>
    <tr>
      <th>151</th>
      <td>tt3954426</td>
      <td>tvEpisode</td>
      <td>Bleeding Kansas</td>
      <td>Bleeding Kansas</td>
      <td>0</td>
      <td>2014.0</td>
      <td>NaN</td>
      <td>42.0</td>
      <td>[Drama, Western]</td>
      <td>8.6</td>
      <td>437</td>
      <td>7.991253</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>tt6644294</td>
      <td>tvEpisode</td>
      <td>The Hostile Hospital: Part Two</td>
      <td>The Hostile Hospital: Part Two</td>
      <td>0</td>
      <td>2018.0</td>
      <td>NaN</td>
      <td>40.0</td>
      <td>[Adventure, Comedy, Drama]</td>
      <td>8.3</td>
      <td>812</td>
      <td>7.976536</td>
    </tr>
    <tr>
      <th>1242</th>
      <td>tt3677742</td>
      <td>tvSpecial</td>
      <td>Saturday Night Live: 40th Anniversary Special</td>
      <td>Saturday Night Live: 40th Anniversary Special</td>
      <td>0</td>
      <td>2015.0</td>
      <td>NaN</td>
      <td>106.0</td>
      <td>[Comedy]</td>
      <td>8.1</td>
      <td>1931</td>
      <td>7.965312</td>
    </tr>
    <tr>
      <th>1217</th>
      <td>tt3642464</td>
      <td>tvEpisode</td>
      <td>Giant Woman</td>
      <td>Giant Woman</td>
      <td>0</td>
      <td>2014.0</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>[Adventure, Animation, Comedy]</td>
      <td>8.4</td>
      <td>566</td>
      <td>7.947641</td>
    </tr>
    <tr>
      <th>544</th>
      <td>tt0734655</td>
      <td>tvEpisode</td>
      <td>The Little People</td>
      <td>The Little People</td>
      <td>0</td>
      <td>1962.0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>[Drama, Fantasy, Horror]</td>
      <td>8.1</td>
      <td>1559</td>
      <td>7.937290</td>
    </tr>
    <tr>
      <th>49</th>
      <td>tt9119838</td>
      <td>tvEpisode</td>
      <td>Parisian Legend Has It...</td>
      <td>Parisian Legend Has It...</td>
      <td>0</td>
      <td>2019.0</td>
      <td>NaN</td>
      <td>42.0</td>
      <td>[Drama]</td>
      <td>8.9</td>
      <td>263</td>
      <td>7.936330</td>
    </tr>
    <tr>
      <th>357</th>
      <td>tt4084774</td>
      <td>tvEpisode</td>
      <td>Trial and Punishment</td>
      <td>Trial and Punishment</td>
      <td>0</td>
      <td>2015.0</td>
      <td>NaN</td>
      <td>56.0</td>
      <td>[Adventure, Drama]</td>
      <td>8.8</td>
      <td>289</td>
      <td>7.928908</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>tt4174072</td>
      <td>tvEpisode</td>
      <td>Immortal Emerges from Cave</td>
      <td>Immortal Emerges from Cave</td>
      <td>0</td>
      <td>2017.0</td>
      <td>NaN</td>
      <td>53.0</td>
      <td>[Action, Adventure, Crime]</td>
      <td>8.0</td>
      <td>2898</td>
      <td>7.914287</td>
    </tr>
    <tr>
      <th>790</th>
      <td>tt4279086</td>
      <td>tvEpisode</td>
      <td>And Santa's Midnight Run</td>
      <td>And Santa's Midnight Run</td>
      <td>0</td>
      <td>2014.0</td>
      <td>NaN</td>
      <td>42.0</td>
      <td>[Action, Adventure, Comedy]</td>
      <td>8.2</td>
      <td>823</td>
      <td>7.901687</td>
    </tr>
    <tr>
      <th>972</th>
      <td>tt0048028</td>
      <td>movie</td>
      <td>East of Eden</td>
      <td>East of Eden</td>
      <td>0</td>
      <td>1955.0</td>
      <td>NaN</td>
      <td>118.0</td>
      <td>[Drama]</td>
      <td>7.9</td>
      <td>38543</td>
      <td>7.893678</td>
    </tr>
    <tr>
      <th>819</th>
      <td>tt0032156</td>
      <td>movie</td>
      <td>The Story of the Last Chrysanthemum</td>
      <td>Zangiku monogatari</td>
      <td>0</td>
      <td>1939.0</td>
      <td>NaN</td>
      <td>143.0</td>
      <td>[Drama, Romance]</td>
      <td>7.9</td>
      <td>2974</td>
      <td>7.823470</td>
    </tr>
  </tbody>
</table>
</div>
<br>


Dari hasil diatas dapat diketahui bahwa:

- Dari tahap yang sudah dilakukan sebelumnya, dapat dilihat sekarang daftar film telah diurutkan dari score tertinggi ke terendah. Film dengan **averageRating** yang tinggi tidak selalu mendapat posisi yang lebih tinggi dibanding film dengan averageRating lebih rendah, hal ini disebabkan karena kita juga memperhitungkan faktor banyaknya votes.
- Sistem rekomendasi ini masih bisa ditingkatkan dengan menambah filter spesifik tentang titleType, startYear, ataupun filter yang lain.

Selanjutnya yang akan kita lakukan adalah membuat fungsi untuk melakukan filter berdasarkan **isAdult, startYear, dan genres** dan mengetahui hasil sistem rekomendasi film yang diberikan.


```python
# Copy dataframe
new_df = movie_rating_df.copy()

# Create recommender system with filtering
def user_prefer_recommender(df, ask_adult, ask_start_year, ask_genre, top):
    # Ask_adult = yes/no
    if ask_adult.lower() == 'yes':
        df = df.loc[df['isAdult'] == 1]
    elif ask_adult.lower() == 'no':
        df = df.loc[df['isAdult'] == 0]

    # Ask_start_year (numeric)
    df = df.loc[df['startYear'] >= int(ask_start_year)]

    # Ask_genre = 'all' or other genres
    if ask_genre.lower() == 'all':
        df = df
    else:
        def filter_genre(x):
            if ask_genre.lower() in str(x).lower():
                return True
            else:
                return False
        df = df.loc[df['genres'].apply(lambda x: filter_genre(x))]

    # Get rows with greater than or equal m numVotes
    df = df.loc[df['numVotes'] >= m]
    df = df.sort_values(by='score', ascending=False)
    
    # Get top movies
    df = df[:top]
    return df

# Result movies recommendation
user_prefer_recommender(new_df, ask_adult = 'no', ask_start_year = 2000, ask_genre = 'drama', top=10)
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
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>originalTitle</th>
      <th>isAdult</th>
      <th>startYear</th>
      <th>endYear</th>
      <th>runtimeMinutes</th>
      <th>genres</th>
      <th>averageRating</th>
      <th>numVotes</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>68</th>
      <td>tt4110822</td>
      <td>tvEpisode</td>
      <td>S.O.S. Part 2</td>
      <td>S.O.S. Part 2</td>
      <td>0</td>
      <td>2015.0</td>
      <td>NaN</td>
      <td>43.0</td>
      <td>[Action, Adventure, Drama]</td>
      <td>9.4</td>
      <td>3820</td>
      <td>9.254624</td>
    </tr>
    <tr>
      <th>1181</th>
      <td>tt7697962</td>
      <td>tvEpisode</td>
      <td>Chapter Seventeen: The Missionaries</td>
      <td>Chapter Seventeen: The Missionaries</td>
      <td>0</td>
      <td>2019.0</td>
      <td>NaN</td>
      <td>54.0</td>
      <td>[Drama, Fantasy, Horror]</td>
      <td>9.2</td>
      <td>1536</td>
      <td>8.892450</td>
    </tr>
    <tr>
      <th>326</th>
      <td>tt7124590</td>
      <td>tvEpisode</td>
      <td>Chapter Thirty-Four: Judgment Night</td>
      <td>Chapter Thirty-Four: Judgment Night</td>
      <td>0</td>
      <td>2018.0</td>
      <td>NaN</td>
      <td>42.0</td>
      <td>[Crime, Drama, Mystery]</td>
      <td>9.1</td>
      <td>1859</td>
      <td>8.850993</td>
    </tr>
    <tr>
      <th>71</th>
      <td>tt8399426</td>
      <td>tvEpisode</td>
      <td>Savages</td>
      <td>Savages</td>
      <td>0</td>
      <td>2018.0</td>
      <td>NaN</td>
      <td>58.0</td>
      <td>[Drama, Fantasy, Romance]</td>
      <td>9.0</td>
      <td>1428</td>
      <td>8.700045</td>
    </tr>
    <tr>
      <th>1234</th>
      <td>tt2843830</td>
      <td>tvEpisode</td>
      <td>VIII.</td>
      <td>VIII.</td>
      <td>0</td>
      <td>2014.0</td>
      <td>NaN</td>
      <td>57.0</td>
      <td>[Adventure, Drama]</td>
      <td>8.9</td>
      <td>1753</td>
      <td>8.660784</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>tt2503932</td>
      <td>tvEpisode</td>
      <td>Trial and Error</td>
      <td>Trial and Error</td>
      <td>0</td>
      <td>2013.0</td>
      <td>NaN</td>
      <td>43.0</td>
      <td>[Drama, Fantasy, Horror]</td>
      <td>8.6</td>
      <td>2495</td>
      <td>8.451165</td>
    </tr>
    <tr>
      <th>1281</th>
      <td>tt3166390</td>
      <td>tvEpisode</td>
      <td>Looking for a Plus-One</td>
      <td>Looking for a Plus-One</td>
      <td>0</td>
      <td>2014.0</td>
      <td>NaN</td>
      <td>28.0</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>8.7</td>
      <td>396</td>
      <td>8.014679</td>
    </tr>
    <tr>
      <th>151</th>
      <td>tt3954426</td>
      <td>tvEpisode</td>
      <td>Bleeding Kansas</td>
      <td>Bleeding Kansas</td>
      <td>0</td>
      <td>2014.0</td>
      <td>NaN</td>
      <td>42.0</td>
      <td>[Drama, Western]</td>
      <td>8.6</td>
      <td>437</td>
      <td>7.991253</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>tt6644294</td>
      <td>tvEpisode</td>
      <td>The Hostile Hospital: Part Two</td>
      <td>The Hostile Hospital: Part Two</td>
      <td>0</td>
      <td>2018.0</td>
      <td>NaN</td>
      <td>40.0</td>
      <td>[Adventure, Comedy, Drama]</td>
      <td>8.3</td>
      <td>812</td>
      <td>7.976536</td>
    </tr>
    <tr>
      <th>49</th>
      <td>tt9119838</td>
      <td>tvEpisode</td>
      <td>Parisian Legend Has It...</td>
      <td>Parisian Legend Has It...</td>
      <td>0</td>
      <td>2019.0</td>
      <td>NaN</td>
      <td>42.0</td>
      <td>[Drama]</td>
      <td>8.9</td>
      <td>263</td>
      <td>7.936330</td>
    </tr>
  </tbody>
</table>
</div>
<br>


Hasil di atas adalah **rekomendasi 10 film terbaik** bergenre drama pada tahun 2000 keatas dengan kategori bukan hanya untuk dewasa.
