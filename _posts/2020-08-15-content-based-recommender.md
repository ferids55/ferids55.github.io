---
layout: post
title: Movies Recommendation (Part 2)
color: magenta
thumbnail: "assets/img/thumbnails/movies.jpg"
tags: [Preprocessing, Movies Recommendation, Content Based Recommender]
---

Pada project sebelumnya kita telah membuat [Simple Recommender System](/2020/08/05/simple-recommender.html) yang dibuat hanya dengan menggunakan formula Weighted Rating, yaitu mengurutkan score yang terdapat komponen average rating secara descending, kita dapat mengetahui (secara estimasi) film mana yang menurut para audience paling menarik.

## Content Based Recommender System

Kali ini, kita akan membuat sistem rekomendasi yang menggunakan content/feature dari film/entitas tersebut, kemudian melakukan perhitungan terhadap kesamaannya satu dan yang lain sehingga ketika kita menunjuk ke satu film, kita akan mendapat beberapa film lain yang memiliki kesamaan dengan film tersebut. Hal ini biasa kita sebut sebagai **Content Based Recommender System**.

Sebagai contoh, berdasarkan kesamaan plot yang ada dan genre yang ada, ketika audience lebih menyukai film Narnia, maka sistem rekomendasi ini juga akan merekomendasikan film seperti Harry Potter atau The Lords of The Rings yang memiliki genre yang mirip.

## Dataset

Dataset yang akan digunakan dalam pembahasan ini, meliputi:
- `movie_rating_df.csv` yang berisi informasi data film-film beserta rating nya.   
  [https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/movie_rating_df.csv](https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/movie_rating_df.csv)
- `actor_name.csv` yang berisi data aktor yang memainkan film.   
  [https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/actor_name.csv](https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/actor_name.csv)
- `director_writers.csv` yang berisi data direktur dan penulis dari film yang ada.   
  [https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/directors_writers.csv](https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/directors_writers.csv)

## Library

Library yang digunakan dalam project ini antara lain:
- **numpy** untuk perhitungan numerik dengan array atau matriks.
- **pandas** untuk manipulasi dan analisis data.
- **sklearn** untuk pemodelan data.


```python
# Load library
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')
```

## Load Dataset

Melakukan pembacaan file csv ke dalam bentuk dataframe, kemudian melakukan preview data dan info data nya.

**Table Movies**


```python
# Load file movie_rating.csv
movie_rating_df = pd.read_csv('data/movie_rating.csv')

# Print first five rows
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0000001</td>
      <td>short</td>
      <td>Carmencita</td>
      <td>Carmencita</td>
      <td>0</td>
      <td>1894.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Documentary,Short</td>
      <td>5.6</td>
      <td>1608</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0000002</td>
      <td>short</td>
      <td>Le clown et ses chiens</td>
      <td>Le clown et ses chiens</td>
      <td>0</td>
      <td>1892.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>Animation,Short</td>
      <td>6.0</td>
      <td>197</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0000003</td>
      <td>short</td>
      <td>Pauvre Pierrot</td>
      <td>Pauvre Pierrot</td>
      <td>0</td>
      <td>1892.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>Animation,Comedy,Romance</td>
      <td>6.5</td>
      <td>1285</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0000004</td>
      <td>short</td>
      <td>Un bon bock</td>
      <td>Un bon bock</td>
      <td>0</td>
      <td>1892.0</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>Animation,Short</td>
      <td>6.1</td>
      <td>121</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0000005</td>
      <td>short</td>
      <td>Blacksmith Scene</td>
      <td>Blacksmith Scene</td>
      <td>0</td>
      <td>1893.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Comedy,Short</td>
      <td>6.1</td>
      <td>2050</td>
    </tr>
  </tbody>
</table>
</div>
<br>



```python
# View info data movie_rating_df
movie_rating_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 751614 entries, 0 to 751613
    Data columns (total 11 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   tconst          751614 non-null  object 
     1   titleType       751614 non-null  object 
     2   primaryTitle    751614 non-null  object 
     3   originalTitle   751614 non-null  object 
     4   isAdult         751614 non-null  int64  
     5   startYear       751614 non-null  float64
     6   endYear         16072 non-null   float64
     7   runtimeMinutes  751614 non-null  float64
     8   genres          486766 non-null  object 
     9   averageRating   751614 non-null  float64
     10  numVotes        751614 non-null  int64  
    dtypes: float64(4), int64(2), object(5)
    memory usage: 63.1+ MB
    

Terlihat pada kolom **endYear dan genres** terdapat missing values karena jumlah data nya lebih sedikit dari keseluruhan data.

**Table Actors**


```python
# Load file actor_name.csv
actor_df = pd.read_csv('data/actor_name.csv')

# Print five first rows
actor_df.head()
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
      <th>nconst</th>
      <th>primaryName</th>
      <th>birthYear</th>
      <th>deathYear</th>
      <th>primaryProfession</th>
      <th>knownForTitles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm1774132</td>
      <td>Nathan McLaughlin</td>
      <td>1973</td>
      <td>\N</td>
      <td>special_effects,make_up_department</td>
      <td>tt0417686,tt1713976,tt1891860,tt0454839</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm10683464</td>
      <td>Bridge Andrew</td>
      <td>\N</td>
      <td>\N</td>
      <td>actor</td>
      <td>tt7718088</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm1021485</td>
      <td>Brandon Fransvaag</td>
      <td>\N</td>
      <td>\N</td>
      <td>miscellaneous</td>
      <td>tt0168790</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm6940929</td>
      <td>Erwin van der Lely</td>
      <td>\N</td>
      <td>\N</td>
      <td>miscellaneous</td>
      <td>tt4232168</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm5764974</td>
      <td>Svetlana Shypitsyna</td>
      <td>\N</td>
      <td>\N</td>
      <td>actress</td>
      <td>tt3014168</td>
    </tr>
  </tbody>
</table>
</div>




```python
# View info data actor_df
actor_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 6 columns):
     #   Column             Non-Null Count  Dtype 
    ---  ------             --------------  ----- 
     0   nconst             1000 non-null   object
     1   primaryName        1000 non-null   object
     2   birthYear          1000 non-null   object
     3   deathYear          1000 non-null   object
     4   primaryProfession  891 non-null    object
     5   knownForTitles     1000 non-null   object
    dtypes: object(6)
    memory usage: 47.0+ KB
    

Terlihat pada **kolom birthYear dan deathYear** terdapat nilai `\\N` yang berarti bernilai **NULL** karena kesalahan pembacaan data. Lalu pada info data kolom **primaryProfession** juga terdapat *missing values* karena jumlah datanya lebih sedikit dibandingkan keseluruhan data.

**Table Director_Writers**


```python
# Load file directors_writers.csv
director_writers_df = pd.read_csv('data/directors_writers.csv')

# Print first five rows
director_writers_df.head()
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
      <th>director_name</th>
      <th>writer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0011414</td>
      <td>David Kirkland</td>
      <td>John Emerson,Anita Loos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0011890</td>
      <td>Roy William Neill</td>
      <td>Arthur F. Goodrich,Burns Mantle,Mary Murillo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0014341</td>
      <td>Buster Keaton,John G. Blystone</td>
      <td>Jean C. Havez,Clyde Bruckman,Joseph A. Mitchell</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0018054</td>
      <td>Cecil B. DeMille</td>
      <td>Jeanie Macpherson</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0024151</td>
      <td>James Cruze</td>
      <td>Max Miller,Wells Root,Jack Jevne</td>
    </tr>
  </tbody>
</table>
</div>




```python
# View info data director_writers_df
director_writers_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 986 entries, 0 to 985
    Data columns (total 3 columns):
     #   Column         Non-Null Count  Dtype 
    ---  ------         --------------  ----- 
     0   tconst         986 non-null    object
     1   director_name  986 non-null    object
     2   writer_name    986 non-null    object
    dtypes: object(3)
    memory usage: 23.2+ KB
    

Terlihat bahwa tidak ditemukan *missing values* karena info jumlah data sesuai.

## Update Dataframe

Melakukan manipulasi data agar siap diolah lebih lanjut sesuai kebutuhan.

**Update Table Director_Writers**

Kita akan mengubah kolom **director_name dan writer_name** dari string menjadi list.


```python
# Change values string to list
director_writers_df['director_name'] = director_writers_df['director_name'].apply(lambda row: row.split(','))
director_writers_df['writer_name'] = director_writers_df['writer_name'].apply(lambda row: row.split(','))

# Print update rows data
director_writers_df.head()
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
      <th>director_name</th>
      <th>writer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0011414</td>
      <td>[David Kirkland]</td>
      <td>[John Emerson, Anita Loos]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0011890</td>
      <td>[Roy William Neill]</td>
      <td>[Arthur F. Goodrich, Burns Mantle, Mary Murillo]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0014341</td>
      <td>[Buster Keaton, John G. Blystone]</td>
      <td>[Jean C. Havez, Clyde Bruckman, Joseph A. Mitc...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0018054</td>
      <td>[Cecil B. DeMille]</td>
      <td>[Jeanie Macpherson]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0024151</td>
      <td>[James Cruze]</td>
      <td>[Max Miller, Wells Root, Jack Jevne]</td>
    </tr>
  </tbody>
</table>
</div>



**Update Table Actors**

Kita hanya akan membutuhkan kolom **nconst, primaryName, dan knownForTitles** untuk mencocokkan aktor/aktris ini dengan film yang ada. 


```python
# Selecting columns
actor_df = actor_df[['nconst','primaryName','knownForTitles']]

# Print update rows data
actor_df.head()
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
      <th>nconst</th>
      <th>primaryName</th>
      <th>knownForTitles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm1774132</td>
      <td>Nathan McLaughlin</td>
      <td>tt0417686,tt1713976,tt1891860,tt0454839</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm10683464</td>
      <td>Bridge Andrew</td>
      <td>tt7718088</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm1021485</td>
      <td>Brandon Fransvaag</td>
      <td>tt0168790</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm6940929</td>
      <td>Erwin van der Lely</td>
      <td>tt4232168</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm5764974</td>
      <td>Svetlana Shypitsyna</td>
      <td>tt3014168</td>
    </tr>
  </tbody>
</table>
</div>



Kita ingin tahu mengenai variasi dari jumlah film yang dapat dibintangi oleh seorang aktor. Tentunya seorang aktor dapat membintangi lebih dari 1 film, bukan? maka akan diperlukan untuk membuat table yang mempunyai relasi 1-1 ke masing-masing title movie tersebut. Kita akan melakukan **unnest** terhadap table tersebut. Langkah yang dilakukan antara lain,
 -   Melakukan pengecekan variasi jumlah film yang dibintangi oleh aktor.
 -   Mengubah kolom **knownForTitles** menjadi list of list.


```python
# Check counts
print(actor_df['knownForTitles'].apply(lambda x: len(x.split(','))).unique())
```

    [4 1 2 3]
    

Terlihat jumlah isi data pada kolom **knownForTitles** paling banyak adalah 4.


```python
# Change values column knownForTitles to list of list
actor_df['knownForTitles'] = actor_df['knownForTitles'].apply(lambda x: x.split(','))
# Print update rows data
actor_df.head()
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
      <th>nconst</th>
      <th>primaryName</th>
      <th>knownForTitles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm1774132</td>
      <td>Nathan McLaughlin</td>
      <td>[tt0417686, tt1713976, tt1891860, tt0454839]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm10683464</td>
      <td>Bridge Andrew</td>
      <td>[tt7718088]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm1021485</td>
      <td>Brandon Fransvaag</td>
      <td>[tt0168790]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm6940929</td>
      <td>Erwin van der Lely</td>
      <td>[tt4232168]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm5764974</td>
      <td>Svetlana Shypitsyna</td>
      <td>[tt3014168]</td>
    </tr>
  </tbody>
</table>
</div>



Karena pada data sebelumnya dapat dilihat bahwa seorang aktor dapat membintangi 1 sampai 4 film, diperlukan untuk membuat table yang mempunyai **relasi 1-1** dari aktor ke masing-masing judul film tersebut. 


```python
# Create empty list
new_df = []
for x in ['knownForTitles']:
    # Repeat index
    idx = actor_df.index.repeat(actor_df['knownForTitles'].str.len())   
    # Slicing values to create new rows
    temp_df = pd.DataFrame({x: np.concatenate(actor_df[x].values)})   
    # Change values of index
    temp_df.index = idx
    # Append to new_df
    new_df.append(temp_df)
    
# Concat the result
df_concat = pd.concat(new_df, axis=1)
# Join values with old data
unnested_df = df_concat.join(actor_df.drop(['knownForTitles'], 1), how='left')
# Convert into list
unnested_df = unnested_df[actor_df.columns.tolist()]
# Print update new rows
unnested_df.head()
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
      <th>nconst</th>
      <th>primaryName</th>
      <th>knownForTitles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm1774132</td>
      <td>Nathan McLaughlin</td>
      <td>tt0417686</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nm1774132</td>
      <td>Nathan McLaughlin</td>
      <td>tt1713976</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nm1774132</td>
      <td>Nathan McLaughlin</td>
      <td>tt1891860</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nm1774132</td>
      <td>Nathan McLaughlin</td>
      <td>tt0454839</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm10683464</td>
      <td>Bridge Andrew</td>
      <td>tt7718088</td>
    </tr>
  </tbody>
</table>
</div>



Selanjutnya, kita akan mengelompokkan isi data **primaryName** menjadi list group berdasarkan kolom **knownForTitles**.


```python
# Drop column nconst
unnested_df = unnested_df.drop(['nconst'], axis=1)
# Create empty list
new_df2 = []
for col in ['primaryName']:
    # Grouping by column knownForTitles
    temp_df2 = unnested_df.groupby(['knownForTitles'])[col].apply(list)
    # Append to empty list
    new_df2.append(temp_df2)

# Concat data
cast_df = pd.concat(new_df2, axis=1).reset_index()
# Rename columns
cast_df.columns = ['knownForTitles','cast_name']
cast_df.head()
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
      <th>knownForTitles</th>
      <th>cast_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0008125</td>
      <td>[Charles Harley]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0009706</td>
      <td>[Charles Harley]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0010304</td>
      <td>[Natalie Talmadge]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0011414</td>
      <td>[Natalie Talmadge]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0011890</td>
      <td>[Natalie Talmadge]</td>
    </tr>
  </tbody>
</table>
</div>



Terlihat bahwa ada nilai dari kolom **cast_name** yang sama tetapi memiliki nilai **knownForTitles** yang berbeda. Hal membuktikan bahwa seorang aktor/aktris pernah membintangi film yang berbeda.

## Join Tables

Tahap berikutnya adalah melakukan penggabungan tabel atau *dataframe* yang telah di update sebelumnya, yakni:
- **Inner Join** antara `cast_df` dan `movie_rating_df`  (field **knownForTitles dan tconst**)
- **Left Join** antara `cast_movies_df` dengan `director_writer_df` (field **tconst dan tconst**)


```python
# Join cast_df and movie_rating_df
cast_movies_df = pd.merge(cast_df, movie_rating_df, left_on='knownForTitles', right_on='tconst', how='inner')
# Join base_df and director_writers_df
base_df = pd.merge(cast_movies_df, director_writers_df, left_on='tconst', right_on='tconst', how='left')

# Print first five rows
base_df.head()
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
      <th>knownForTitles</th>
      <th>cast_name</th>
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
      <th>director_name</th>
      <th>writer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0011414</td>
      <td>[Natalie Talmadge]</td>
      <td>tt0011414</td>
      <td>movie</td>
      <td>The Love Expert</td>
      <td>The Love Expert</td>
      <td>0</td>
      <td>1920.0</td>
      <td>NaN</td>
      <td>60.0</td>
      <td>Comedy,Romance</td>
      <td>4.9</td>
      <td>136</td>
      <td>[David Kirkland]</td>
      <td>[John Emerson, Anita Loos]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0011890</td>
      <td>[Natalie Talmadge]</td>
      <td>tt0011890</td>
      <td>movie</td>
      <td>Yes or No</td>
      <td>Yes or No</td>
      <td>0</td>
      <td>1920.0</td>
      <td>NaN</td>
      <td>72.0</td>
      <td>NaN</td>
      <td>6.3</td>
      <td>7</td>
      <td>[Roy William Neill]</td>
      <td>[Arthur F. Goodrich, Burns Mantle, Mary Murillo]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0014341</td>
      <td>[Natalie Talmadge]</td>
      <td>tt0014341</td>
      <td>movie</td>
      <td>Our Hospitality</td>
      <td>Our Hospitality</td>
      <td>0</td>
      <td>1923.0</td>
      <td>NaN</td>
      <td>65.0</td>
      <td>Comedy,Romance,Thriller</td>
      <td>7.8</td>
      <td>9621</td>
      <td>[Buster Keaton, John G. Blystone]</td>
      <td>[Jean C. Havez, Clyde Bruckman, Joseph A. Mitc...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0018054</td>
      <td>[Reeka Roberts]</td>
      <td>tt0018054</td>
      <td>movie</td>
      <td>The King of Kings</td>
      <td>The King of Kings</td>
      <td>0</td>
      <td>1927.0</td>
      <td>NaN</td>
      <td>155.0</td>
      <td>Biography,Drama,History</td>
      <td>7.3</td>
      <td>1826</td>
      <td>[Cecil B. DeMille]</td>
      <td>[Jeanie Macpherson]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0024151</td>
      <td>[James Hackett]</td>
      <td>tt0024151</td>
      <td>movie</td>
      <td>I Cover the Waterfront</td>
      <td>I Cover the Waterfront</td>
      <td>0</td>
      <td>1933.0</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>Drama,Romance</td>
      <td>6.3</td>
      <td>455</td>
      <td>[James Cruze]</td>
      <td>[Max Miller, Wells Root, Jack Jevne]</td>
    </tr>
  </tbody>
</table>
</div>
<br>


## Data Cleaning

Setelah melakukan join table sebelumnya, sekarang hal yang akan kembali kita lakukan adalah melakukan cleaning pada data yang sudah dihasilkan. 


```python
# View info data base_df
base_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1060 entries, 0 to 1059
    Data columns (total 15 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   knownForTitles  1060 non-null   object 
     1   cast_name       1060 non-null   object 
     2   tconst          1060 non-null   object 
     3   titleType       1060 non-null   object 
     4   primaryTitle    1060 non-null   object 
     5   originalTitle   1060 non-null   object 
     6   isAdult         1060 non-null   int64  
     7   startYear       1060 non-null   float64
     8   endYear         110 non-null    float64
     9   runtimeMinutes  1060 non-null   float64
     10  genres          745 non-null    object 
     11  averageRating   1060 non-null   float64
     12  numVotes        1060 non-null   int64  
     13  director_name   986 non-null    object 
     14  writer_name     986 non-null    object 
    dtypes: float64(4), int64(2), object(9)
    memory usage: 132.5+ KB
    


```python
# Check NULL columns
null_cols = base_df.columns[base_df.isnull().any()]
base_df[null_cols].isnull().sum()
```




    endYear          950
    genres           315
    director_name     74
    writer_name       74
    dtype: int64



Dari hasil diatas diketahui bahwa:
- Kolom **knownForTitles dan tconst** memiliki nilai yang sama.
- Kolom **endYear, genres, director_name, writer_name** terdapat missing values.

Untuk mengatasi hal tersebut yang akan kita lakukan antara lain:
- Menghapus kolom **knownForTitles**
- Mengisi missing values dengan `Unknown` kecuali kolom endYear.


```python
# Drop colomn knownForTitles
base_df = base_df.drop(['knownForTitles'], axis=1)
# Fill missing values column genres with 'Unknown'
base_df['genres'] = base_df['genres'].fillna('Unknown')
# Fill missing values column director_name and writer_name with 'Unknown'
base_df[['director_name','writer_name']] = base_df[['director_name','writer_name']].fillna('Unknown')
# Convert values column genres into list
base_df['genres'] = base_df['genres'].apply(lambda x: x.split(','))
```

**Reformat Table**

Hal selanjutnya yang akan kita lakukan adalah melakukan reformat pada table `base_df`, seperti menhapus kolom yang tidak diperlukan dan mengubah nama kolom.


```python
# Drop unnecessary columns
clean_df = base_df.drop(['tconst','isAdult','endYear','originalTitle'], axis=1)
# Ordering columns
clean_df = clean_df[['primaryTitle','titleType','startYear','runtimeMinutes',
                     'genres','averageRating','numVotes','cast_name',
                     'director_name','writer_name']]
# Rename columns
clean_df.columns = ['title','type','start','duration','genres','rating','votes',
                    'cast_name','director_name','writer_name']
# Print clean data
clean_df.head()
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
      <th>title</th>
      <th>type</th>
      <th>start</th>
      <th>duration</th>
      <th>genres</th>
      <th>rating</th>
      <th>votes</th>
      <th>cast_name</th>
      <th>director_name</th>
      <th>writer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Love Expert</td>
      <td>movie</td>
      <td>1920.0</td>
      <td>60.0</td>
      <td>[Comedy, Romance]</td>
      <td>4.9</td>
      <td>136</td>
      <td>[Natalie Talmadge]</td>
      <td>[David Kirkland]</td>
      <td>[John Emerson, Anita Loos]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes or No</td>
      <td>movie</td>
      <td>1920.0</td>
      <td>72.0</td>
      <td>[Unknown]</td>
      <td>6.3</td>
      <td>7</td>
      <td>[Natalie Talmadge]</td>
      <td>[Roy William Neill]</td>
      <td>[Arthur F. Goodrich, Burns Mantle, Mary Murillo]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Our Hospitality</td>
      <td>movie</td>
      <td>1923.0</td>
      <td>65.0</td>
      <td>[Comedy, Romance, Thriller]</td>
      <td>7.8</td>
      <td>9621</td>
      <td>[Natalie Talmadge]</td>
      <td>[Buster Keaton, John G. Blystone]</td>
      <td>[Jean C. Havez, Clyde Bruckman, Joseph A. Mitc...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The King of Kings</td>
      <td>movie</td>
      <td>1927.0</td>
      <td>155.0</td>
      <td>[Biography, Drama, History]</td>
      <td>7.3</td>
      <td>1826</td>
      <td>[Reeka Roberts]</td>
      <td>[Cecil B. DeMille]</td>
      <td>[Jeanie Macpherson]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I Cover the Waterfront</td>
      <td>movie</td>
      <td>1933.0</td>
      <td>80.0</td>
      <td>[Drama, Romance]</td>
      <td>6.3</td>
      <td>455</td>
      <td>[James Hackett]</td>
      <td>[James Cruze]</td>
      <td>[Max Miller, Wells Root, Jack Jevne]</td>
    </tr>
  </tbody>
</table>
</div>



## Metadata

Untuk membuat sistem rekomendasi kita akan mengambil **metadata** yang dibutuhkan, yaitu kolom **title, cast_name, genres, director_name, dan writer_name**.


```python
# Selecting metadata
feature_df = clean_df[['title', 'cast_name', 'genres', 'director_name','writer_name']]
# Print first five rows
feature_df.head()
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
      <th>title</th>
      <th>cast_name</th>
      <th>genres</th>
      <th>director_name</th>
      <th>writer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Love Expert</td>
      <td>[Natalie Talmadge]</td>
      <td>[Comedy, Romance]</td>
      <td>[David Kirkland]</td>
      <td>[John Emerson, Anita Loos]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes or No</td>
      <td>[Natalie Talmadge]</td>
      <td>[Unknown]</td>
      <td>[Roy William Neill]</td>
      <td>[Arthur F. Goodrich, Burns Mantle, Mary Murillo]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Our Hospitality</td>
      <td>[Natalie Talmadge]</td>
      <td>[Comedy, Romance, Thriller]</td>
      <td>[Buster Keaton, John G. Blystone]</td>
      <td>[Jean C. Havez, Clyde Bruckman, Joseph A. Mitc...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The King of Kings</td>
      <td>[Reeka Roberts]</td>
      <td>[Biography, Drama, History]</td>
      <td>[Cecil B. DeMille]</td>
      <td>[Jeanie Macpherson]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I Cover the Waterfront</td>
      <td>[James Hackett]</td>
      <td>[Drama, Romance]</td>
      <td>[James Cruze]</td>
      <td>[Max Miller, Wells Root, Jack Jevne]</td>
    </tr>
  </tbody>
</table>
</div>



# Building Content Based Recommender System 

### **Step 1**

Membuat fungsi untuk menghilangkan spasi dari setiap baris dan setiap elemennya, dengan cara membuat `lower case` terlebih dahulu kemudian mengecek apakah bertipe list atau string biasa.


```python
# Step 1
def sanitize(x):
    try:
        # Check if values is list
        if isinstance(x, list):
            return [i.replace(' ','').lower() for i in x]
        # If values is string
        else:
            return [x.replace(' ','').lower()]
    except:
        print(x)
        
# Column with list group       
feature_cols = ['cast_name','genres','writer_name','director_name']

# Apply function sanitize 
for col in feature_cols:
    feature_df[col] = feature_df[col].apply(sanitize)
```

### **Step 2**

Membuat fungsi untuk membuat **metadata soup** (menggabungkan semua feature menjadi 1 bagian kalimat) untuk setiap judulnya.


```python
# Create function metadata soup
def soup_feature(x):
    return ' '.join(x['cast_name']) + ' ' + ' '.join(x['genres']) + ' ' + ' '.join(x['director_name']) + ' ' + ' '.join(x['writer_name'])

# Create new column soup
feature_df['soup'] = feature_df.apply(soup_feature, axis=1)
# Check soup result
feature_df['soup'].head()
```




    0    natalietalmadge comedy romance davidkirkland j...
    1    natalietalmadge unknown roywilliamneill arthur...
    2    natalietalmadge comedy romance thriller buster...
    3    reekaroberts biography drama history cecilb.de...
    4    jameshackett drama romance jamescruze maxmille...
    Name: soup, dtype: object



### **Step 3**

Menyiapkan `CountVectorizer (stop_words = english)` dan `fit` dengan soup yang kita buat sebelumnya. **CountVectorizer** adalah tipe paling sederhana dari *vectorizer*.

Sebagai contoh terdapat 3 text A, B, dan C, dimana text nya adalah
- A: The Sun is a star
- B: My Love is like a red, red rose
- C : Mary had a little lamb

Untuk mengkonversi teks berikut menjadi bentuk vector menggunakan **CountVectorizer**. Langkah-langkahnya adalah: 
- Menghitung ukuran dari *vocabulary*. **Vocabulary** adalah jumlah dari kata unik yang ada dari text tersebut. Maka hasil *vocabulary* dari set ketiga text tersebut adalah: *the, sun, is, a, star, my, love, like, red, rose, mary, had, little, lamb*. Secara total, ukuran *vocabulary* adalah 14. 
- Tidak **include stop words (english)**, seperti *as, is, a, the*, dan sebagainya karena kata tersebut sudah umum sekali. Dengan mengeliminasi *stop words*, maka *clean size vocabulary* kita adalah *like, little, lamb, love, mary, red, rose, sun, star* **(sorted alphabet ascending)**.

Dengan menggunakan CountVectorizer, maka hasil yang kita dapatkan adalah sebagai berikut:
- A : (0,0,0,0,0,0,0,1,1), terdiri atas **sun:1, star:1**
- B : (1,0,0,1,0,2,1,0,0), terdiri atas **like:1, love:1, red:2, rose:1**
- C : (0,1,1,0,1,0,0,0,0), terdiri atas **little:1, lamb:1, mary:1**


```python
# Create CountVectorizer
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(feature_df['soup'])

print(count_matrix.shape)
```

    (1060, 10026)
    

### **Step 4**

Membuat model *similarity* antara **count matrix**. Pada langkah ini, kita akan menghitung **score cosine similarity** dari setiap pasangan judul (berdasarkan semua kombinasi pasangan yang ada, dengan kata lain kita akan membuat 675 x 675 matrix, dimana cell di kolom i dan j menunjukkan *similarity score* antara judul i dan j. Kita dapat dengan mudah melihat bahwa matrix ini simetris dan setiap elemen pada diagonal adalah 1, karena itu adalah *similarity score* dengan dirinya sendiri.
 
Kita akan menggunakan formula **cosine similarity** untuk membuat model. Score ini sangatlah berguna dan mudah untuk dihitung. Formula untuk perhitungan **cosine similarity** antara 2 text, adalah sebagai berikut:

$cosine(x,y)=\frac{x.y^T}{||x||.||y||}$

Output yang didapat antara **range -1 sampai 1**. Score yang hampir mencapai 1 artinya kedua entitas tersebut sangatlah mirip sedangkan score yang hampir mencapai -1 artinya kedua entitas tersebut adalah beda.


```python
# Using cosine similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Print output
print(cosine_sim)
```

    [[1.         0.15430335 0.35355339 ... 0.         0.         0.13608276]
     [0.15430335 1.         0.10910895 ... 0.         0.         0.        ]
     [0.35355339 0.10910895 1.         ... 0.         0.08703883 0.09622504]
     ...
     [0.         0.         0.         ... 1.         0.         0.        ]
     [0.         0.         0.08703883 ... 0.         1.         0.10050378]
     [0.13608276 0.         0.09622504 ... 0.         0.10050378 1.        ]]
    

### **Step 5**

Selanjutnya yang harus dilakukan adalah **reverse mapping** dengan judul sebagai index nya.


```python
# Create indices from column title
indices = pd.Series(feature_df.index, index=feature_df['title']).drop_duplicates()

# Create function recommender system
def content_based_recommender(title):
    # Get index title
    idx = indices[title]
    
    # Convert array similarity score to list
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort by highest score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get similarity score (index 1 to 11)
    sim_scores = sim_scores[1:11]

    # Get index title based similarity score
    movie_indices = [i[0] for i in sim_scores]
    # Get rows with index title
    return base_df.iloc[movie_indices]

# Apply function recommender system
content_based_recommender('The Love Expert')
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
      <th>cast_name</th>
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
      <th>director_name</th>
      <th>writer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>344</th>
      <td>[Anat Dychtwald]</td>
      <td>tt0237123</td>
      <td>tvSeries</td>
      <td>Coupling</td>
      <td>Coupling</td>
      <td>0</td>
      <td>2000.0</td>
      <td>2004.0</td>
      <td>30.0</td>
      <td>[Comedy, Romance]</td>
      <td>8.5</td>
      <td>41571</td>
      <td>[Martin Dennis]</td>
      <td>[Steven Moffat]</td>
    </tr>
    <tr>
      <th>1052</th>
      <td>[Metin Namlisesli]</td>
      <td>tt9124840</td>
      <td>movie</td>
      <td>Organik Ask</td>
      <td>Organik Ask</td>
      <td>0</td>
      <td>2018.0</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>[Comedy, Romance]</td>
      <td>4.3</td>
      <td>89</td>
      <td>[Kamil Cetin]</td>
      <td>[Volkan Girgin]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Natalie Talmadge]</td>
      <td>tt0014341</td>
      <td>movie</td>
      <td>Our Hospitality</td>
      <td>Our Hospitality</td>
      <td>0</td>
      <td>1923.0</td>
      <td>NaN</td>
      <td>65.0</td>
      <td>[Comedy, Romance, Thriller]</td>
      <td>7.8</td>
      <td>9621</td>
      <td>[Buster Keaton, John G. Blystone]</td>
      <td>[Jean C. Havez, Clyde Bruckman, Joseph A. Mitc...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>[Constance De Mattiazzi]</td>
      <td>tt0043762</td>
      <td>movie</td>
      <td>Lullaby of Broadway</td>
      <td>Lullaby of Broadway</td>
      <td>0</td>
      <td>1951.0</td>
      <td>NaN</td>
      <td>92.0</td>
      <td>[Comedy, Musical, Romance]</td>
      <td>6.8</td>
      <td>893</td>
      <td>[David Butler]</td>
      <td>[Earl Baldwin]</td>
    </tr>
    <tr>
      <th>398</th>
      <td>[Wai Chi Wong]</td>
      <td>tt0308670</td>
      <td>movie</td>
      <td>Oi ching bak min bau</td>
      <td>Oi ching bak min bau</td>
      <td>0</td>
      <td>2001.0</td>
      <td>NaN</td>
      <td>101.0</td>
      <td>[Comedy, Romance]</td>
      <td>6.8</td>
      <td>47</td>
      <td>[Steven Lo]</td>
      <td>[Canny Leung, Chi Shan Leung]</td>
    </tr>
    <tr>
      <th>441</th>
      <td>[Matthew Fuchs]</td>
      <td>tt0396269</td>
      <td>movie</td>
      <td>Wedding Crashers</td>
      <td>Wedding Crashers</td>
      <td>0</td>
      <td>2005.0</td>
      <td>NaN</td>
      <td>119.0</td>
      <td>[Comedy, Romance]</td>
      <td>6.9</td>
      <td>323737</td>
      <td>[David Dobkin]</td>
      <td>[Steve Faber, Bob Fisher]</td>
    </tr>
    <tr>
      <th>142</th>
      <td>[Harvey J. Alperin]</td>
      <td>tt0094889</td>
      <td>movie</td>
      <td>Cocktail</td>
      <td>Cocktail</td>
      <td>0</td>
      <td>1988.0</td>
      <td>NaN</td>
      <td>104.0</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>5.9</td>
      <td>76694</td>
      <td>[Roger Donaldson]</td>
      <td>[Heywood Gould]</td>
    </tr>
    <tr>
      <th>325</th>
      <td>[Tim Horsely]</td>
      <td>tt0198284</td>
      <td>movie</td>
      <td>After Sex</td>
      <td>After Sex</td>
      <td>0</td>
      <td>2000.0</td>
      <td>NaN</td>
      <td>96.0</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>4.4</td>
      <td>753</td>
      <td>[Cameron Thor]</td>
      <td>[Thomas M. Kostigen]</td>
    </tr>
    <tr>
      <th>345</th>
      <td>[Ngan-Ying Poon]</td>
      <td>tt0237501</td>
      <td>movie</td>
      <td>Ninth Happiness</td>
      <td>Gau sing bou hei</td>
      <td>0</td>
      <td>1998.0</td>
      <td>NaN</td>
      <td>86.0</td>
      <td>[Comedy, Musical, Romance]</td>
      <td>5.9</td>
      <td>118</td>
      <td>[Clifton Ko]</td>
      <td>[Raymond To]</td>
    </tr>
    <tr>
      <th>410</th>
      <td>[Catherine May]</td>
      <td>tt0340109</td>
      <td>movie</td>
      <td>Fast Food High</td>
      <td>Fast Food High</td>
      <td>0</td>
      <td>2003.0</td>
      <td>NaN</td>
      <td>92.0</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>5.2</td>
      <td>174</td>
      <td>[Nisha Ganatra]</td>
      <td>[Tassie Cameron, Jackie May]</td>
    </tr>
  </tbody>
</table>
</div>
<br>


Hasil diatas adalah rekomendasi **10 film terbaik** berdasarkan kemiripan data film **The Love Expert**.
