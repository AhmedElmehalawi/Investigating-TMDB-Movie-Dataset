# Project: Investigating TMDB Movie Dataset
   
## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
## Introduction
#### Overview
With the use of TMDB movie dataset which contains information about 10,000 movies, including user ratings and revenue, we are gonna investigate this dataset in order to answer some questions about it and extract some conclusions.
#### Questions
    - Which movies made maximum and minimum and minimum profits?
    - Who is most movie director?
    - In which year there was most profit?
    - What is most geners in movies?
    - What is the relation between profits over years?


```python
# Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df = pd.read_csv('tmdb-movies.csv')
```

<a id='wrangling'></a>
## Data Wrangling


### General Properties


```python
# The first row of df
df.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>imdb_id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>cast</th>
      <th>homepage</th>
      <th>director</th>
      <th>tagline</th>
      <th>...</th>
      <th>overview</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companies</th>
      <th>release_date</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget_adj</th>
      <th>revenue_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>135397</td>
      <td>tt0369610</td>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...</td>
      <td>http://www.jurassicworld.com/</td>
      <td>Colin Trevorrow</td>
      <td>The park is open.</td>
      <td>...</td>
      <td>Twenty-two years after the events of Jurassic ...</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Universal Studios|Amblin Entertainment|Legenda...</td>
      <td>6/9/15</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget_adj</th>
      <th>revenue_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>1.086600e+04</td>
      <td>1.086600e+04</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>1.086600e+04</td>
      <td>1.086600e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>66064.177434</td>
      <td>0.646441</td>
      <td>1.462570e+07</td>
      <td>3.982332e+07</td>
      <td>102.070863</td>
      <td>217.389748</td>
      <td>5.974922</td>
      <td>2001.322658</td>
      <td>1.755104e+07</td>
      <td>5.136436e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>92130.136561</td>
      <td>1.000185</td>
      <td>3.091321e+07</td>
      <td>1.170035e+08</td>
      <td>31.381405</td>
      <td>575.619058</td>
      <td>0.935142</td>
      <td>12.812941</td>
      <td>3.430616e+07</td>
      <td>1.446325e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000</td>
      <td>0.000065</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>1.500000</td>
      <td>1960.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>10596.250000</td>
      <td>0.207583</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>90.000000</td>
      <td>17.000000</td>
      <td>5.400000</td>
      <td>1995.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>20669.000000</td>
      <td>0.383856</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>99.000000</td>
      <td>38.000000</td>
      <td>6.000000</td>
      <td>2006.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75610.000000</td>
      <td>0.713817</td>
      <td>1.500000e+07</td>
      <td>2.400000e+07</td>
      <td>111.000000</td>
      <td>145.750000</td>
      <td>6.600000</td>
      <td>2011.000000</td>
      <td>2.085325e+07</td>
      <td>3.369710e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>417859.000000</td>
      <td>32.985763</td>
      <td>4.250000e+08</td>
      <td>2.781506e+09</td>
      <td>900.000000</td>
      <td>9767.000000</td>
      <td>9.200000</td>
      <td>2015.000000</td>
      <td>4.250000e+08</td>
      <td>2.827124e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10866 entries, 0 to 10865
    Data columns (total 21 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   id                    10866 non-null  int64  
     1   imdb_id               10856 non-null  object 
     2   popularity            10866 non-null  float64
     3   budget                10866 non-null  int64  
     4   revenue               10866 non-null  int64  
     5   original_title        10866 non-null  object 
     6   cast                  10790 non-null  object 
     7   homepage              2936 non-null   object 
     8   director              10822 non-null  object 
     9   tagline               8042 non-null   object 
     10  keywords              9373 non-null   object 
     11  overview              10862 non-null  object 
     12  runtime               10866 non-null  int64  
     13  genres                10843 non-null  object 
     14  production_companies  9836 non-null   object 
     15  release_date          10866 non-null  object 
     16  vote_count            10866 non-null  int64  
     17  vote_average          10866 non-null  float64
     18  release_year          10866 non-null  int64  
     19  budget_adj            10866 non-null  float64
     20  revenue_adj           10866 non-null  float64
    dtypes: float64(4), int64(6), object(11)
    memory usage: 1.7+ MB
    

### Data Cleaning
#### Drop Extraneous Columns


```python
extraneous_columns = ['id', 'imdb_id', 'homepage', 'tagline', 'keywords', 'overview', 'budget_adj',
       'revenue_adj', 'vote_count', 'vote_average', 'production_companies', 'cast']
df.drop(extraneous_columns, axis=1, inplace=True)

```


```python
df.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>release_date</th>
      <th>release_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>6/9/15</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dataframe for dates
dates = df[['release_year', 'release_date']].copy()
dates.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>release_year</th>
      <th>release_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>6/9/15</td>
    </tr>
  </tbody>
</table>
</div>



#### Date Foramt


```python
# Getting day and month from realeas date
dates[['month','day','bad_year']] = dates.release_date.str.split("/",expand=True) 
dates.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>release_year</th>
      <th>release_date</th>
      <th>month</th>
      <th>day</th>
      <th>bad_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>6/9/15</td>
      <td>6</td>
      <td>9</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
dates.dtypes
```




    release_year     int64
    release_date    object
    month           object
    day             object
    bad_year        object
    dtype: object




```python
dates['release_year'] = dates['release_year'].astype(str)
dates.dtypes
```




    release_year    object
    release_date    object
    month           object
    day             object
    bad_year        object
    dtype: object




```python
dates['date'] = dates['release_year'] + '-' + dates['month'] + '-' + dates['day']
dates['date'] = pd.to_datetime(dates['date'])
dates.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10866 entries, 0 to 10865
    Data columns (total 6 columns):
     #   Column        Non-Null Count  Dtype         
    ---  ------        --------------  -----         
     0   release_year  10866 non-null  object        
     1   release_date  10866 non-null  object        
     2   month         10866 non-null  object        
     3   day           10866 non-null  object        
     4   bad_year      10866 non-null  object        
     5   date          10866 non-null  datetime64[ns]
    dtypes: datetime64[ns](1), object(5)
    memory usage: 509.5+ KB
    


```python
df['release_date'] = dates['date']
```


```python
df.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>release_date</th>
      <th>release_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>2015-06-09</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10866 entries, 0 to 10865
    Data columns (total 9 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   popularity      10866 non-null  float64       
     1   budget          10866 non-null  int64         
     2   revenue         10866 non-null  int64         
     3   original_title  10866 non-null  object        
     4   director        10822 non-null  object        
     5   runtime         10866 non-null  int64         
     6   genres          10843 non-null  object        
     7   release_date    10866 non-null  datetime64[ns]
     8   release_year    10866 non-null  int64         
    dtypes: datetime64[ns](1), float64(1), int64(4), object(3)
    memory usage: 764.1+ KB
    

#### Dropping Null Values


```python
df.dropna(inplace=True)
df.isnull().sum()
```




    popularity        0
    budget            0
    revenue           0
    original_title    0
    director          0
    runtime           0
    genres            0
    release_date      0
    release_year      0
    dtype: int64




```python
df.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>director</th>
      <th>runtime</th>
      <th>genres</th>
      <th>release_date</th>
      <th>release_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>Colin Trevorrow</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>2015-06-09</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>



<a id='eda'></a>
## Exploratory Data Analysis


### Which movies made maximum and minimum profits?


```python
# Maximum Profit
df['profit'] = df['revenue'] - df['budget']
df_max = df[df['profit'] == df['profit'].max()]
print(df_max)
```

          popularity     budget     revenue original_title       director  \
    1386    9.432768  237000000  2781505847         Avatar  James Cameron   
    
          runtime                                    genres release_date  \
    1386      162  Action|Adventure|Fantasy|Science Fiction   2009-12-10   
    
          release_year      profit  
    1386          2009  2544505847  
    


```python
# Minimum Profit
df_min = df[df['profit'] == df['profit'].min()]
print(df_min)
```

          popularity     budget   revenue     original_title    director  runtime  \
    2244     0.25054  425000000  11087569  The Warrior's Way  Sngmoo Lee      100   
    
                                             genres release_date  release_year  \
    2244  Adventure|Fantasy|Action|Western|Thriller   2010-12-02          2010   
    
             profit  
    2244 -413912431  
    

> The maximum Profit was made by "Avatar" and the minimum profit was made by "The Warrior's Way".

### Who is most movie director?


```python
df.director.mode()
```




    0    Woody Allen
    dtype: object



> The director who directed the most was Woody Allen.

### In which year there was most profit?


```python
df.groupby('release_year').mean()['profit'].plot(kind='line', figsize = (10,10), color = 'orange',legend='profit')
plt.ylabel ('profit')
plt.title ('profits Vs release year')
```




    Text(0.5, 1.0, 'profits Vs release year')




<!--image  -->        
![output_31_1](https://user-images.githubusercontent.com/44305776/173207913-4c978f89-7949-4148-a22b-3a8eee70c9e1.png)



```python
df.groupby('release_year').mean()['profit'].sort_values(ascending=False)
```




    release_year
    1995    3.615205e+07
    1977    3.542111e+07
    1992    3.486006e+07
    2002    3.289090e+07
    2001    3.223294e+07
    2003    3.166685e+07
    2004    3.134685e+07
    1997    3.091145e+07
    2015    3.071459e+07
    1990    3.049428e+07
    1989    3.003873e+07
    1993    2.924024e+07
    2012    2.821746e+07
    2011    2.723087e+07
    2007    2.707149e+07
    1994    2.644686e+07
    2010    2.619791e+07
    2009    2.573738e+07
    2005    2.527149e+07
    1979    2.508738e+07
    1999    2.495749e+07
    1982    2.494628e+07
    1991    2.436366e+07
    2008    2.385986e+07
    1998    2.377864e+07
    2013    2.373097e+07
    2014    2.364166e+07
    2000    2.312390e+07
    1996    2.278054e+07
    1983    2.235527e+07
    1987    2.202119e+07
    2006    2.198420e+07
    1973    2.106891e+07
    1975    2.048207e+07
    1985    1.969492e+07
    1988    1.954308e+07
    1986    1.899376e+07
    1984    1.815536e+07
    1980    1.802772e+07
    1978    1.785819e+07
    1981    1.708352e+07
    1967    1.633801e+07
    1974    1.599065e+07
    1976    1.444374e+07
    1972    1.146127e+07
    1965    1.108219e+07
    1970    1.083150e+07
    1961    9.405909e+06
    1964    7.178539e+06
    1969    6.510580e+06
    1971    5.980247e+06
    1962    5.026804e+06
    1968    4.943435e+06
    1960    3.842127e+06
    1963    3.355103e+06
    1966    5.909106e+05
    Name: profit, dtype: float64



> The most average profits was made in 1995.

### What is most geners in movies?


```python
def split_compound_columns(column):
    """Split columns which has data like this; a|b|c
    Argument: column need to be seperated by '|' 
    Returns: Column of all seperated values;
    a
    b
    c
    """
    
    column = df[column].str.cat(sep = '|')
    splitted_column = pd.Series(column.split('|'))
    return splitted_column
```


```python
genres = split_compound_columns('genres')
genres.value_counts().plot.pie( subplots=True,figsize=(20,20), legend=True, autopct='%.1f%%',title='a')
plt.title('Movies Genres')
```




    Text(0.5, 1.0, 'Movies Genres')




<!--image  -->        
![output_36_1](https://user-images.githubusercontent.com/44305776/173207906-2c930597-5406-4530-9b4b-963caf0cf88a.png)


->> The most popular movie genres are drama, comedy, thriller and action.


```python
#plotting a histogram of the Time Duration of the movies

sns.set_style('darkgrid')

plt.rc('xtick')
plt.rc('ytick')

plt.figure(figsize=(10,7), dpi = 100)

plt.xlabel('Time Duration')
plt.ylabel('Movie Numbers')
plt.title('The Time Duration of the movies')

plt.hist(df['runtime'], rwidth = 1, bins =30)
plt.show()
```


<!--image  -->    
![output_38_0](https://user-images.githubusercontent.com/44305776/173207901-429a4657-e572-4c26-8596-5d7151c6d6cd.png)



> The time duration of most of the movies is around [100-120] min.


```python
plt.figure(figsize=(10,7), dpi = 100)

sns.boxplot(df['runtime'])

plt.show()
```

    C:\Anaconda\lib\site-packages\seaborn\_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


<!--image  -->    
![output_40_1](https://user-images.githubusercontent.com/44305776/173207894-4a2c5cdd-96dd-47d9-94ee-7d787dc4293d.png)



### What is the relation between profits over years?


```python
df.plot(x= 'release_year' ,y= 'profit' ,kind= 'scatter', color='orange', figsize=(10,10),legend='profit')
plt.title('Relation between each year realease and Profits')
```




    Text(0.5, 1.0, 'Relation between each year realease and Profits')




<!--image  -->    
![output_42_1](https://user-images.githubusercontent.com/44305776/173207877-0159ec81-1c9b-4666-bf9f-3f763f412389.png)



->>  Positive correlation between Release Year and Profit.


```python
df.corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>release_year</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>popularity</th>
      <td>1.000000</td>
      <td>0.544858</td>
      <td>0.663094</td>
      <td>0.140527</td>
      <td>0.091347</td>
      <td>0.628833</td>
    </tr>
    <tr>
      <th>budget</th>
      <td>0.544858</td>
      <td>1.000000</td>
      <td>0.734685</td>
      <td>0.193883</td>
      <td>0.117470</td>
      <td>0.569941</td>
    </tr>
    <tr>
      <th>revenue</th>
      <td>0.663094</td>
      <td>0.734685</td>
      <td>1.000000</td>
      <td>0.165239</td>
      <td>0.058068</td>
      <td>0.976165</td>
    </tr>
    <tr>
      <th>runtime</th>
      <td>0.140527</td>
      <td>0.193883</td>
      <td>0.165239</td>
      <td>1.000000</td>
      <td>-0.117172</td>
      <td>0.138113</td>
    </tr>
    <tr>
      <th>release_year</th>
      <td>0.091347</td>
      <td>0.117470</td>
      <td>0.058068</td>
      <td>-0.117172</td>
      <td>1.000000</td>
      <td>0.032752</td>
    </tr>
    <tr>
      <th>profit</th>
      <td>0.628833</td>
      <td>0.569941</td>
      <td>0.976165</td>
      <td>0.138113</td>
      <td>0.032752</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(15,10))         
sns.heatmap(df.corr(), annot=True, fmt="f",ax=ax)
plt.title('correlation matrix for DataFrame')
```




    Text(0.5, 1.0, 'correlation matrix for DataFrame')



<!--image  -->
![output_45_1](https://user-images.githubusercontent.com/44305776/173207853-b3bd2804-e988-494e-89fe-5742ed7b503d.png)    


<a id='conclusions'></a>
## Conclusions

#### Results
    1. Not always the high budget of the movie leads to gaining high profits.
    2. The most likeable genres are drama,comedy, thriller and action.
    3. The less likeable genres are tv movie, western, foreign and war.
    4. Dealing with popular actors in the cast besides a great director is a gurantee.
    

#### Limitations
    1. Some profits are negative.
    2. If we didn’t clean the data, there is no consistency in it so it’s necessary to do so.
