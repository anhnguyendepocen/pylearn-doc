'''
Pandas: Data Manipulation
=========================

It is often said that 80% of data analysis is spent on the cleaning and 
preparing data. To get a handle on the problem, this chapter focuses on a
small, but important, aspect of data manipulation and cleaning with Pandas.

**Sources**:
 
- Kevin Markham: https://github.com/justmarkham

- Pandas doc: http://pandas.pydata.org/pandas-docs/stable/index.html

**Data structures**

- **Series** is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.). The axis labels are collectively referred to as the index. The basic method to create a Series is to call `pd.Series([1,3,5,np.nan,6,8])`

- **DataFrame** is a 2-dimensional labeled data structure with columns of potentially different types. You can think of it like a spreadsheet or SQL table, or a dict of Series objects. It stems from the `R data.frame()` object.
'''

from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Create DataFrame
----------------
'''

columns = ['name', 'age', 'gender', 'job']

user1 = pd.DataFrame([['alice', 19, "F", "student"], 
                      ['john', 26, "M", "student"]],
    columns=columns)

user2 = pd.DataFrame([['eric', 22, "M", "student"], 
                      ['paul', 58, "F", "manager"]],
    columns=columns)

user3 = pd.DataFrame(dict(name=['peter', 'julie'],
                          age=[33, 44], gender=['M', 'F'], 
                          job=['engineer', 'scientist']))

'''
Concatenate DataFrame
---------------------
'''

user1.append(user2)
users = pd.concat([user1, user2, user3])

'''
Join DataFrame
--------------
'''

user4 = pd.DataFrame(dict(name=['alice', 'john', 'eric', 'julie'],
                          height=[165, 180, 175, 171]))

# Use intersection of keys from both frames
pd.merge(users, user4, on="name")

# Use union of keys from both frames
users = pd.merge(users, user4, on="name", how='outer')

'''
Summarizing
-----------
'''

# examine the users data
users                   # print the first 30 and last 30 rows
type(users)             # DataFrame
users.head()            # print the first 5 rows
users.tail()            # print the last 5 rows
users.describe()        # summarize all numeric columns
users.index             # "the index" (aka "the labels")
users.columns           # column names (which is "an index")
users.dtypes            # data types of each column
users.shape             # number of rows and columns
users.values            # underlying numpy array
users.info()            # concise summary (includes memory usage as of pandas 0.15.0)

# summarize all columns (new in pandas 0.15.0)
users.describe(include='all')       # describe all Series
users.describe(include=['object'])  # limit to one (or more) types

'''
Columns selection
-----------------
'''

users['gender']         # select one column
type(users['gender'])   # Series
users.gender            # select one column using the DataFrame 

# select multiple columns
users[['age', 'gender']]        # select two columns
my_cols = ['age', 'gender']     # or, create a list...
users[my_cols]                  # ...and use that list to select columns
type(users[my_cols])            # DataFrame

'''
Rows selection
--------------
'''

# iloc is strictly integer position based
df = users.copy()
df.iloc[0]     # first row
df.iloc[0, 0]  # first item of first row
df.iloc[0, 0] = 55

for i in range(users.shape[0]):
    row = df.iloc[i]
    row.age *= 100 # setting a copy, and not the original frame data.

print(df)  # df is not modified

# ix supports mixed integer and label based access.
df = users.copy()
df.ix[0]         # first row
df.ix[0, "age"]  # first item of first row
df.ix[0, "age"] = 55

for i in range(df.shape[0]):
    df.ix[i, "age"] *= 10

print(df)  # df is modified

'''
Rows selction / filtering
-------------------------
'''

# simple logical filtering
users[users.age < 20]        # only show users with age < 20
young_bool = users.age < 20  # or, create a Series of booleans...
users[young_bool]            # ...and use that Series to filter rows
users[users.age < 20].job    # select one column from the filtered results

# advanced logical filtering
users[users.age < 20][['age', 'job']]           # select multiple columns
users[(users.age > 20) & (users.gender=='M')]   # use multiple conditions
users[users.job.isin(['student', 'engineer'])]  # filter specific values

'''
Sorting
-------
'''

df = users.copy()

df.age.sort_values()                      # only works for a Series
df.sort_values(by='age')                  # sort rows by a specific column
df.sort_values(by='age', ascending=False) # use descending order instead
df.sort_values(by=['job', 'age'])         # sort by multiple columns
df.sort_values(by=['job', 'age'], inplace=True) # modify df

'''
Reshaping by pivoting
---------------------
'''

# “Unpivots” a DataFrame from wide format to long (stacked) format,
staked = pd.melt(users, id_vars="name", var_name="variable", value_name="value")
print(staked)

#     name variable      value
#0   alice      age         19
#1    john      age         26
#2    eric      age         22
#3    paul      age         58
#4   peter      age         33
#5   julie      age         44
#6   alice   gender          F
#             ...
#11  julie   gender          F
#12  alice      job    student
#             ...
#17  julie      job  scientist
#18  alice   height        165
#             ...
#23  julie   height        171

# “pivots” a DataFrame from long (stacked) format to wide format,
print(staked.pivot(index='name', columns='variable', values='value'))

#variable age gender height        job
#name                                 
#alice     19      F    165    student
#eric      22      M    175    student
#john      26      M    180    student
#julie     44      F    171  scientist
#paul      58      F    NaN    manager
#peter     33      M    NaN   engineer


'''
Quality control: dudlicate data
-------------------------------
'''

df = users.append(df.iloc[0], ignore_index=True)

print(df.duplicated())                 # Series of booleans 
# (True if a row is identical to a previous row)
df.duplicated().sum()                  # count of duplicates
df[df.duplicated()]                    # only show duplicates
df.age.duplicated()                    # check a single column for duplicates
df.duplicated(['age', 'gender']).sum() # specify columns for finding duplicates
df = df.drop_duplicates()              # drop duplicate rows

'''
Quality control: missing data
-----------------------------
'''

# missing values are often just excluded
df = users.copy()

df.describe(include='all')              # excludes missing values

# find missing values in a Series
df.height.isnull()           # True if NaN, False otherwise
df.height.notnull()          # False if NaN, True otherwise
df[df.height.notnull()]      # only show rows where age is not NaN
df.height.isnull().sum()     # count the missing values

# find missing values in a DataFrame
df.isnull()             # DataFrame of booleans
df.isnull().sum()       # calculate the sum of each column

# Strategy 1: drop missing values
df.dropna()             # drop a row if ANY values are missing
df.dropna(how='all')    # drop a row only if ALL values are missing

# Strategy2: fill in missing values
df.height.mean()
df = users.copy()

df.ix[df.height.isnull(), "height"] = df["height"].mean()

'''
Rename values
-------------
'''

df = users.copy()

print(df.columns)
df.columns = ['age', 'genre', 'travail', 'nom', 'taille']

df.travail = df.travail.map({ 'student':'etudiant',  'manager':'manager', 
                'engineer':'ingenieur', 'scientist':'scientific'})
assert df.travail.isnull().sum() == 0

'''
Dealing with outliers
---------------------
'''

size = pd.Series(np.random.normal(loc=175, size=20, scale=10))
# Corrupt the first 3 measures 
size[:3] += 500

'''
Based on parametric statistics: use the mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume random variable follows the normal distribution
Exclude data outside 3 standard-deviations:
- Probability that a sample lies within 1 sd: 68.27%
- Probability that a sample lies within 2 sd: 99.73% (68.27 + 2 * 15.73)
https://fr.wikipedia.org/wiki/Loi_normale#/media/File:Boxplot_vs_PDF.svg
'''

size_outlr_mean = size.copy()
size_outlr_mean[((size - size.mean()).abs() > 3 * size.std())] = size.mean()
print(size_outlr_mean.mean())


'''
Based on non-parametric statistics: use the median
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Median absolute deviation. Use the Median which is a robust non-parametric statistics
https://en.wikipedia.org/wiki/Median_absolute_deviation
'''

mad = 1.4826 * np.median(np.abs(size - size.median()))
size_outlr_mad = size.copy()

size_outlr_mad[((size - size.median()).abs() > 3 * mad)] = size.median()
print(size_outlr_mad.mean(), size_outlr_mad.median())


'''
Groupby
-------

'''

for grp, data in users.groupby("job"):
    print(grp, data)

'''
File I/O
--------
'''

'''
csv
~~~
'''

import tempfile, os.path
tmpdir = tempfile.gettempdir()
csv_filename = os.path.join(tmpdir, "users.csv")
users.to_csv(csv_filename, index=False)
other = pd.read_csv(csv_filename)

'''
Read csv from url
~~~~~~~~~~~~~~~~~
'''

url = 'https://raw.github.com/duchesnay/pylearn-doc/master/data/salary_table.csv'
salary = pd.read_csv(url)

'''
Excel
~~~~~
'''

xls_filename = os.path.join(tmpdir, "users.xlsx")
users.to_excel(xls_filename, sheet_name='users', index=False)

pd.read_excel(xls_filename, sheetname='users')

# Multiple sheets
with pd.ExcelWriter(xls_filename) as writer:
    users.to_excel(writer, sheet_name='users', index=False)
    df.to_excel(writer, sheet_name='salary', index=False)

pd.read_excel(xls_filename, sheetname='users')
pd.read_excel(xls_filename, sheetname='salary')

'''
Exercise
--------

Given modified Frame provided beloow, 
'''

df = users.copy()
df.ix[[0, 2], "age"] = None
df.ix[[1, 3], "gender"] = None

'''
1. Write a function  fillmissing_with_mean(df) that fill all missing value of numerical column with the mean of the current columns.

2) Save the original users and "imputed" frame in a single excel file "users.xlsx" with 2 sheets: original, imputed
'''