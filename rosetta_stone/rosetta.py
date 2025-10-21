import streamlit as st

st.title('Data Science Rosetta Stone!')
lang_var = st.multiselect('Select a library', ['Pandas'], default=['Pandas'])
st.write('Note this may have errors, please let me know if you find any.')

show_adv = st.toggle("Show advanced parameters", value=False)

st.subheader('Importing libraries')

with st.expander("Importing libraries", expanded=False):

    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
import pandas as pd
import numpy as np
    """
        if show_adv:
            pandas_code += """
# --- advanced ---
import time
import datetime as dt
import math
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import sklearn
"""
    st.code(pandas_code, language="python")

st.subheader('Importing datasets')

with st.expander("Creating a series with **pd.Series()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
series = pd.Series([1, 2, 3])
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
series = pd.Series(
    data=[1, 2, 3],
    index=['a', 'b', 'c'],     # custom index
    dtype='int64',             # explicit dtype
    name='my_series',          # series name
    copy=False,                # avoid copying where possible
)
"""
    st.code(pandas_code, language="python")

with st.expander("Creating a dataframe with **pd.DataFrame()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df = pd.DataFrame({
    'col_one': ['A', 'B', 'C', 'D'],
    'col_two': [1, 2, 3, 4]
})
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
df = pd.DataFrame(
    data={'col_one': ['A', 'B', 'C', 'D'], 'col_two': [1, 2, 3, 4]},
    index=pd.Index([0, 1, 2, 3], name='row_id'),  # named index
    dtype={'col_two': 'int64'},                   # per-column dtype (via astype after creation in most cases)
    copy=False,
)
"""
    st.code(pandas_code, language="python")

with st.expander("Loading in a CSV file as a dataframe with **.read_csv()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df = pd.read_csv('data.csv')
    """
        if show_adv:
            pandas_code += """
# --- advanced ---
df = pd.read_csv('data.csv',
    sep=',',                       # delimiter
    header=0,                      # row number to use as the column names (0-indexed)
    names=None,                    # or provide custom names list to override header
    index_col='Date',              # use 'Date' as index
    usecols=['Date', 'Region', 'Sales'],  # read only a subset of columns
    dtype={'Sales': 'float64'},    # set column dtypes
    parse_dates=['Date'],          # parse date columns
    dayfirst=False,                # date parsing option
    na_values=['NA', '-'],         # treat these as missing
    keep_default_na=True,          # keep default NaN strings too
    encoding='utf-8',              # file encoding
    nrows=1000,                    # read first N rows
    skiprows=0,                    # skip initial rows
    engine='python'                # engine (python/c) - python supports regex sep
)
"""
    st.code(pandas_code, language="python")

with st.expander("Loading in an Excel file as a dataframe with **.read_excel()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df = pd.read_excel('data.xlsx')
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
df = pd.read_excel('data.xlsx',
    sheet_name='my_sheet_name',         # sheet name (or index, or list of names)
    header=0,                           # row number to use as header (0-indexed)
    names=['Date', 'Region', 'Sales'],  # custom column names (overrides header)
    index_col='Date',                   # use the 'Date' column as index
    usecols=['Date', 'Region', 'Sales'],# subset of columns
    dtype={'Sales': 'float64'},         # set dtypes
    parse_dates=['Date'],               # parse dates
    na_values=['NA', '-', ''],          # strings recognized as NaN
    skiprows=[0],                       # skip rows by index
    nrows=100,                          # read only first N rows
    engine='openpyxl'                   # engine for .xlsx
)
"""
    st.code(pandas_code, language="python")

st.subheader('Describing datasets')

with st.expander("Descriptive statistics for a dataframe with **.info()** and **.describe()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas

df.info() 
# prints info about a DataFrame including the 
# index, dtypes, and column names, non-null values counts, and memory usage

df.describe()
# returns a DataFrame with descriptive statistics
# for all numeric columns (count, mean, std, min, 25%, 50%, 75%, max)
"""

    st.code(pandas_code, language="python")

with st.expander("Descriptive statistics for a column", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Find all distinct values of a column
df['col_one'].unique()

# Value counts of categorical columns
df['col_one'].value_counts()
"""
        if show_adv:
            pandas_code += """df['col_one'].value_counts(normalize=True)  # Value counts of categorical columns (proportions)
df['col_one'].value_counts(bins=5)          # Value counts of binned numeric columns

# prints info about a Series (index, dtype, and name, counts)
df['col_one'].info()

# returns a Series with descriptive statistics (count, unique, top, freq)
df['col_one'].describe()
"""
    pandas_code += """
# Descriptive statistics for numeric columns
df['col_one'].mean()
df['col_one'].median()
df['col_one'].std()
df['col_one'].min()
df['col_one'].max()
"""

    st.code(pandas_code, language="python")

st.subheader('Filter a subset of columns')

with st.expander("Selecting columns using **.filter()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.filter(items=['col_one'])
df.filter(items=['col_one', 'col_two', 'col_three'])
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
df.filter(like='sales', axis='columns')      # columns that contain 'sales' in the name
df.filter(regex='_2024$', axis='columns')    # columns that end with '_2024'
df.filter(regex='^sales_', axis='columns')   # columns that start with 'sales_'
df.filter(regex='A|B', axis='columns')       # columns that contain either 'A' or 'B' (regex OR)
"""
    st.code(pandas_code, language="python")

with st.expander("Selecting columns using brackets **[]**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df['col_one']
df[['col_one', 'col_two', 'col_three']]
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
# Assuming a df with columns: col_one, col_two, col_three
# the following are equivalent:

# select the first column
# select the first three columns
# select columns from col_one to col_three (inclusive)

df.loc[:, 'col_one']
df.loc[:, ['col_one', 'col_two', 'col_three']]
df.loc[:,'col_one':'col_three'] 

df.iloc[:,0]
df.iloc[:,[0,1,2]]
df.iloc[:,0:3]
"""
    st.code(pandas_code, language="python")

with st.expander("Dropping columns using **.drop()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.drop(columns=['col_one'])
df.drop(columns=['col_one','col_two'])
"""
    st.code(pandas_code, language="python")

st.subheader('Filter a subset of rows')

if show_adv:
    st.markdown("_**Subsetting rows by index**_")

    with st.expander("Subsetting rows using **.iloc[]**", expanded=False):
        pandas_code = ''
        if 'Pandas' in lang_var:
            pandas_code += """# pandas
    df.iloc[2]              # return the 3rd row as Series
    df.iloc[[0, 3, 5]]      # return the 1st, 4th, and 6th rows
    df.iloc[2:6]            # return the 3rd to 6th rows
    df.iloc[0:5]            # return the first 5 rows
    df.iloc[-3:]            # return the last 3 rows
    """
        st.code(pandas_code, language="python")

st.markdown("_**Conditional statements to filter rows**_")

with st.expander("Subsetting rows using **.query()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.query("num_col >= 100")
df.query("str_col != 'Blue'")
df.query("str_col in ['A', 'B']")
df.query('(num_col > 2 and num_col < 8) or (str_col == "North")')
"""
        if show_adv:
            pandas_code += """
# --- advanced ---

# variables in df.query() must be prefixed with @
df.query("num_col >= @number_variable")  # using a variable
df.query("str_col in @list_variable")  # using a variable


"""
    st.code(pandas_code, language="python")

with st.expander("Subsetting rows using boolean masking **df[df[]]**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df[df["num_col"] >= 100]
df[df["str_col"] != "Blue"]
df[df["str_col"].isin(["A", "B"])]
df[(df["num_col"].between(2, 8)) | (df["str_col"] == "North")]
"""
        if show_adv:
            pandas_code += """
# --- advanced ---

df[df["num_col"] >= number_variable]
df[df["str_col"].isin(list_variable)]

"""
    st.code(pandas_code, language="python")

with st.expander("Subsetting rows using **.loc[]**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.loc[df["num_col"] >= 100]
df.loc[df["str_col"] != "Blue"]
df.loc[df["str_col"].isin(["A", "B"])]
df.loc[(df["num_col"].between(2, 8)) | (df["str_col"] == "North")]
"""
        if show_adv:
            pandas_code += """
# --- advanced ---

df.loc[df["num_col"] >= number_variable]
df.loc[df["str_col"].isin(list_variable)]

"""
    st.code(pandas_code, language="python")

st.markdown("_**String related conditional statements**_")

with st.expander("Subsetting rows using **.query()** (strings)", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.query('str_col.str.contains("string", na=False)', engine="python")
df.query('str_col.str.startswith("string", na=False)', engine="python")
df.query('str_col.str.endswith("string", na=False)', engine="python")
df.query('str_col.str.match(regex_pattern, na=False)', engine="python")
"""
    st.code(pandas_code, language="python")

with st.expander("Subsetting rows using boolean masking **df[df[]]** (strings)", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df[df["str_col"].str.contains("string", na=False)]
df[df["str_col"].str.startswith("string", na=False)]
df[df["str_col"].str.endswith("string", na=False)]
df[df["str_col"].str.match(regex_pattern, na=False)]
"""
    st.code(pandas_code, language="python")

st.markdown("_**Null related conditional statements**_")

with st.expander("Subsetting rows using **.dropna()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Removes all rows with any null values
df.dropna()

# Only removes rows when nulls are in specific columns
df.dropna(subset=['col_one', 'col_two']) 
"""
    if show_adv:
            pandas_code += """
# --- advanced ---
# Keeps rows with at least n non-null values
df.dropna(thresh=n) 
"""
    st.code(pandas_code, language="python")

st.markdown("_**Duplicate related conditional statements**_")

with st.expander("Subsetting rows using **.drop_duplicates()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Removes duplicate rows
df.drop_duplicates()

# Removes duplicate rows based on specific columns
df.drop_duplicates(subset=['col_one', 'col_two'])
"""
    if show_adv:
            pandas_code += """
# --- advanced ---

# Keeps the first occurrence of duplicates
df.drop_duplicates(keep='first')

# Keeps the last occurrence of duplicates
df.drop_duplicates(keep='last')

# Removes all duplicates, keeping only unique rows
df.drop_duplicates(keep=False)
"""
    st.code(pandas_code, language="python")

st.subheader('Cleaning datasets')

st.markdown("_**Renaming columns**_")

with st.expander("Rename columns with **.columns**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas

# Rename all columns by assigning a new list to df.columns
df.columns = ["new_col_1", "new_col_2", "new_col_3"]
"""
    st.code(pandas_code, language="python")

with st.expander("Rename columns with **.rename()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.rename(columns={"col_one": "new_col_1"})
df.rename(columns={"col_one": "new_col_1", 
                   "col_two": "new_col_2"})
"""
        if show_adv:
            pandas_code += """
# --- advanced ---

# rename all columns to lowercase with '_col' suffix
df.rename(columns=lambda x: x.lower() + '_col')

"""
    st.code(pandas_code, language="python")

st.markdown("_**Casting data types and using data accessors**_")

with st.expander("Casting data types with **.astype()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.astype({"race"       :'category', 
           "is_active"  :'bool',
           "age"        :'int',
           "zip"        :'string'})
"""
        if show_adv:
            pandas_code += """
# --- advanced ---

# convert all columns to string type
df.astype(str)

# convert age column to int type, ignoring errors
df.astype({'age': 'int'}, errors='ignore')
"""
    st.code(pandas_code, language="python")

if show_adv:
    with st.expander("Casting data types with **pd.to_numeric()**", expanded=False):
        pandas_code = ''
        if 'Pandas' in lang_var:
            pandas_code += """# pandas
pd.to_numeric(df['num_col_str'], errors='coerce') # converts to numeric, setting errors to NaN
pd.to_numeric(df['num_col_str'], downcast='integer') # converts column to smallest integer subtype
pd.to_numeric(df['num_col_str'], downcast='float')   # converts column to smallest float subtype
    """
        st.code(pandas_code, language="python")

with st.expander("Casting data types with **pd.to_datetime()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# To find the format of the date strings, 
# visit https://strftime.org/ or https://devhints.io/datetime

pd.to_datetime(df['date_col_str'], format='%Y-%m-%d') # explicit format

pd.to_datetime(df['date_col_str'], infer_datetime_format=True) # guess format

pd.to_datetime(df['date_col_str'], errors='coerce') 
# invalid strings become null (NaT)
"""
        if show_adv:
            pandas_code += """
# --- advanced ---

df['col_dt'].dt.strftime('%Y-%m-%d') # datetime to string in specified format
df['col_dt'].dt.tz_localize('UTC') # adds tz info, doesnt alter time
df['col_dt'].dt.tz_convert('America/New_York') # alters time to fit tz
df['col_dt'].dt.normalize()  # sets all times to 00:00
df['col_dt'].dt.floor('D')  # round down to nearest day
df['col_dt'].dt.ceil('H')   # round up to nearest hour
"""
    st.code(pandas_code, language="python")

if show_adv:
    with st.expander("Casting data types with **pd.to_timedelta()**", expanded=False):
        pandas_code = ''
        if 'Pandas' in lang_var:
            pandas_code += """# pandas

# Timedelta type will usually manifest when subtracting two datetime cols 
# (time_diff: timedelta type, start_date/end_date: datetime type):
df['time_diff'] = df['end_date'] - df['start_date']

# You can do arithmetic operations on datetime columns with timedelta type:
df['week_after_start_date'] = df['start_date'] + pd.to_timedelta(7, unit='D')

# You can convert int columns into timedelta type:
pd.to_timedelta(df["col_milliseconds_int"], unit="ms")
pd.to_timedelta(df["col_seconds_int"],      unit="s") 
pd.to_timedelta(df["col_minutes_int"],      unit="min")
pd.to_timedelta(df["col_hours_int"],        unit="h")
pd.to_timedelta(df["col_days_int"],         unit="D")
pd.to_timedelta(df["col_weeks_int"],        unit="W")
"""
        st.code(pandas_code, language="python")

if show_adv:
    with st.expander("Casting data types with **pd.Categorical()**", expanded=False):
        pandas_code = ''
        if 'Pandas' in lang_var:
            pandas_code += """# pandas

pd.Categorical(df['col_str'], categories=['low', 'med', 'high'], ordered=True)
# turns a string column into an ordered categorical column

df['col_cat'].cat.reorder_categories(['low', 'medium', 'high'], ordered=True)
# reorders categories of an existing categorical column
"""
        st.code(pandas_code, language="python")

with st.expander("Datetime type accessor **.dt**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df['col'].dt.year           # Grabs the year from the datetime
df['col'].dt.month          # Grabs the month from the datetime
df['col'].dt.day            # Grabs the day from the datetime
df['col'].dt.hour           # Grabs the hour from the datetime
df['col'].dt.minute         # Grabs the minute from the datetime
df['col'].dt.second         # Grabs the second from the datetime
df['col'].dt.day_name()     # Grabs the day name from the datetime
df['col'].dt.month_name()   # Grabs the month name from the datetime
df['col'].dt.dayofweek      # Grabs the day of the week from the datetime 
                            # (Monday=0, Sunday=6)

"""

    st.code(pandas_code, language="python")

with st.expander("String type accessor **.str**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df['col'].str.strip()                          # Remove leading/trailing spaces
df['col'].str.lower()                          # Convert to lowercase
df['col'].str.upper()                          # Convert to uppercase
df['col'].str.title()                          # Capitalize first letter of each word
df['col'].str.capitalize()                     # Capitalize first letter, rest lower
df['col'].str.replace('old', 'new')            # Replace text
df['col'].str.replace(r'\d+', '', regex=True)  # Remove digits (regex finds digits, replaces with "")
df['col'].str[:4]                              # First 4 characters
df['col'].str[-3:]                             # Last 3 characters
df['col'].str.slice(2, 6)                      # Characters from index 2-5
df['col'].str.get(0)                           # Character at position 0
"""

    st.code(pandas_code, language="python")

if show_adv:
    with st.expander("Categorical type accessor **.cat**", expanded=False):
        pandas_code = ''
        if 'Pandas' in lang_var:
            pandas_code += """# pandas
df['col'].cat.categories                    
# Get all defined categories of categorical column

df['col'].cat.codes                         
# Get integer codes of categorical column

df['col'].cat.add_categories(['new_cat'])   
# Adds a new category

df['col'].cat.remove_categories(['old_cat'])
# Removes a category

# Rename categories
df['col'].cat.rename_categories({'old_cat_name': 'new_cat_name'}) 

df['col'].cat.rename_categories({'old_cat_name_1': 'new_cat_name_1',
                                 'old_cat_name_2': 'new_cat_name_2'}) 
"""
        st.code(pandas_code, language="python")

st.markdown("_**Replacing values in a column**_")

with st.expander("Replacing values with **.replace()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Replace a single value across the entire dataframe
df.replace('old_value', 'new_value')

# Replace a single value in a specific column
df["col"].replace('old_value', 'new_value')


# Replace multiple values with the same new value across the entire dataframe
df.replace(['old_value1', 'old_value2', 'old_value3'], 'new_value')

# Replace multiple values with the same new value in a specific column
df["col"].replace(['old_value1', 'old_value2', 'old_value3'], 'new_value')
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
# Replace values in a specific column
df.replace({'col_one': {'old_value': 'new_value'}})

# Replace multiple values in specific columns
df.replace({'col_one': {'old_value1': 'new_value1', 
                        'old_value2': 'new_value2'}},
           {'col_two': {'old_value3': 'new_value3',
                        'old_value4': 'new_value4'}})

# Regex replacement in a specific column
df.replace(regex={'col_one': {'r_old_pattern': 'r_new_pattern'},
                  'col_two': {'r_old_pattern2': 'r_new_pattern2'}})
"""
    st.code(pandas_code, language="python")

with st.expander("Replacing null values with **.fillna()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Replace all nulls in dataframe with x
df.fillna(x) 

# Replace nulls in a specific column with x
df['col_one'].fillna(x)

# Replace nulls in a specific column with the mean of that column
df['col_one'].fillna(df['col_one'].mean())
"""
        if show_adv:
            pandas_code += """
# --- advanced ---

# Forward fill nulls in a specific column (the row above fills the null)
df['col_one'].fillna(method='ffill')

# Backward fill nulls in a specific column (the row below fills the null)
df['col_one'].fillna(method='bfill')
"""
    st.code(pandas_code, language="python")

st.markdown("_**Arranging rows in a certain order**_")

with st.expander("Sort rows with **.sort_values()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.sort_values('col_one') # sorts df by col_one ascending

df.sort_values('col_one', ascending=False) # sorts df by col_one descending

# sorts df by col_one ascending, then col_two descending
df.sort_values(['col_one', 'col_two'], ascending=[True, False]) 
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
# places nulls at the beginning
df.sort_values("col_one", na_position='first')  
"""
    st.code(pandas_code, language="python")

st.subheader('Reshaping datasets')

st.markdown("_**Grouping and aggregating data**_")

with st.expander("Group and aggregate data with **.groupby()** and **.agg()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.groupby('Race', as_index=False).size() # Returns the number of rows per group

df.groupby('Race', as_index=False)['Income'].median()

(df.groupby('Race', as_index=False)
   .agg({
        "Income":"median",
        "id":"count",
        "Age":"mean"
   })
)
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
# Using pd.NamedAgg to specify new agg column names
(df.groupby(['Race', 'Sex'], as_index=False)
   .agg(
      new_col1=pd.NamedAgg(column = 'Income', aggfunc = np.median),
      new_col2=pd.NamedAgg(column = 'id', aggfunc = 'count'), #any column will work
      new_col3=pd.NamedAgg(column = 'Age', aggfunc = np.mean)
))
"""
    st.code(pandas_code, language="python")

st.markdown("_**Pivoting data**_")

with st.expander("Pivot longer: Reshape wide to long with **.melt()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.melt(
    id_vars=['columns_staying_put'],
    var_name=['col1_melting','col2_melting'])
"""
    st.code(pandas_code, language="python")

with st.expander("Pivot wider: Reshape long to wide with **.pivot_table()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.pivot_table(index=['col1_staying','col2_staying'],
      columns='col_pivoting',
      values='val_pivoting',
      aggfunc='mean'
)
"""
    st.code(pandas_code, language="python")

st.markdown("_**Joining dataframes**_")

with st.expander("Concatenating dataframes with **pd.concat()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
pd.concat([df1,df2], axis="index") # vertically stack (row-wise)
pd.concat([df1,df2], axis="columns") # horizontally stack (column-wise)
"""
    st.code(pandas_code, language="python")

with st.expander("Join two dataframes with **pd.merge()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Left join on two dataframes
df1.merge(df2, left_on='df1_id', right_on='df2_id', how='left')

# Inner join on two dataframes
df1.merge(df2, left_on='df1_id', right_on='df2_id', how='inner')

# Outer join on two dataframes
df1.merge(df2, left_on='df1_id', right_on='df2_id', how='outer')
"""
        if show_adv:
            pandas_code += """
# --- advanced ---

# Left anti join on two dataframes
(df1
    .merge(df2, left_on='df1_id', right_on='df2_id', how='left', indicator=True)
    .query('_merge == "left_only"')
    .drop(columns='_merge')
)

# Right anti join on two dataframes
(df1
    .merge(df2, left_on='df1_id', right_on='df2_id', how='right', indicator=True)
    .query('_merge == "right_only"')
    .drop(columns='_merge')
)
"""
    st.code(pandas_code, language="python")



st.subheader('Augmenting datasets')

with st.expander("Creating new columns with bracket notation **[]**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Create a new column based on existing columns
df['new_col'] = df['col_one'] + df['col_two']

# Create a new column based on a condition
df['is_adult'] = np.where(df['age'] >= 18, True, False)
"""
    st.code(pandas_code, language="python")


with st.expander("Creating new columns with **.assign()**", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Create multiple new columns using assign

df = df.assign(
    new_col1 = df['col_one'] * 2,
    new_col2 = df['col_two'] + 100
)

# Create a new column based on condition
df = df.assign(
    is_senior = lambda x: np.where(x['age'] >= 65, True, False)
)

# Create a new column using a custom function with apply
def categorize_age(age):
    if age < 18:
        return 'child'
    elif age < 65:
        return 'adult'
    else:
        return 'senior'

df['age_group'] = df['age'].apply(categorize_age)
"""
    st.code(pandas_code, language="python")
