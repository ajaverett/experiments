import streamlit as st

st.title('Data Science Rosetta Stone!')

lang_var = st.multiselect('Select a library', ['Pandas', 'PySpark', 'SQL'], default=['Pandas'])

st.write('Note this may have errors, please let me know if you find any.')

show_adv = st.toggle("Show advanced parameters", value=False)

st.subheader('Importing libraries')

with st.expander("Importing libraries", expanded=False):
    # --- PANDAS ---
    if 'Pandas' in lang_var:
        pandas_code = """# pandas
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

    # --- PYSPARK ---
    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Session
spark = SparkSession.builder.appName("app").getOrCreate()
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Spark Config optimization example
spark.conf.set("spark.sql.shuffle.partitions", "5")
"""
        st.code(pyspark_code, language="python")

# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- SQL (Structured Query Language) is not imported.
-- It is run against a database engine (Postgres, MySQL, Snowflake, SparkSQL).
"""
        st.code(sql_code, language="sql")

st.subheader('Importing datasets')

with st.expander("Creating a Series / Single Column", expanded=False):
    # --- PANDAS ---
    if 'Pandas' in lang_var:
        pandas_code = """# pandas
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

# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
SELECT * FROM (VALUES (1), (2), (3)) AS t(my_series);
"""
        st.code(sql_code, language="sql")

with st.expander("Creating a dataframe", expanded=False):
    # --- PANDAS ---
    if 'Pandas' in lang_var:
        pandas_code = """# pandas
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
    dtype={'col_two': 'int64'},                   # per-column dtype 
    copy=False,
)
"""
        st.code(pandas_code, language="python")

    # --- PYSPARK ---
    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
df = spark.createDataFrame(
    [('A', 1), ('B', 2), ('C', 3), ('D', 4)],
    ['col_one', 'col_two']
)
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Using strict Schema structure
schema = StructType([
    StructField("col_one", StringType(), True),
    StructField("col_two", IntegerType(), True)
])

df = spark.createDataFrame(
    data=[('A', 1), ('B', 2), ('C', 3), ('D', 4)],
    schema=['col_one', 'col_two']
)
"""
        st.code(pyspark_code, language="python")

# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
CREATE TABLE df (
    col_one VARCHAR(1),
    col_two INT
);

INSERT INTO df VALUES 
('A', 1), ('B', 2), ('C', 3), ('D', 4);
"""
        st.code(sql_code, language="sql")

with st.expander("Loading in a CSV file", expanded=False):
    # --- PANDAS ---
    if 'Pandas' in lang_var:
        pandas_code = """# pandas
df = pd.read_csv('data.csv')
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
df = pd.read_csv('data.csv',
    sep=',',                           # delimiter
    header=0,                          # row number for col names
    names=None,                        # custom names
    index_col='Date',                  # use 'Date' as index
    usecols=['Date', 'Region'],        # read subset of columns
    dtype={'Sales': 'float64'},        # set column dtypes
    parse_dates=['Date'],              # parse date columns
    na_values=['NA', '-'],             # treat these as missing
    encoding='utf-8',                  # file encoding
    nrows=1000                         # read first N rows
)
"""
        st.code(pandas_code, language="python")

    # --- PYSPARK ---
    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
df = spark.read.csv('data.csv', header=True, inferSchema=True)
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
df = spark\\
    .read\\
    .format("csv")\\
    .option("header", "true")\\
    .option("delimiter", ",")\\
    .option("inferSchema", "false")\\
    .option("mode", "FAILFAST")\\
    .schema(custom_schema_object)\\
    .load("data.csv")
"""
        st.code(pyspark_code, language="python")

with st.expander("Loading in an Excel file", expanded=False):
    # --- PANDAS ---
    if 'Pandas' in lang_var:
        pandas_code = """# pandas
df = pd.read_excel('data.xlsx')
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
df = pd.read_excel('data.xlsx',
    sheet_name='my_sheet_name',         
    header=0,                           
    usecols=['Date', 'Sales'],
    dtype={'Sales': 'float64'},         
    na_values=['NA', '-', ''],          
    engine='openpyxl'                   
)
"""
        st.code(pandas_code, language="python")

    # --- PYSPARK ---
    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Spark requires an external jar for Excel (com.crealytics:spark-excel)
df = spark.read.format("com.crealytics.spark.excel") \\
    .option("header", "true") \\
    .load("data.xlsx")
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
df = spark.read.format("com.crealytics.spark.excel")\\
    .option("header", "true")\\
    .option("dataAddress", "'MySheet'!A1:C100")\\
    .option("inferSchema", "true")\\
    .load("data.xlsx")
"""
        st.code(pyspark_code, language="python")

st.subheader('Describing datasets')

with st.expander("Descriptive statistics for a dataframe", expanded=False):
    # --- PANDAS ---
    if 'Pandas' in lang_var:
        pandas_code = """# pandas

df.info() 
# prints info about a DataFrame including the 
# index, dtypes, and column names, non-null values counts

df.describe()
# returns a DataFrame with descriptive statistics
"""
        st.code(pandas_code, language="python")

    # --- PYSPARK ---
    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
df.printSchema()
# prints the tree structure of the schema (dtypes)

df.describe().show()
# returns a DataFrame with descriptive statistics
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
df.summary().show()  
# like describe(), but includes quartiles (25%, 50%, 75%)
"""
        st.code(pyspark_code, language="python")

# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Get column metadata (Postgres/MySQL)
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'df';
"""
        st.code(sql_code, language="sql")

with st.expander("Descriptive statistics for a column", expanded=False):
    # --- PANDAS ---
    if 'Pandas' in lang_var:
        pandas_code = """# pandas
# Find all distinct values of a column
df['col_one'].unique()

# Value counts of categorical columns
df['col_one'].value_counts()

# Descriptive statistics for numeric columns
df['col_one'].mean()
df['col_one'].min()
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
df['col_one'].value_counts(normalize=True)  # proportions
df['col_one'].describe()                    # detailed stats
"""
        st.code(pandas_code, language="python")

    # --- PYSPARK ---
    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Find all distinct values of a column
df.select('col_one').distinct().show()

# Value counts of categorical columns
df.groupBy('col_one').count().orderBy('count', ascending=False).show()

# Descriptive statistics for numeric columns
df.select(F.mean('col_one'), F.min('col_one'), F.max('col_one')).show()
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Value counts with proportions (requires Window)
df.groupBy('col_one').count() \\
  .withColumn('percent', F.col('count')/F.sum('count').over(Window.partitionBy())) \\
  .show()

# Approx Quantiles
df.stat.approxQuantile("col_one", [0.25, 0.5, 0.75], 0.0)
"""
        st.code(pyspark_code, language="python")
    # --- SQL ---
        if 'SQL' in lang_var:
            sql_code = """-- SQL
-- # Find all distinct values of a column
SELECT DISTINCT col_one FROM df;

-- Value Counts
SELECT col_one, COUNT(*) 
FROM df 
GROUP BY col_one 
ORDER BY 2 DESC;

-- Numeric Stats
SELECT 
    AVG(col_one), 
    MIN(col_one), 
    MAX(col_one) 
FROM df;
    """
            if show_adv:
                sql_code += """
-- --- advanced ---
-- Percentiles (approximate median)
SELECT 
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col_one) 
FROM df;
"""
            st.code(sql_code, language="sql")

st.subheader('Filter a subset of columns')

with st.expander("Selecting columns", expanded=False):
    # --- PANDAS ---
    if 'Pandas' in lang_var:
        pandas_code = """# pandas
# By name
df.filter(items=['col_one'])

# By bracket
df[['col_one', 'col_two']]
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
df.filter(like='sales', axis='columns')      # contains 'sales'
df.filter(regex='_2024$', axis='columns')    # ends with '_2024'
df.iloc[:,0:3]                               # by index position
"""
        st.code(pandas_code, language="python")

    # --- PYSPARK ---
    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# By name
df.select('col_one')

# Multiple columns
df.select('col_one', 'col_two')
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Select using list comprehension (Regex equivalent)
df.select([c for c in df.columns if 'sales' in c])

# Regex selection (Spark 2.3+)
df.select(df.colRegex("`^sales_.*`"))
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- By name
SELECT col_one FROM df;

-- Multiple columns
SELECT col_one, col_two FROM df;
"""
        st.code(sql_code, language="sql")

with st.expander("Dropping columns", expanded=False):
    # --- PANDAS ---
    if 'Pandas' in lang_var:
        pandas_code = """# pandas
df.drop(columns=['col_one'])
df.drop(columns=['col_one','col_two'])
"""
        st.code(pandas_code, language="python")

    # --- PYSPARK ---
    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
df.drop('col_one')
df.drop('col_one', 'col_two')
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- In SQL, you select the columns you want to KEEP.
SELECT col_two, col_three FROM df;

-- To actually delete a column from storage:
ALTER TABLE df DROP COLUMN col_one;
"""
        st.code(sql_code, language="sql")

st.subheader('Filter a subset of rows')

if show_adv:
    st.markdown("_**Subsetting rows by index**_")

    with st.expander("Subsetting rows using **.iloc[]** (Position)", expanded=False):
        pandas_code = ''
        if 'Pandas' in lang_var:
            pandas_code += """# pandas
df.iloc[2]              # return the 3rd row
df.iloc[[0, 3, 5]]      # return the 1st, 4th, and 6th rows
df.iloc[2:6]            # return the 3rd to 6th rows
df.iloc[0:5]            # return the first 5 rows
"""
        st.code(pandas_code, language="python")

        if 'PySpark' in lang_var:
            pyspark_code = """# pyspark
# Spark DataFrames are distributed and unordered, so there is no 
# direct integer indexing like .iloc.

# Return the first n rows as a list of Row objects
df.take(5) 
df.limit(5).show()
"""
            if show_adv:
                pyspark_code += """
# --- advanced ---
# To filter by specific row numbers, you must generate an index 
# using a Window function (requires a column to order by).

w = Window.orderBy("some_date_col")
df_idx = df.withColumn("row_id", F.row_number().over(w))

# Now you can filter like .iloc
df_idx.filter(F.col("row_id") == 3)          # 3rd row
df_idx.filter(F.col("row_id").between(2, 6)) # 2nd to 6th row
"""
            st.code(pyspark_code, language="python")

            if 'SQL' in lang_var and show_adv:
                sql_code = """-- SQL
-- SQL tables are unordered sets. You must generate an index.
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER(ORDER BY date_col) as rn 
    FROM df
) sub
WHERE rn = 3;
"""
                st.code(sql_code, language="sql")

st.markdown("_**Conditional statements to filter rows**_")

with st.expander("Subsetting rows using SQL Syntax (**query/filter**)", expanded=False):
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
df.query("num_col >= @number_variable") 
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# You can pass SQL strings directly to .filter() or .where()
df.filter("num_col >= 100")
df.filter("str_col != 'Blue'")
df.filter("str_col in ('A', 'B')") # Note SQL syntax uses () for lists
df.filter("(num_col > 2 AND num_col < 8) OR (str_col == 'North')")
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Using f-strings to inject variables into SQL expressions
df.filter(f"num_col >= {number_variable}")
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
SELECT * FROM df WHERE num_col >= 100;
SELECT * FROM df WHERE str_col != 'Blue';
SELECT * FROM df WHERE str_col IN ('A', 'B');
SELECT * FROM df WHERE (num_col > 2 AND num_col < 8) OR str_col = 'North';
"""
        st.code(sql_code, language="sql")

with st.expander("Subsetting rows using Boolean Masking", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df[df["num_col"] >= 100]
df[df["str_col"] != "Blue"]
df[df["str_col"].isin(["A", "B"])]
df[(df["num_col"].between(2, 8)) | (df["str_col"] == "North")]
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
df.filter(F.col("num_col") >= 100)
df.filter(F.col("str_col") != "Blue")
df.filter(F.col("str_col").isin("A", "B"))

# Bitwise operators: & (and), | (or), ~ (not)
# Parentheses are mandatory for multiple conditions
df.filter(
    (F.col("num_col").between(2, 8)) | (F.col("str_col") == "North")
)
"""
        st.code(pyspark_code, language="python")

with st.expander("Subsetting rows by Label/Condition (**loc**)", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.loc[df["num_col"] >= 100]
df.loc[df["str_col"].isin(["A", "B"])]
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Spark does not differentiate between "loc" (label) and standard filtering
df.filter(F.col("num_col") >= 100)
df.filter(F.col("str_col").isin("A", "B"))
"""
        st.code(pyspark_code, language="python")

st.markdown("_**String related conditional statements**_")

with st.expander("Subsetting rows using SQL Syntax (strings)", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.query('str_col.str.contains("string", na=False)', engine="python")
df.query('str_col.str.startswith("string", na=False)', engine="python")
df.query('str_col.str.endswith("string", na=False)', engine="python")
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Using SQL LIKE syntax
df.filter("str_col LIKE '%string%'")
df.filter("str_col LIKE 'string%'") # startswith
df.filter("str_col LIKE '%string'") # endswith
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
SELECT * FROM df WHERE str_col LIKE '%string%'; -- contains
SELECT * FROM df WHERE str_col LIKE 'string%';  -- starts with
SELECT * FROM df WHERE str_col LIKE '%string';  -- ends with
"""
        st.code(sql_code, language="sql")

with st.expander("Subsetting rows using Boolean Masking (strings)", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df[df["str_col"].str.contains("string", na=False)]
df[df["str_col"].str.startswith("string", na=False)]
df[df["str_col"].str.endswith("string", na=False)]
df[df["str_col"].str.match(regex_pattern, na=False)]
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
df.filter(F.col("str_col").contains("string"))
df.filter(F.col("str_col").startswith("string"))
df.filter(F.col("str_col").endswith("string"))

# Regex matching (RLIKE)
df.filter(F.col("str_col").rlike(regex_pattern))
"""
        st.code(pyspark_code, language="python")

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

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Removes all rows with any null values
df.na.drop() # or df.dropna()

# Only removes rows when nulls are in specific columns
df.na.drop(subset=['col_one', 'col_two'])
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Keeps rows with at least n non-null values
df.na.drop(thresh=n)

# Drop only if ALL columns are null (default is 'any')
df.na.drop(how="all")
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Removes rows with any nulls (manual check)
SELECT * FROM df 
WHERE col_one IS NOT NULL 
  AND col_two IS NOT NULL;
"""
        st.code(sql_code, language="sql")

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
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Removes duplicate rows
df.dropDuplicates()

# Removes duplicate rows based on specific columns
df.dropDuplicates(subset=['col_one', 'col_two'])
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Spark's dropDuplicates picks an arbitrary row. 
# To enforce keep='first' or 'last', use Window functions:

w = Window.partitionBy("col_one").orderBy(F.col("date_col").desc())

# Filter for the "last" (latest date) row per group
df.withColumn("rn", F.row_number().over(w)) \\
  .filter(F.col("rn") == 1) \\
  .drop("rn")
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Removes duplicate rows
SELECT DISTINCT * FROM df;

-- Deduplicate based on specific columns (Keep first/last)
WITH CTE AS (
    SELECT *, 
    ROW_NUMBER() OVER(PARTITION BY col_one ORDER BY date_col DESC) as rn
    FROM df
)
SELECT * FROM CTE WHERE rn = 1;
"""
        st.code(sql_code, language="sql")

st.subheader('Cleaning datasets')

st.markdown("_**Renaming columns**_")

with st.expander("Rename columns by list", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas

# Rename all columns by assigning a new list to df.columns
df.columns = ["new_col_1", "new_col_2", "new_col_3"]
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Rename all columns
df = df.toDF("new_col_1", "new_col_2", "new_col_3")
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Removes duplicate rows
SELECT DISTINCT * FROM df;

-- Deduplicate based on specific columns (Keep first/last)
WITH CTE AS (
    SELECT *, 
    ROW_NUMBER() OVER(PARTITION BY col_one ORDER BY date_col DESC) as rn
    FROM df
)
SELECT * FROM CTE WHERE rn = 1;
"""
        st.code(sql_code, language="sql")

with st.expander("Rename columns mapping **(old -> new)**", expanded=False):
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

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Rename specific columns (chaining required for multiple)
df = df.withColumnRenamed("col_one", "new_col_1") \\
       .withColumnRenamed("col_two", "new_col_2")
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Bulk rename using list comprehension and select
new_cols = [F.col(c).alias(c.lower() + '_col') for c in df.columns]
df = df.select(*new_cols)
"""
        st.code(pyspark_code, language="python")

st.markdown("_**Casting data types and using data accessors**_")

with st.expander("Casting data types", expanded=False):
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
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Cast individual columns
df = df.withColumn("is_active", F.col("is_active").cast("boolean")) \\
       .withColumn("age", F.col("age").cast("integer")) \\
       .withColumn("zip", F.col("zip").cast("string"))
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Cast using selectExpr (SQL style)
df = df.selectExpr(
    "cast(race as string) as race",
    "cast(is_active as boolean) as is_active",
    "cast(age as int) as age"
)
"""
        st.code(pyspark_code, language="python")

# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
SELECT 
    CAST(is_active AS BOOLEAN),
    CAST(age AS INT),
    CAST(zip AS VARCHAR)
FROM df;

-- Postgres Shorthand
SELECT age::INT FROM df;
"""
        st.code(sql_code, language="sql")

if show_adv:
    with st.expander("Casting to Numeric", expanded=False):
        pandas_code = ''
        if 'Pandas' in lang_var:
            pandas_code += """# pandas
pd.to_numeric(df['num_col_str'], errors='coerce') # converts to numeric, setting errors to NaN
pd.to_numeric(df['num_col_str'], downcast='integer') 
    """
            st.code(pandas_code, language="python")

        if 'PySpark' in lang_var:
            pyspark_code = """# pyspark
# Cast to Double/Float (non-numeric becomes null automatically)
df.withColumn("num_col", F.col("num_col_str").cast("double"))
"""
            st.code(pyspark_code, language="python")

with st.expander("Casting to Datetime/Timestamp", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
pd.to_datetime(df['date_col_str'], format='%Y-%m-%d') # explicit format
pd.to_datetime(df['date_col_str'], errors='coerce') 
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
df['col_dt'].dt.strftime('%Y-%m-%d') # datetime to string
df['col_dt'].dt.tz_localize('UTC')   # adds tz info
df['col_dt'].dt.normalize()          # sets all times to 00:00
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Convert string to date (default format yyyy-MM-dd)
df.withColumn("date_col", F.to_date(F.col("date_str")))

# Convert with specific format
df.withColumn("date_col", F.to_date(F.col("date_str"), "yyyy-MM-dd"))
df.withColumn("ts_col", F.to_timestamp(F.col("ts_str"), "yyyy-MM-dd HH:mm:ss"))
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Date to String
df.withColumn("str_col", F.date_format(F.col("date_col"), "yyyy-MM-dd"))

# Truncate (normalize)
F.date_trunc("day", F.col("ts_col")) # sets time to 00:00:00
F.date_trunc("hour", F.col("ts_col"))
"""
        st.code(pyspark_code, language="python")


    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Standard ANSI SQL cast (assumes YYYY-MM-DD format)
SELECT CAST(date_str AS DATE) FROM df;

-- Cast string to Timestamp
SELECT CAST(ts_str AS TIMESTAMP) FROM df;

-- Postgres / SparkSQL shorthand
SELECT date_str::DATE, ts_str::TIMESTAMP FROM df;
"""
        if show_adv:
            sql_code += """
-- --- advanced ---
-- Parsing custom formats (Engine specific!)

-- Postgres / SparkSQL:
SELECT TO_DATE('02/14/2024', 'MM/DD/YYYY');
SELECT TO_TIMESTAMP('2024-02-14 10:00', 'YYYY-MM-DD HH24:MI');

-- MySQL:
SELECT STR_TO_DATE('02/14/2024', '%m/%d/%Y');

-- SQL Server:
SELECT CONVERT(DATE, '02/14/2024', 101); -- 101 is standard US style
"""
            st.code(sql_code, language="sql")

if show_adv:
    with st.expander("Date Arithmetic (Timedelta)", expanded=False):
        pandas_code = ''
        if 'Pandas' in lang_var:
            pandas_code += """# pandas
df['time_diff'] = df['end_date'] - df['start_date']
df['week_after'] = df['start_date'] + pd.to_timedelta(7, unit='D')
"""
            st.code(pandas_code, language="python")

        if 'PySpark' in lang_var:
            pyspark_code = """# pyspark
# Date difference (returns days as integer)
df.withColumn("days_diff", F.datediff(F.col("end"), F.col("start")))

# Add/Subtract Days
df.withColumn("next_week", F.date_add(F.col("start_date"), 7))
df.withColumn("prev_week", F.date_sub(F.col("start_date"), 7))

# Advanced Interval Math (Add hours/minutes)
df.withColumn("future", F.col("ts_col") + F.expr("INTERVAL 2 HOURS"))
"""
            st.code(pyspark_code, language="python")
# --- SQL ---
        if 'SQL' in lang_var:
            sql_code = """-- SQL
-- Date Diff
SELECT end_date - start_date FROM df;
"""
            st.code(sql_code, language="sql")


with st.expander("Datetime properties (Year, Month, etc.)", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df['col'].dt.year 
df['col'].dt.month
df['col'].dt.day
df['col'].dt.dayofweek
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
df.select(
    F.year(F.col("date_col")),
    F.month(F.col("date_col")),
    F.dayofmonth(F.col("date_col")),
    F.dayofweek(F.col("date_col")) # Sunday=1, Saturday=7
)
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Add Interval
SELECT start_date + INTERVAL '7 days' FROM df;

-- Extract parts
SELECT 
    EXTRACT(YEAR FROM date_col),
    EXTRACT(MONTH FROM date_col)
FROM df;
"""
        st.code(sql_code, language="sql")



with st.expander("String manipulation", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df['col'].str.strip()           # Remove whitespace
df['col'].str.lower()           # Lowercase
df['col'].str.replace('a', 'b') # Replace text
df['col'].str[:4]               # Slicing
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
F.trim(F.col("col"))            # Remove whitespace
F.lower(F.col("col"))           # Lowercase
F.upper(F.col("col"))           # Uppercase
F.initcap(F.col("col"))         # Title Case

# Replace text (Regex)
F.regexp_replace(F.col("col"), "old", "new")

# Slicing (Note: Spark indices are 1-based)
F.substring(F.col("col"), 1, 4) # First 4 chars
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
SELECT 
    TRIM(col),
    LOWER(col),
    UPPER(col),
    REPLACE(col, 'old', 'new'),
    SUBSTRING(col, 1, 4) -- 1-based indexing
FROM df;
"""
        st.code(sql_code, language="sql")

st.markdown("_**Replacing values in a column**_")

with st.expander("Replacing values", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Replace a single value
df.replace('old_value', 'new_value')
df["col"].replace('old_value', 'new_value')

# Replace multiple values
df.replace(['old_1', 'old_2'], 'new_value')
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Replace in entire DataFrame or subset
df.replace("old_value", "new_value") 
df.replace(["old_1", "old_2"], "new_value", subset=["col"])

# Conditional replacement (The "Spark Way")
df.withColumn("col", 
    F.when(F.col("col") == "old_value", "new_value")
     .otherwise(F.col("col"))
)
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
SELECT 
    CASE 
        WHEN col = 'old_value' THEN 'new_value'
        ELSE col 
    END AS col
FROM df;
"""
        st.code(sql_code, language="sql")


with st.expander("Replacing null values", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Replace all nulls in dataframe with x
df.fillna(x) 

# Replace nulls in a specific column with x
df['col_one'].fillna(x)
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
df['col_one'].ffill() # Forward fill
df['col_one'].bfill() # Backward fill
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Replace all nulls (type-safe: 0 for ints, "NA" for strings)
df.na.fill("NA") 
df.na.fill(0)

# Replace in specific columns
df.na.fill(0, subset=["col_one", "col_two"])
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Forward/Backward fill requires Window functions in Spark
from pyspark.sql.window import Window

w = Window.orderBy("date_col").rowsBetween(Window.unboundedPreceding, 0)
df.withColumn("filled", F.last("col_one", ignorenulls=True).over(w))
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Coalesce returns the first non-null value
SELECT COALESCE(col_one, 0) FROM df;
"""
        st.code(sql_code, language="sql")

st.markdown("_**Arranging rows in a certain order**_")

with st.expander("Sort rows", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.sort_values('col_one')
df.sort_values('col_one', ascending=False)
df.sort_values(['col_one', 'col_two'], ascending=[True, False]) 
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
df.orderBy(F.col("col_one"))
df.orderBy(F.col("col_one").desc())

# Sort by multiple columns
df.orderBy(F.col("col_one").asc(), F.col("col_two").desc())
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Null positioning
df.orderBy(F.col("col_one").asc_nulls_first())
df.orderBy(F.col("col_one").asc_nulls_last())
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
SELECT * FROM df ORDER BY col_one ASC;
SELECT * FROM df ORDER BY col_one DESC;
SELECT * FROM df ORDER BY col_one ASC, col_two DESC;
"""
        if show_adv:
            sql_code += """
-- --- advanced ---
-- Null positioning
SELECT * FROM df ORDER BY col_one ASC NULLS FIRST;
"""
        st.code(sql_code, language="sql")

st.subheader('Reshaping datasets')

st.markdown("_**Grouping and aggregating data**_")

with st.expander("Group and aggregate data", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.groupby('Race', as_index=False).size()

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
# Named Aggregation
(df.groupby(['Race', 'Sex'], as_index=False)
   .agg(
      new_col1=pd.NamedAgg(column = 'Income', aggfunc = np.median),
      new_col2=pd.NamedAgg(column = 'id', aggfunc = 'count')
))
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Count rows per group
df.groupBy("Race").count().show()

# Aggregations (using F imports)
df.groupBy("Race").agg(
    F.expr("percentile_approx(Income, 0.5)").alias("median_income"),
    F.count("id").alias("count_id"),
    F.mean("Age").alias("mean_age")
).show()
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Group by multiple columns
df.groupBy("Race", "Sex").agg(
    F.sum("Income").alias("total_income")
)
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
SELECT Race, COUNT(*) 
FROM df 
GROUP BY Race;

SELECT 
    Race, 
    AVG(Age) as mean_age, 
    COUNT(id) as count_id
FROM df
GROUP BY Race;
"""
        st.code(sql_code, language="sql")
st.markdown("_**Pivoting data**_")

with st.expander("Pivot longer: Reshape wide to long (Unpivot)", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.melt(
    id_vars=['id_col'],             # The columns to keep
    value_vars=['col1','col2'],     # The columns to stack
    var_name='category_col',        # Name of new header column
    value_name='value_col'          # Name of new value column
)
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Spark has no direct .melt(). We use the SQL 'stack' function.
# This creates two new columns: 'category_col' and 'value_col'

df.selectExpr(
    "id_col",
    "stack(2, 'col1', col1, 'col2', col2) as (category_col, value_col)"
)
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Pivot Longer (Unpivot) -> UNION ALL
SELECT id, 'col1' as category, col1 as value FROM df
UNION ALL
SELECT id, 'col2' as category, col2 as value FROM df;
"""
        st.code(sql_code, language="sql")

with st.expander("Pivot wider: Reshape long to wide", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.pivot_table(
    index=['col_staying'],
    columns='col_pivoting',
    values='val_pivoting',
    aggfunc='mean'
)
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# 1. Group by the column staying put
# 2. Pivot the column containing new headers
# 3. Aggregate the values

df.groupBy("col_staying") \\
  .pivot("col_pivoting") \\
  .mean("val_pivoting")
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Optimization: Provide list of distinct values to pivot
# (avoids an extra job to compute distinct values)
df.groupBy("col_staying") \\
  .pivot("col_pivoting", ["A", "B", "C"]) \\
  .sum("val_pivoting")
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Pivot Wider -> Conditional Aggregation
SELECT 
    col_staying,
    SUM(CASE WHEN col_pivoting = 'A' THEN val_pivoting END) AS A,
    SUM(CASE WHEN col_pivoting = 'B' THEN val_pivoting END) AS B
FROM df
GROUP BY col_staying;
"""
        st.code(sql_code, language="sql")
st.markdown("_**Joining dataframes**_")

with st.expander("Concatenating dataframes", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
pd.concat([df1,df2], axis="index")   # vertical (stack rows)
pd.concat([df1,df2], axis="columns") # horizontal (stack cols)
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Vertical Stack (Union)
df1.union(df2) 

# Vertical Stack by Name (resolves column order automatically)
df1.unionByName(df2)
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Horizontal stacking is not native to distributed dataframes 
# because row order is not guaranteed. 
# You must join on a common ID or generated Row Index.
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Vertical (Union)
SELECT * FROM df1 UNION ALL SELECT * FROM df2;
"""
        st.code(sql_code, language="sql")


with st.expander("Join two dataframes", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Left/Inner/Outer
df1.merge(df2, left_on='id1', right_on='id2', how='left')
df1.merge(df2, left_on='id1', right_on='id2', how='inner')
df1.merge(df2, left_on='id1', right_on='id2', how='outer')
"""
        if show_adv:
            pandas_code += """
# --- advanced ---
# Left anti join (Rows in df1 that are NOT in df2)
(df1.merge(df2, on='id', how='left', indicator=True)
    .query('_merge == "left_only"')
    .drop(columns='_merge'))
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# Left/Inner/Outer
df1.join(df2, df1.id1 == df2.id2, how='left')
df1.join(df2, df1.id1 == df2.id2, how='inner')
df1.join(df2, df1.id1 == df2.id2, how='outer')

# If column names are identical, pass a string to avoid duplicated columns
df1.join(df2, "common_id", how="left")
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Left anti join (native in Spark)
df1.join(df2, "id", how="left_anti")

# Cross join (Cartesian product)
df1.crossJoin(df2)
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Joins
SELECT * FROM df1 LEFT JOIN df2 ON df1.id = df2.id;
SELECT * FROM df1 INNER JOIN df2 ON df1.id = df2.id;

-- Anti Join
SELECT * FROM df1 
WHERE NOT EXISTS (SELECT 1 FROM df2 WHERE df1.id = df2.id);
"""
        st.code(sql_code, language="sql")

st.subheader('Augmenting datasets')

with st.expander("Creating new columns", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
# Bracket notation
df['new_col'] = df['col1'] + df['col2']
df['is_adult'] = np.where(df['age'] >= 18, True, False)

# .assign()
df = df.assign(new_col = df['col1'] * 2)

# .apply() (Slow, but flexible)
df['age_group'] = df['age'].apply(lambda x: 'child' if x < 18 else 'adult')
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# .withColumn() is the standard way to add/replace columns
df = df.withColumn("new_col", F.col("col1") + F.col("col2"))

# Conditional logic (equivalent to np.where)
df = df.withColumn("is_adult", 
    F.when(F.col("age") >= 18, True).otherwise(False)
)
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# UDFs (User Defined Functions) - Comparable to .apply()
# Note: Python UDFs can be slow; use native Spark functions if possible.
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def categorize_age(age):
    if age < 18: return 'child'
    elif age < 65: return 'adult'
    else: return 'senior'

df = df.withColumn("age_group", categorize_age(F.col("age")))
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
SELECT 
    *,
    col1 + col2 AS new_col,
    CASE WHEN age >= 18 THEN 1 ELSE 0 END AS is_adult
FROM df;
"""
        st.code(sql_code, language="sql")

st.subheader('Saving and exporting datasets')

with st.expander("Saving and exporting datasets", expanded=False):
    pandas_code = ''
    if 'Pandas' in lang_var:
        pandas_code += """# pandas
df.to_csv('filename.csv', index=False)
df.to_excel('filename.xlsx', index=False)
df.to_pickle('filename.pkl')
"""
        st.code(pandas_code, language="python")

    if 'PySpark' in lang_var:
        pyspark_code = """# pyspark
# CSV
df.write.csv('filename.csv', header=True, mode='overwrite')

# Parquet (Preferred for Spark)
df.write.parquet('filename.parquet', mode='overwrite')
"""
        if show_adv:
            pyspark_code += """
# --- advanced ---
# Partitioning (creates folder structure /date=2024-01-01/...)
df.write.partitionBy("date_col").parquet("data/")

# Save to Table (Hive/Delta)
df.write.saveAsTable("database.table_name")
"""
        st.code(pyspark_code, language="python")
# --- SQL ---
    if 'SQL' in lang_var:
        sql_code = """-- SQL
-- Create new table from results
CREATE TABLE new_table AS
SELECT * FROM df;

-- Insert into existing
INSERT INTO existing_table
SELECT * FROM df;
"""
        st.code(sql_code, language="sql")