import streamlit as st

st.title('Data Science Rosetta Stone!')
lang_var = st.multiselect('Select a library', ('Pandas', 'Tidyverse','Polars','SQL', 'PySpark'), default='Pandas')
st.write('Note this may have errors, please let me know if you find any.')


st.subheader('Imports')
imports = ''
if 'Pandas' in lang_var: imports += '''
#pandas
import pandas as pd
'''
if 'Tidyverse' in lang_var: imports += '''
#tidyverse
library(tidyverse)
'''
if 'Polars' in lang_var: imports += '''
#polars
import polar as pl
'''
if 'PySpark' in lang_var:
    imports += '''
#pyspark
# if in Databricks, imports are natively handled
from pyspark.sql import functions as F
'''
st.code(imports)


st.subheader('Columns of a dataframe')
columns_var = ''
if 'Pandas' in lang_var: columns_var += '''
#pandas
series = pd.Series([1,2,3])
'''
if 'Tidyverse' in lang_var: columns_var += '''
#tidyverse
vector <- c(1,2,3)
'''
if 'Polars' in lang_var: columns_var += '''
#polars
series = pl.Series([1,2,3])
'''
st.code(columns_var)


st.subheader('Create a dataframe')
create_df = ''
if 'Pandas' in lang_var: create_df += '''
#pandas
df = pd.DataFrame(
      {'col_one': ['A', 'B', 'C','D'],
       'col_two': [1, 2, 3, 4]}
)
'''
if 'Tidyverse' in lang_var: create_df += '''
#tidyverse
df <- tibble(
    col_one = c('A', 'B', 'C', 'D'),
    col_two = c(1, 2, 3, 4)
)
'''
if 'Polars' in lang_var: create_df += '''
#polars
df = pl.DataFrame(
      {'col_one': ['A', 'B', 'C','D'],
       'col_two': [1, 2, 3, 4]}
)
'''
if 'SQL' in lang_var: create_df += '''
#sql
CREATE TABLE df (
    col_one CHAR(1),
    col_two INT
);

INSERT INTO df (col_one, col_two)
VALUES
    ('A', 1),
    ('B', 2),
    ('C', 3),
    ('D', 4);
'''

if 'PySpark' in lang_var: create_df += '''
#pyspark

schema = StructType([
    StructField("col_one", StringType(), True),
    StructField("col_two", IntegerType(), True)
])

data = [
    ('A', 1),
    ('B', 2),
    ('C', 3),
    ('D', 4)
]

df = spark.createDataFrame(data, schema)

'''

st.code(create_df)


st.subheader('Read CSV file into a dataframe')
read_csv = ''
if 'Pandas' in lang_var: read_csv += '''
#pandas
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', header=None)
'''
if 'Tidyverse' in lang_var: read_csv += '''
#tidyverse
df <- read_csv('data.csv')
df <- read_csv('data.csv', col_names = F)
'''
if 'Polars' in lang_var: read_csv += '''
#polars
df = pl.read_csv('data.csv')
df = pl.read_csv('data.csv', has_header=False)
'''
if 'PySpark' in lang_var: read_csv += '''
#pyspark
df = spark.read.csv('data.csv', header=True, inferSchema=True)
df_no_header = spark.read.csv('data.csv', header=False, inferSchema=True)
'''
st.code(read_csv)


st.subheader('Count how many of each value in a column')
count_df = ''
if 'Pandas' in lang_var: count_df += '''
#pandas
df['col_one'].value_counts()
'''
if 'Tidyverse' in lang_var: count_df += '''
#tidyverse
df %>% count(col_one)
'''
if 'Polars' in lang_var: count_df += '''
#polars
df['col_one'].value_counts()
'''
if 'SQL' in lang_var: count_df += '''
#sql
SELECT col_one, COUNT(*) AS count
FROM df
GROUP BY col_one;
'''
if 'PySpark' in lang_var: count_df += '''
#pyspark
df.groupBy('col_one').count()
'''
st.code(count_df)


st.subheader('Calculate statistics of a column')
stats_df = ''
if 'Pandas' in lang_var: stats_df += '''
#pandas
df['col_one'].mean()
df['col_one'].median()
df['col_one'].std()
df['col_one'].min()
df['col_one'].max()
'''
if 'Tidyverse' in lang_var: stats_df += '''
#tidyverse
df$col_one %>% mean
df$col_one %>% median
df$col_one %>% sd
df$col_one %>% min
df$col_one %>% max
'''
if 'Polars' in lang_var: stats_df += '''
#polars
df['col_one'].mean()
df['col_one'].median()
df['col_one'].std()
df['col_one'].min()
df['col_one'].max()
'''
if 'SQL' in lang_var: stats_df += '''
#pandas
SELECT AVG(col_one) FROM df;
SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY col_one) AS median FROM df;
SELECT STDDEV(col_one) AS std FROM df;
SELECT MIN(col_one) AS min_val FROM df;
SELECT MAX(col_one) AS max_val FROM df;
'''
if 'PySpark' in lang_var: stats_df += '''
#pyspark
df.agg(F.mean("col_one"))   # add .collect()[0][0] to the end to get the value
df.approxQuantile("Salary", [0.5], error) # 0.5 is the median (A smaller error means more precise but more computation)
df.agg(F.stddev("col_one"))
df.agg(F.min("col_one"))
df.agg(F.max("col_one"))
'''
st.code(stats_df)


st.subheader('Keep columns')
select_cols = ''
if 'Pandas' in lang_var: select_cols += '''
#pandas
df.filter(items=['col_one'])
df.filter(items=['col_one','col_two'])
df.filter(regex='[pt]al')
df.loc[:,df.columns.str.startswith("prefix_")]
df.loc[:,df.columns.str.endswith("_suffix")]
df.loc[:,df.columns.str.contains("_infix_")]
'''
if 'Tidyverse' in lang_var: select_cols += '''
#tidyverse
df %>% select(col_one)
df %>% select(col_one,col_two)
df %>% select(matches("[pt]al"))
df %>% select(starts_with("prefix_"))
df %>% select(ends_with("_suffix"))
df %>% select(contains("_infix_"))
'''
if 'Polars' in lang_var: select_cols += '''
#polars
df.select('col_one')
df.filter(['col_one','col_two'])
df.filter(pl.col('[pt]al'))
df.select(pl.col("^prefix_.*$"))
df.select(pl.col("^*_suffix$"))
df.select(pl.col("^.*_infix_.*$"))
'''
if 'SQL' in lang_var: select_cols += '''
#sql
SELECT col_one FROM df;
SELECT col_one, col_two FROM df;
'''
if 'PySpark' in lang_var: select_cols += '''
#pyspark
df.select('col_one').show()
df.select('col_one', 'col_two').show()
df.select([col for col in df.columns if col.startswith("prefix_")]).show()
df.select([col for col in df.columns if col.endswith("_suffix")]).show()
df.select([col for col in df.columns if "_infix_" in col]).show()
'''

st.code(select_cols)


st.subheader('Drop columns')
drop_cols = ''
if 'Pandas' in lang_var:
    drop_cols += '''
#pandas
df.drop(columns=['col_one'])
df.drop(columns=['col_one','col_two'])
'''
if 'Tidyverse' in lang_var:
    drop_cols += '''
#tidyverse
df %>% select(!col_one)
df %>% select(!c(col_one,col_two))
'''
if 'Polars' in lang_var:
    drop_cols += '''
#polars
df.drop('col_one')
df.drop(['col_one','col_two'])
'''
if 'SQL' in lang_var:
    drop_cols += '''
#sql
ALTER TABLE df DROP COLUMN col_one;
ALTER TABLE df DROP COLUMN col_one DROP COLUMN col_two;
'''
if 'PySpark' in lang_var: drop_cols += '''
#pyspark
df.drop('col_one')
df.drop('col_one', 'col_two')
'''
st.code(drop_cols)


st.subheader('Rename columns')
rename_cols = ''
if 'Pandas' in lang_var:
    rename_cols += '''
#pandas
df.rename(columns={"column_one": "new_col_1"})
df.rename(columns={"column_one": "new_col_1", 
                   "column_two": "new_col_2"})
'''
if 'Tidyverse' in lang_var:
    rename_cols += '''
#tidyverse
df %>% rename(new_col_1 = column_one)
df %>% rename(new_col_1 = column_one, 
              new_col_2 = column_two)
'''
if 'Polars' in lang_var:
    rename_cols += '''
#polars
df.rename({"column_one": "new_col_1"})
df.rename({"column_one": "new_col_1", 
           "column_two": "new_col_2"})
'''
if 'SQL' in lang_var:
    rename_cols += '''
#sql
ALTER TABLE df RENAME COLUMN column_one TO new_col_1;
ALTER TABLE df RENAME COLUMN column_one TO new_col_1 RENAME COLUMN column_two TO new_col_2;
'''
if 'PySpark' in lang_var: rename_cols += '''
#pyspark
df = df.withColumnRenamed('column_one', 'new_col_1')
df = df.withColumnsRenamed({
    "column_one": "new_col_1",
    "column_two": "new_col_2"})
'''
st.code(rename_cols)


st.subheader('Convert data types of columns')
data_type_conv = ''
if 'Pandas' in lang_var:
    data_type_conv += '''
#pandas
df.astype({"Race":'category', 
           "Age":'int64',
           "Zip":'string'})
'''
if 'Tidyverse' in lang_var:
    data_type_conv += '''
#tidyverse
df %>% mutate(Race = as.factor(Race), 
              Age = as.numeric(Age),
              Zip = as.character(Zip))
'''
if 'Polars' in lang_var:
    data_type_conv += '''
#polars
df.with_columns(pl.col('Race').cast(pl.Categorical, strict=False),
                pl.col('Age').cast(pl.Int64, strict=False),
                pl.col('Zip').cast(pl.Utf8, strict=False))
'''
if 'SQL' in lang_var:
    data_type_conv += '''
#sql
ALTER TABLE df ALTER COLUMN Age INTEGER;
ALTER TABLE df ALTER COLUMN Zip TYPE VARCHAR(max_length);
'''
if 'PySpark' in lang_var: data_type_conv += '''
#pyspark
df = df.withColumn('Race', F.col('Race').cast('string'))
df = df.withColumn('Age',  F.col('Age').cast('int'))
df = df.withColumn('Zip',  F.col('Zip').cast('string'))
'''
st.code(data_type_conv)


st.subheader('Subset/locate rows and columns')
subset_data = ''
if 'Pandas' in lang_var:
    subset_data += '''
#pandas
df.loc[:,:] #all rows and columns
df.loc[1,:] #second row and all columns
df.loc[[1,6],:] #second and seventh row and all columns
df.loc[[1:6],:] #second to seventh row and all columns
df.loc[:,['col_one']] #all rows and column_one
df.loc[:,['col_one','col_three']] #all rows and column_one and column_three
df.loc[:,'col_one':'col_three'] #all rows and column_one to column_three
df.iloc[:,[1,3]] #all rows and second and fourth column
df.iloc[:,1:3] #all rows and second to third column
'''
if 'Tidyverse' in lang_var:
    subset_data += '''
#tidyverse
df[,] #all rows and columns
df[1,] #first row and all columns
df[c(1,6),] #first and sixth row and all columns
df[c(1:6),] #first to sixth row and all columns
df[,'col_one'] #all rows and column_one
df[,c('col_one','col_three')] #all rows and column_one and column_three
df %>% select(col_one:col_three) #all rows and column_one to column_three
df[,c(1,3)] #all rows and first and third column
df[,c('1:3')] #all rows and first to third column
'''
if 'Polars' in lang_var:
    subset_data += '''
#polars
df[:,:] #all rows and columns
df[1,:] #first row and all columns
df[[1,6],:] #first and sixth row and all columns
df[1:6,:] #first to sixth row and all columns
df[:,['Survived']] #all rows and column_one
df[:,['Survived','Sex']] #all rows and column_one and column_three
df[:,'Survived':'Sex'] #all rows and column_one to column_three
df[:,[1,3]] #all rows and second and fourth column
df[:,1:3] #all rows and second to third column
'''
st.code(subset_data)


st.subheader('Filter data by row values')
filter_data = ''
if 'Pandas' in lang_var:
    filter_data += '''
#pandas
df.query("col_one >= 100")
df.query("col_one != 'Blue'")
df.query("col_one in ['A', 'B']")
df.query("Race == 'White' and Gender == 'Male'")
df.query("not (Race == 'White' and Gender == 'Male')")

df[df["col_one"] >= 100]
df[df["col_one"] != "Blue"]
df[df["col_one"].isin(['A', 'B'])]
df[df[(Race == "White") & (Gender == "Male")]]
df[df[~((Race == "White") & (Gender == "Male"))]]
'''
if 'Tidyverse' in lang_var:
    filter_data += '''
#tidyverse
df %>% filter(col_one >= 100)
df %>% filter(col_one != "Blue")
df %>% filter(col_one %in% c('A','B'))
df %>% filter(!(Race == "White" & Gender == "Male"))
'''
if 'Polars' in lang_var:
    filter_data += '''
#polars
df.filter(pl.col("col_one") >= 100)
df.filter(pl.col("col_one") != 'Blue')
df.filter(pl.col("col_one").is_in(['A', 'B']))
df.filter(~((pl.col("Race")=='White')&(pl.col("Gender")=='Male')))
'''
if 'SQL' in lang_var:
    filter_data += '''
#sql
SELECT * FROM df WHERE col_one >= 100;
SELECT * FROM df WHERE col_one <> 'Blue';
SELECT * FROM df WHERE col_one IN ('A', 'B');
SELECT * FROM df WHERE NOT (Race = 'White' AND Gender = 'Male');
'''
if 'PySpark' in lang_var: filter_data += '''
#pyspark
df.filter(F.col('col_one') >= 100)
df.filter(F.col('col_one') != 'Blue')
df.filter(F.col('col_one').isin(['A', 'B']))
df.filter(~((F.col('Race') == 'White') & (F.col('Gender') == 'Male')))
'''
st.code(filter_data)


st.subheader('Filter data by string values in rows')
filter = ''
if 'Pandas' in lang_var:
    filter += '''
#pandas
df.query('col_one.str.contains("string").values')
df.query('col_one.str.contains(["string1", "string2"]).values')
df.query('col_one.str.startswith("string").values')
df.query('col_one.str.endswith("string").values')
df.query('col_one.str.match(regex_pattern).values')
'''
if 'Tidyverse' in lang_var:
    filter += '''
#tidyverse
df %>% filter(str_detect(col_one, "string"))
df %>% filter(str_detect(col_one, c("string1", "string2")))
df %>% filter(str_starts(col_one, "string"))
df %>% filter(str_ends(col_one, "string"))
df %>% filter(str_match(col_one, regex_pattern))
'''
if 'Polars' in lang_var:
    filter += '''
#polars
df.filter(pl.col("col_one").str.contains("string"))
df.filter(pl.col("col_one").str.contains("string"|string2))
df.filter(pl.col("col_one").str.startswith("string"))
df.filter(pl.col("col_one").str.endswith("string"))
'''
if 'SQL' in lang_var:
    filter += '''
#sql
SELECT * FROM df WHERE col_one LIKE '%string%';
SELECT * FROM df WHERE col_one LIKE '%string1%' OR col_one LIKE '%string2%';
SELECT * FROM df WHERE col_one LIKE 'string%';
SELECT * FROM df WHERE col_one LIKE '%string';
SELECT * FROM df WHERE col_one ~ 'regex_pattern';
'''
if 'PySpark' in lang_var: filter += '''
#pyspark
df.filter(F.col('col_one').contains("string"))
df.filter((F.col("col_one").contains("string1")) | (F.col("col_one").contains("string2")))
df.filter(F.col('col_one').startswith("string"))
df.filter(F.col('col_one').endswith("string"))
df.filter(F.col('col_one').rlike(regex_pattern))
'''
st.code(filter)


st.subheader('Arrange dataframe by values in a column')
arrange_data = ''
if 'Pandas' in lang_var:
    arrange_data += '''
#pandas
df.sort_values('col_one')
df.sort_values('col_one', ascending=False)
'''
if 'Tidyverse' in lang_var:
    arrange_data += '''
#tidyverse
df %>% arrange(col_one)
df %>% arrange(col_one %>% desc())
'''
if 'Polars' in lang_var:
    arrange_data += '''
#polars
df.sort('col_one')
df.sort('col_one', descending=True)
'''
if 'SQL' in lang_var:
    arrange_data += '''
#sql
SELECT * FROM my_table ORDER BY col_one;
SELECT * FROM my_table ORDER BY col_one DESC;
'''
if 'PySpark' in lang_var:
    arrange_data += '''
#pyspark
df.orderBy('col_one')
df.orderBy(F.desc('col_one'))
'''
st.code(arrange_data)


st.subheader('Find distinct values in a column')
distinct = ''
if 'Pandas' in lang_var:
    distinct += '''
#pandas
df.drop_duplicates(subset = ["col_one"])
df.drop_duplicates()
'''
if 'Tidyverse' in lang_var:
    distinct += '''
#tidyverse
df %>% distinct(col_one, .keep_all = T)
df %>% distinct()
'''
if 'Polars' in lang_var:
    distinct += '''
#polars
df.unique(subset=["col_one"])
df.unique()
'''
if 'SQL' in lang_var:
    distinct += '''
#sql
SELECT DISTINCT ON (col_one) * FROM my_table;
SELECT DISTINCT * FROM my_table;
'''
if 'PySpark' in lang_var:
    distinct += '''
#pyspark
df.dropDuplicates(['col_one']).show()
df.dropDuplicates().show()
'''
st.code(distinct)

st.subheader('Replace values')
replace=''
if 'Pandas' in lang_var:
    replace += '''
#pandas
df.replace(2,"foo")
df[['col_one','col_two']].replace(2,"foo")
df['col_one'].replace(2,"foo")
'''
if 'Tidyverse' in lang_var:
    replace += '''
# tidyverse
df %>% mutate(across(everything(), ~replace(., . ==  2 , "foo")))
df %>% mutate(across(c(col_one,col_two), ~replace(., . ==  2 , "foo")))
df %>% mutate(col_one = ifelse(col_one == 2, "foo", col_one))
'''
if 'Polars' in lang_var:
    replace += '''
# polars
# ...no good option yet
'''
if 'SQL' in lang_var:
    replace += '''
#sql
UPDATE df SET col_one = 'foo' WHERE col_one = 2;
'''
if 'PySpark' in lang_var:
    replace += '''
#pyspark
df = df.withColumn('col_one', F.when(F.col('col_one') == 2, 'foo').otherwise(F.col('col_one')))
'''
st.code(replace)


st.subheader('Drop missing values')
drop_na = ''
if 'Pandas' in lang_var:
    drop_na += '''
#pandas
df.dropna()
df.dropna(subset=['col_one', 'col_two'])
df.dropna(thresh=n) #integer threshold
'''
if 'Tidyverse' in lang_var:
    drop_na += '''
#tidyverse
df %>% drop_na()
df %>% drop_na(c(col_one, col_two))
df %>% select(where(~mean(is.na(.)) < n)) #percent threshold
'''
if 'Polars' in lang_var:
    drop_na += '''
#polars
df.drop_nulls() # will not drop NaNs
df.fill_nan(None).drop_nulls() # will drop NaNs
df.drop_nulls(subset=['col_one', 'col_two']) # will not drop NaNs
df.fill_nan(None).drop_nulls(subset=['col_one', 'col_two']) # will drop NaNs
'''
if 'SQL' in lang_var:
    drop_na += '''
#sql
DELETE FROM df WHERE col_one IS NULL
'''
if 'PySpark' in lang_var:
    drop_na += '''
df.na.drop()
df.na.drop(subset=['col_one', 'col_two'])
'''
st.code(drop_na)

st.subheader('Replace missing values')
replace_missing = ''
if 'Pandas' in lang_var:
    replace_missing += '''
#pandas
df.fillna(x)
df['col_one'].fillna(x)
df['col_one'].fillna(method='ffill')
df['col_two'].fillna(df['col_two'].mean())
'''
if 'Tidyverse' in lang_var:
    replace_missing += '''
#tidyverse
df %>% replace(is.na(.), x)
df %>% mutate(col_one = ifelse(is.na(col_one), x, col_one))
df %>% fill(col_one, .direction = "up")
df %>% mutate(col_one = ifelse(is.na(col_one), mean(df$col_one, na.rm = T), col_one))
'''
if 'Polars' in lang_var:
    replace_missing += '''
#polars
df.fill_null(x) # will only fill nulls
df.fill_nan(x) # will only fill NaNs
df['col_one'].fill_null(x) # will only fill nulls
df['col_one'].fill_nan(x) # will only fill NaNs
df['col_one'].fill_null(method='forward')
df['col_two'].fill_null(fill_value=pl.col('col_two').mean())
'''
if 'SQL' in lang_var:
    replace_missing += '''
#sql
UPDATE df SET col_one = 'x' WHERE col_one IS NULL;
'''

st.code(replace_missing)

#df.fill_null(strategy='backward')
# df.fill_null(strategy='forward')


st.subheader('Group and aggregate data')
group_by = ''
if 'Pandas' in lang_var:
    group_by += '''
#pandas
df.groupby('Race', as_index=False).size() # Returns the number of rows per group

df.groupby('Race', as_index=False)['Income'].median()

(df.groupby('Race', as_index=False)
   .agg({
        "Income":"median",
        "id":"count",
        "Age":"mean"
   })
)

(df.groupby(['Race', 'Sex'], as_index=False)
   .agg(
      new_col1=pd.NamedAgg(column = 'Income', aggfunc = np.median),
      new_col2=pd.NamedAgg(column = 'id', aggfunc = 'count'), #any column will work
      new_col3=pd.NamedAgg(column = 'Age', aggfunc = np.mean)
))
'''
if 'Tidyverse' in lang_var:
    group_by += '''
#tidyverse
df %>% group_by(Race) %>% count()
df %>% group_by(Race) %>% summarize(new_col = median(Income))
df %>% group_by(Race, Sex) %>%
     summarize(
       new_col1 = median(Income),
       new_col2 = n(),
       new_col3 = mean(age)
)
'''
if 'Polars' in lang_var:
    group_by += '''
#polars
df.groupby('Race').count()
df.groupby('Race').median()['Income']
df.groupby(['Sex','Survived']).agg(
    pl.col('Income').median().alias('new_col1'),
    pl.col('id').count().alias('new_col2'), #any column will work
    pl.col('Age').mean().alias('new_col3'),
)
'''
if 'SQL' in lang_var:
    group_by += '''
#sql
SELECT Race, COUNT(*) AS count FROM df GROUP BY Race;
SELECT Race, MEDIAN(Income) FROM df GROUP BY Race;
SELECT Race, Sex, 
       MEDIAN(Income) AS new_col1, 
       COUNT(id) AS new_col2,
       AVG(Age) AS new_col3
FROM df
GROUP BY Race, Sex;
'''
if 'PySpark' in lang_var:
    group_by += '''
#pyspark
df.groupBy('Race').count().show()
df.groupBy('Race').agg(F.expr('percentile_approx(Income, 0.5)').alias('median')).show()
df.groupBy('Race', 'Sex').agg(
    F.expr('percentile_approx(Income, 0.5)').alias('new_col1'),
    F.count('id').alias('new_col2'),
    F.mean('Age').alias('new_col3')
).show()
'''
st.code(group_by)


st.subheader('Pivot longer: Reshape data from wide to long format')
pivot_longer = ''
if 'Pandas' in lang_var:
    pivot_longer += '''
#pandas
df.melt(
    id_vars='columns_staying_put',
    var_name=['col1_melting','col2_melting'])
'''
if 'Tidyverse' in lang_var:
    pivot_longer += '''
#tidyverse
df %>% pivot_longer(
     cols = c("col1_melting", "col2_melting")
)
'''
if 'Polars' in lang_var:
    pivot_longer += '''
#polars
df.melt(
    id_vars='col_staying',
    value_vars=['col1_melting','col2_melting'])

'''
if 'PySpark' in lang_var:
    pivot_longer += '''
#pyspark
df.melt(
    ids=['col_staying'], 
    values=['col1_melting', 'col2_melting'],
    variableColumnName='variable',
    valueColumnName='value'))
'''
st.code(pivot_longer)


st.subheader('Pivot wider: Reshape data from long to wide format')
pivot_wider = ''
if 'Pandas' in lang_var:
    pivot_wider += '''
#pandas
df.pivot_table(index=['col1_staying','col2_staying'],
      columns='col_pivoting',
      values='val_pivoting'
)
'''
if 'Tidyverse' in lang_var:
    pivot_wider += '''
#tidyverse
df %>% pivot_wider(
      names_from = col_pivoting, 
      values_from = val_pivoting
)
'''
if 'Polars' in lang_var:
    pivot_wider += '''
#polars
df.pivot_table(index=['col1_staying','col2_staying'],
      columns='col_pivoting',
      values='val_pivoting'
)
'''
if 'PySpark' in lang_var:
    pivot_wider += '''
#pyspark
(df
    .groupBy('col1_staying', 'col2_staying')
    .pivot('col_pivoting')
    .agg(F.first('val_pivoting')))
'''
st.code(pivot_wider)


st.subheader('Combine two dataframes')
combine_df = ''
if 'Pandas' in lang_var:
    combine_df += '''
#pandas
pd.concat([df1,df2], axis="index") # vertically stack (row-wise)
pd.concat([df1,df2], axis="columns") # horizontally stack (column-wise)
'''
if 'Tidyverse' in lang_var:
    combine_df += '''
#tidyverse
df1 %>% bind_rows(df2)
df1 %>% bind_cols(df2)
'''
if 'Polars' in lang_var:
    combine_df += '''
#polars
pd.concat([df1,df2])
pd.concat([df1,df2], how="horizontal")
'''
if 'PySpark' in lang_var:
    combine_df += '''
#pyspark
df1.union(df2)
df1.join(df2)
'''
st.code(combine_df)


st.subheader('Inner join two dataframes')
merge_df = ''
if 'Pandas' in lang_var:
    merge_df += '''
#pandas
pd.merge(df1, df2, 
     left_on='df1_id', right_on='df2_id'
)
'''
if 'Tidyverse' in lang_var:
    merge_df += '''
#tidyverse
df1 %>% inner_join(
      df2, by = c(df1_id = "df2_id")
)
'''
if 'Polars' in lang_var:
    merge_df += '''
#polars
pd.join(df1, df2, 
     left_on='df1_id', right_on='df2_id'
)
'''
if 'SQL' in lang_var:
    merge_df += '''
#sql
SELECT *
FROM df1
JOIN df2
ON df1.df1_id = df2.df2_id;
'''
if 'PySpark' in lang_var:
    merge_df += '''
#pyspark
df1.join(df2, df1.df1_id == df2.df2_id, "inner")
'''
st.code(merge_df)


st.subheader('Left join on two dataframes')
left_join_df = ''
if 'Pandas' in lang_var:
    left_join_df += '''
#pandas
pd.merge(df1, df2, how = 'left',
     left_on='df1_id', right_on='df2_id'
)
'''
if 'Tidyverse' in lang_var:
    left_join_df += '''
#tidyverse
df1 %>% left_join(df2, 
      by = c(df1_id = "df2_id")
)
'''
if 'Polars' in lang_var:
    left_join_df += '''
#polars
pd.join(df1, df2, how = 'left',
     left_on='df1_id', right_on='df2_id'
)
'''
if 'SQL' in lang_var:
    left_join_df += '''
#sql
SELECT *
FROM df1
LEFT JOIN df2
ON df1.df1_id = df2.df2_id;
'''
if 'PySpark' in lang_var:
    left_join_df += '''
#pyspark
df1.join(df2, df1.df1_id == df2.df2_id, "left")
'''
st.code(left_join_df)


st.subheader('Create new columns')
create_cols = ''
if 'Pandas' in lang_var:
    create_cols += '''
#pandas
df.assign(
  twomore = lambda df: df.x + 2,
  twoless = lambda df: df.x - 2
)
'''
if 'Tidyverse' in lang_var:
    create_cols += '''
#tidyverse
df %>% mutate(
  twomore = x + 2,
  twoless = x - 2
)
'''
if 'Polars' in lang_var:
    create_cols += '''
#polars
df.with_columns([
    (pl.col("x") + 2).alias("twomore"),
    (pl.col("x") - 2).alias("twoless"),
])
'''
if 'SQL' in lang_var:
    create_cols += '''
#sql
SELECT *, 
       x + 2 AS twomore,
       x - 2 AS twoless
FROM df;
'''
if 'PySpark' in lang_var:
    create_cols += '''
#pyspark
(df
    .withColumn('twomore', F.col('x') + 2)
    .withColumn('twoless', F.col('x') - 2))

'''
st.code(create_cols)

st.subheader('Create new columns with conditional logic')
create_cols_cond = ''
if 'Pandas' in lang_var:
    create_cols_cond += '''
#pandas
df.assign(
  if_twomore = lambda df: 
    np.where(df.column == True, 
             df.x + 2, 
             df.x)
)
'''
if 'Tidyverse' in lang_var:
    create_cols_cond += '''
#tidyverse
df %>% mutate(
  if_twomore = ifelse(
    column == T, 
    x + 2, 
    x)
)
'''
if 'Polars' in lang_var:
    create_cols_cond += '''
#polars
df.with_columns(
    pl.when(pl.col("column") == True)
    .then(pl.col("x")+2)
    .otherwise(pl.col("x"))
    .alias("if_twomore")
)
'''
if 'SQL' in lang_var:
    create_cols_cond += '''
#sql
SELECT *, 
       CASE WHEN column = true THEN x + 2 ELSE x END AS if_twomore
FROM my_table;
'''
if 'PySpark' in lang_var:
    create_cols_cond += '''
#pyspark
df.withColumn(
    'if_twomore', 
    F.when(
        F.col('column') == True, 
        F.col('x') + 2) \\
        .otherwise(F.col('x')))
'''

st.code(create_cols_cond)














