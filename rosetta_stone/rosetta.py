import streamlit as st

st.title('Pandas Reference Guide')
st.write('Note: This may have errors, please let me know if you find any.')

st.header('1. Getting Started')

st.subheader('Imports')
st.code('''
import pandas as pd
import numpy as np
''')

st.subheader('Create a series')
st.code('''
series = pd.Series([1,2,3])
''')

st.subheader('Create a dataframe')
st.code('''
df = pd.DataFrame(
    {'col_one': ['A', 'B', 'C','D'],
     'col_two': [1, 2, 3, 4]}
)
''')

st.subheader('Read CSV file')
st.code('''
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', header=None)
''')

st.header('2. Understanding Your Data')

st.subheader('Count values in a column')
st.code('''
df['col_one'].value_counts()
''')

st.subheader('Calculate statistics')
st.code('''
df['col_one'].mean()
df['col_one'].median()
df['col_one'].std()
df['col_one'].min()
df['col_one'].max()
''')

st.subheader('Find distinct values')
st.code('''
df.drop_duplicates(subset = ["col_one"])
df.drop_duplicates()
''')

st.header('3. Filter Columns')

st.subheader('Keep columns')
st.code('''
df.filter(items=['col_one'])
df.filter(items=['col_one','col_two'])
df.filter(regex='[pt]al')
df.loc[:,df.columns.str.startswith("prefix_")]
df.loc[:,df.columns.str.endswith("_suffix")]
df.loc[:,df.columns.str.contains("_infix_")]
''')

st.subheader('Drop columns')
st.code('''
df.drop(columns=['col_one'])
df.drop(columns=['col_one','col_two'])
''')

st.subheader('Rename columns')
st.code('''
df.rename(columns={"column_one": "new_col_1"})
df.rename(columns={"column_one": "new_col_1", 
                   "column_two": "new_col_2"})
''')

st.subheader('Select columns by position')
st.code('''
df.iloc[:,[1,3]]  # all rows and second and fourth column
df.iloc[:,1:3]    # all rows and second to third column
''')

st.header('4. Filter Rows')

st.subheader('Filter by values')
st.code('''
df.query("col_one >= 100")
df.query("col_one != 'Blue'")
df.query("col_one in ['A', 'B']")
df.query("Race == 'White' and Gender == 'Male'")
df.query("not (Race == 'White' and Gender == 'Male')")

# Alternative syntax
df[df["col_one"] >= 100]
df[df["col_one"] != "Blue"]
df[df["col_one"].isin(['A', 'B'])]
df[(df["Race"] == "White") & (df["Gender"] == "Male")]
df[~((df["Race"] == "White") & (df["Gender"] == "Male"))]
''')

st.subheader('Filter by string patterns')
st.code('''
df.query('col_one.str.contains("string", na=False)', engine="python")
df.query('col_one.str.contains("string1|string2", na=False, regex=True)', engine="python")
df.query('col_one.str.startswith("string", na=False)', engine="python")
df.query('col_one.str.endswith("string", na=False)', engine="python")
df.query('col_one.str.match(regex_pattern, na=False)', engine="python")
''')

st.subheader('Select rows by position')
st.code('''
df.loc[:,:] #all rows and columns
df.loc[1,:] #second row and all columns
df.loc[[1,6],:] #second and seventh row and all columns
df.loc[1:6, :] #second to seventh row and all columns
df.loc[:,['col_one']] #all rows and column_one
df.loc[:,['col_one','col_three']] #all rows and column_one and column_three
df.loc[:,'col_one':'col_three'] #all rows and column_one to column_three
''')

st.subheader('Sort rows')
st.code('''
df.sort_values('col_one')
df.sort_values('col_one', ascending=False)
''')

st.header('5. Handle Data Quality')

st.subheader('Drop missing values')
st.code('''
df.dropna()
df.dropna(subset=['col_one', 'col_two'])
df.dropna(thresh=n) #integer threshold
''')

st.subheader('Fill missing values')
st.code('''
df.fillna(x)
df['col_one'].fillna(x)
df['col_one'].fillna(method='ffill')
df['col_two'].fillna(df['col_two'].mean())
''')

st.subheader('Convert data types')
st.code('''
df.astype({"Race":'category', 
           "Age":'int64',
           "Zip":'string'})
''')

st.subheader('Replace values')
st.code('''
df.replace(2,"foo")
df[['col_one','col_two']].replace(2,"foo")
df['col_one'].replace(2,"foo")
''')

st.header('6. Create New Variables')

st.subheader('Create new columns')
st.code('''
df.assign(
    twomore = lambda df: df.x + 2,
    twoless = lambda df: df.x - 2
)
''')

st.subheader('Create columns with conditional logic')
st.code('''
df.assign(
    if_twomore = lambda df: 
        np.where(df.column == True, 
                 df.x + 2, 
                 df.x)
)
''')

st.header('7. Reshape Data')

st.subheader('Pivot longer (wide to long)')
st.code('''
df.melt(
    id_vars=['columns_staying_put'],
    var_name=['col1_melting','col2_melting'])
''')

st.subheader('Pivot wider (long to wide)')
st.code('''
df.pivot_table(index=['col1_staying','col2_staying'],
               columns='col_pivoting',
               values='val_pivoting',
               aggfunc='mean')
''')

st.header('8. Combine Datasets')

st.subheader('Concatenate dataframes')
st.code('''
pd.concat([df1,df2], axis="index")   # vertically stack (row-wise)
pd.concat([df1,df2], axis="columns") # horizontally stack (column-wise)
''')

st.subheader('Inner join')
st.code('''
pd.merge(df1, df2, 
         left_on='df1_id', right_on='df2_id')
''')

st.subheader('Left join')
st.code('''
pd.merge(df1, df2, how='left',
         left_on='df1_id', right_on='df2_id')
''')

st.header('9. Aggregate & Summarize')

st.subheader('Group and aggregate')
st.code('''
# Count rows per group
df.groupby('Race', as_index=False).size()

# Single aggregation
df.groupby('Race', as_index=False)['Income'].median()

# Multiple aggregations
(df.groupby('Race', as_index=False)
   .agg({
        "Income":"median",
        "id":"count",
        "Age":"mean"
   })
)

# Named aggregations with multiple groups
(df.groupby(['Race', 'Sex'], as_index=False)
   .agg(
      new_col1=pd.NamedAgg(column='Income', aggfunc=np.median),
      new_col2=pd.NamedAgg(column='id', aggfunc='count'),
      new_col3=pd.NamedAgg(column='Age', aggfunc=np.mean)
))
''')
