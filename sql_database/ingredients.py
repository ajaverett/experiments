import pandas as pd
import numpy as np
import sqlite3

con = sqlite3.connect('database.sqlite')

# pd.read_sql_query("""
#     SELECT name 
#     FROM sqlite_master 
#     WHERE type='table'
# """, con)

pd.read_sql_query(
"""
    SELECT * 
    FROM RAW_recipes
    LIMIT 10 
""", con)


df = pd.read_sql_query(
"""
    SELECT ingredients, n_ingredients 
    FROM RAW_recipes 
""", con)



df = df\
    .assign(
        ingredients = lambda x: x.ingredients.str.split("'"))\
    .explode("ingredients")\
    .query('ingredients.str.match("[A-Za-z]").values')\
    .assign(count = 1)\
    .groupby(['ingredients','n_ingredients'], 
             as_index=False)\
    .agg(
        total = pd.NamedAgg(
            column = 'count', 
            aggfunc = np.sum))

df\
    .sort_values(['n_ingredients','total'], ascending = [True, False])\
    .query("n_ingredients in [2,3,4,5,6]")\
    .drop(columns=['n_ingredients'])\
    .groupby(['ingredients'], as_index=False)\
    .agg(
        total = pd.NamedAgg(
                column = 'total', 
                aggfunc = np.sum))\
    .query("total > 50")\
    .sort_values("total",ascending=False)