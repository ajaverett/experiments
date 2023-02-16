import pandas as pd
import numpy as np
import sqlite3
import ast


con = sqlite3.connect('database.sqlite')

# pd.read_sql_query("""
#     SELECT name 
#     FROM sqlite_master 
#     WHERE type='table'
# """, con)

df = pd.read_sql_query(
"""
SELECT  id, 
        minutes, 
        tags, 
        n_steps,
        ingredients,
        n_ingredients
FROM RAW_recipes
""", con)


df = df\
    .assign(
        is_healthy = lambda x: 
            x.tags.str.contains("healthy").values,
        tags=lambda x: 
            x.tags.apply(lambda y: ast.literal_eval(y)),
        ingredients=lambda x: 
            x.ingredients.apply(lambda y: ast.literal_eval(y)))\
    .drop(columns='tags')   
        
ingredients = [item\
               for sublist\
               in df['ingredients']\
               for item\
               in sublist]

popular_ingredients = pd.DataFrame(ingredients)\
    .rename({0:"tags"}, axis=1)\
    .assign(dummy = 1)\
    .groupby('tags', as_index=False)\
    .count()\
    .sort_values('dummy',ascending=False)\
    .head(100)\
    ['tags']

df= df\
        .assign(ingredients = 
                lambda df_: df_.ingredients.apply(lambda x: [item for item in x if item in list(popular_ingredients)])
                )

df = pd.concat([
        df,
        pd.get_dummies(df['ingredients'].apply(pd.Series).stack()).sum(level=0)], axis=1)\
    .drop(columns = ['ingredients','id'])

#%%
X = df.drop(columns = 'is_healthy').fillna(0)
y = df['is_healthy'].fillna(0)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=0)

ro = RandomOverSampler()
X_pred, y_pred = ro.fit_resample(X, y)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)

score = accuracy_score(y_test, y_pred)

confusion_matrix = confusion_matrix(list(y_test), list(y_pred))
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = clf.classes_)
cm_display.plot(cmap=plt.cm.Purples)


feature_df = pd.DataFrame({'features': X_train.columns,'importance': clf.feature_importances_}).sort_values('importance',ascending = False).head(20)




# %%
