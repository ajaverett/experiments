import sqlite3
import pandas as pd
import os

# Connect to the database
conn = sqlite3.connect('database.sqlite')
cursor = conn.cursor()
# Create a table for each CSV file
folder = 'data'
for filename in os.listdir(folder):
    if filename.endswith('.csv'):
        file = os.path.join(folder, filename)
        table_name = filename[:-4]
        df = pd.read_csv(file)
        df.to_sql(table_name, conn, if_exists='replace', index=False)

# Commit and close the connection
conn.commit()
conn.close()