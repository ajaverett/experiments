#%%
import pandas as pd
import numpy as np
import spacy
import janitor

alma36 = pd.read_csv("alma36_chunks.csv", header=None)[0]

nlp = spacy.load("en_core_web_lg")

doc_vectors = [nlp(text).vector for text in alma36]

norm_vectors = [vec / np.linalg.norm(vec) \
                if np.linalg.norm(vec) > 0 \
                else np.zeros_like(vec) \
                for vec \
                in doc_vectors]

norm_vectors = np.array(norm_vectors)

similarity_matrix = np.around(np.dot(norm_vectors, norm_vectors.T), decimals=3)

np.fill_diagonal(similarity_matrix, np.nan)


df = pd.DataFrame(np.tril(similarity_matrix), 
                  index=alma36, 
                  columns=alma36)\
       .replace(0,np.nan)


df.to_excel("alma36.xlsx")

#%%

import openpyxl
from openpyxl.formatting.rule import ColorScaleRule

workbook = openpyxl.load_workbook('alma36.xlsx')
sheet = workbook.active

for column_cells in sheet.columns:
    length = max(len(str(cell.value)) for cell in column_cells)
    sheet.column_dimensions[openpyxl.utils.get_column_letter(column_cells[0].column)].width = length

color_scale_rule = ColorScaleRule(start_type='min',
                                  start_color='FFD9D9D9',
                                  end_type='max', end_color='FF7AC5CD')     

sheet.conditional_formatting.add('B2:EY155', color_scale_rule)

# Save the workbook
workbook.save('alma36.xlsx')

# %%
