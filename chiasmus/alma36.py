import pandas as pd

df = pd\
    .read_excel('alma36.xlsx')\
    .assign(
        c = lambda x: x["c"].fillna(method='ffill').astype(int),
        v = lambda x: x["v"].fillna(method='ffill').astype(int))\
    .replace("x", 1)\
    .fillna(0)\
    .sum()