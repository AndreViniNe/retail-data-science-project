# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("./data/Retail_Transactions_Dataset.csv")
df.head()

# %% Análise Exploratória
df.shape

#%%
df.isna().sum().sort_values()

#%%
# 1- Total de itens vendidos
total_items = df["Total_Items"].sum()
print("Total de itens vendidos: ", total_items)

# %%
# 2- Valor total de vendas
total_cost = df["Total_Cost"].sum()
print("Valor total de vendas: R$", round(total_cost, 2))

# %%
# 3 e 4- Porcentagem de itens vendidos e valor total de vendas por tipo de loja
df_groupby_store = df.groupby(by="Store_Type")[["Total_Items", "Total_Cost"]].sum()
df_groupby_store.head()

# %%
df_groupby_store["Percent_On_Total_Items"] = round((df_groupby_store["Total_Items"]/total_items)*100, 2)
df_groupby_store["Percent_On_Total_Cost"] = round((df_groupby_store["Total_Cost"]/total_cost)*100, 2)
df_groupby_store
# %%
# 5- Métodos de pagamentos mais e menos utilizados
df.head()

# %%
df["Payment_Method"].value_counts()

# %%
