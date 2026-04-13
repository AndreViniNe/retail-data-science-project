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
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df.head()

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
df_groupby_store = df.groupby(by="Store_Type")[["Total_Items", "Total_Cost"]].sum().copy()
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
# 6- Métodos de pagamento usados nas compras mais caras
df_expensive_purchase = df.sort_values(by="Total_Cost", ascending=False) \
                          .head(10) \
                          ["Payment_Method"].value_counts() \
                          .copy()
df_expensive_purchase

# %%
#7- Quantidade de vendas por hora
df_temporal = df[["Total_Items", "City"]].copy()
df_temporal["DateCalendar"] = df["Date"].dt.date
df_temporal["Hour"] = df["Date"].dt.hour
df_temporal["Week_Day"] = df["Date"].dt.day_of_week
df_temporal.head()

# %%
df_sales_per_hour = (
    df_temporal.groupby("Hour", as_index=False)["Total_Items"]
    .sum()
    .sort_values(by="Hour")
    .reset_index(drop=True)
)

df_sales_per_hour

# %%
df_sales_per_hour['Normalized_Total_Items'] = (df_sales_per_hour['Total_Items']) / 10000
plt.bar(df_sales_per_hour['Hour'], df_sales_per_hour['Normalized_Total_Items'])
plt.xlabel('Hour of the day')
plt.ylabel('Total Items sold (%10000)')
plt.xticks(np.arange(0,24))
plt.yticks(np.arange(0, 25, step=2))
plt.show()

# %%
# 8- Quantidade de vendas por dia da semana
df_sales_per_week_day = (
    df_temporal.groupby("Week_Day", as_index=False)["Total_Items"]
    .sum()
    .sort_values(by="Week_Day")
    .reset_index(drop=True)
)

df_sales_per_week_day

# %%
df_sales_per_week_day["Normalized_Total_Items"] = (df_sales_per_week_day['Total_Items']) / 10000
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.bar(df_sales_per_week_day['Week_Day'], df_sales_per_week_day['Normalized_Total_Items'])
plt.xlabel('Week day')
plt.ylabel('Total Items sold (%10000)')
plt.xticks(np.arange(7), day_names)
plt.show()
# %%
# 9- Dia e hora com mais venda por cidade
df.head()

# %%
df_top_sales = df_temporal.groupby(["City", "Week_Day", "Hour"], as_index=False) \
                          ["Total_Items"].sum()
df_top_sales

# %%
idx = df_top_sales.groupby("City")["Total_Items"].idxmax()
df_top_sales_by_city = (
    df_top_sales.loc[idx, ["City", "Week_Day", "Hour", "Total_Items"]]
    .sort_values(by="City")
    .reset_index(drop=True)
)
df_top_sales_by_city

# %%
plt.figure(figsize=(10,6))
plt.scatter(
df_top_sales_by_city["Hour"],
df_top_sales_by_city["Week_Day"],
s=df_top_sales_by_city["Total_Items"] / 5,
alpha=0.7
)

for _, row in df_top_sales_by_city.iterrows():
    plt.text(row["Hour"] + 0.1, row["Week_Day"] + 0.05, row["City"], fontsize=8)

day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
plt.yticks(range(7), day_names)
plt.xticks(range(24))
plt.xlabel("Hour")
plt.ylabel("Week day")
plt.title("Peak sales window by city (bubble size = Total_Items)")
plt.grid(alpha=0.2)
plt.show()

# %%
