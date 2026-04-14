# %%
import pandas as pd
import numpy as np
import ast

# %%
df = pd.read_csv("./data/Retail_Transactions_Dataset.csv")
df.head()

# %%
df_supermarket = df[df["Store_Type"] == 'Supermarket'].copy()
df_supermarket.head()

# %%
df_supermarket.shape
# %%
# Items mais vendidos e menos vendidos no setor
df_supermarket.dtypes

# %%
df_supermarket["Product"] = df_supermarket["Product"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
)

df_exploded = (
    df_supermarket.explode("Product")
    .assign(Product=lambda d: d["Product"].astype(str).str.strip())
)

df_exploded.head()

# %%
sales_per_product = (
df_exploded
.groupby("Product")
.size()
.sort_values(ascending=False)
)
sales_per_product

# %%
best_product = sales_per_product.head(1)
worst_product = sales_per_product.tail(1)
print(f"Best {best_product} \nWorst {worst_product}")

# %%
product_per_customer_category = (
    df_exploded
        .groupby(["Customer_Category", "Product"], as_index=False)
        .size()
        .rename(columns={"size": "sales_count"})
)
product_per_customer_category

# %%
idx = product_per_customer_category.groupby("Customer_Category")["sales_count"].idxmax()
idx

# %%
top_product_per_customer_category = (
    product_per_customer_category.loc[idx]
    .sort_values("Customer_Category")
    .reset_index(drop=True)
)

top_product_per_customer_category
# %%
