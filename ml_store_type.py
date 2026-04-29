# %%
import pandas as pd
import ast

# %%
from data.import_data import useSupermarketDataset
df_supermarket = useSupermarketDataset()
df_supermarket.head()

#%%
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
