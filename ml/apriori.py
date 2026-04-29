#APRIORI ALGORITHYM
# %%
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from data.import_data import useProductsTransactionsDataset

products_transactions = useProductsTransactionsDataset()

# %%
te = TransactionEncoder()
te_ary = te.fit(products_transactions).transform(products_transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
df_encoded

# %%
from mlxtend.frequent_patterns import apriori
from ml.fp_growth import fpgrowth

apriori(df_encoded, min_support=0.036, use_colnames=True)