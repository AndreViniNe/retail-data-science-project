import pandas as pd

def importRetailsDataset() -> pd.DataFrame:
    try:
        retails_df = pd.read_csv("./Retail_Transactions_Dataset.csv")
        print("Dataset loaded successfully.")
        return retails_df
    except FileNotFoundError:
        print(f"Error: The file ./Retail_Transactions_Dataset.csv was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None
    
def useSupermarketDataset():
    import ast

    df = importRetailsDataset()
    df_supermarket = (
        df[df["Store_Type"] == 'Supermarket'].copy()
                                             ["Product"].apply(
                                                 lambda x: ast.literal_eval(x) 
                                                    if isinstance(x, str) 
                                                    and x.startswith("[") 
                                                    else x
    ))

    return df_supermarket

def useProductsTransactionsDataset():
    df = useSupermarketDataset()
    return df["Product"]
