# Projeto Varejo

Este repositório ignora arquivos de dados grandes na pasta `data/` para evitar limite de tamanho no GitHub.

## Download do dataset (KaggleHub)

1. Instale o pacote:

```bash
pip install kagglehub
```

2. Execute o script abaixo para baixar a versao mais recente do dataset:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("prasad22/retail-transactions-dataset")

print("Path to dataset files:", path)
```

Link do dataset:
https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset?resource=download

## Observacao

- O arquivo `data/Retail_Transactions_Dataset.csv` nao e versionado no Git.
- Mantenha os dados na pasta `data/` localmente.
