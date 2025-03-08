import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import requests
import json
from sklearn.ensemble import RandomForestClassifier

# 1) Criando um pipeline para prever o preço de automóveis

df = pd.DataFrame({
    'Combustivel': ['Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel', 'Etanol', 'Gasolina'],
    'Idade': [3, 5, 2, 8, 1, 6, 4, 7, 3, 5],
    'Quilometragem': [50000, 70000, 30000, 120000, 20000, 90000, 60000, 110000, 45000, 80000],
    'Preco': [30000, 25000, 40000, 15000, 45000, 20000, 28000, 16000, 32000, 22000]
})

X = df[['Combustivel', 'Idade', 'Quilometragem']]
y = df['Preco']

# Definição das transformações

numeric_features = ['Idade', 'Quilometragem']
categorical_features = ['Combustivel']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio (MSE): {mse:.2f}')

# 2) Coletando dados históricos de criptomoeda e treinando um modelo de classificação

def get_crypto_data(symbol='bitcoin', days=30):
    url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days={days}'
    response = requests.get(url)
    data = response.json()
    prices = [entry[1] for entry in data['prices']]
    return prices

data = get_crypto_data()
df_crypto = pd.DataFrame({'Preco_Hoje': data[:-1], 'Preco_Amanha': data[1:]})
df_crypto['Variacao'] = (df_crypto['Preco_Amanha'] > df_crypto['Preco_Hoje']).astype(int)

X_crypto = df_crypto[['Preco_Hoje']]
y_crypto = df_crypto['Variacao']

X_train, X_test, y_train, y_test = train_test_split(X_crypto, y_crypto, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy = np.mean(preds == y_test)

print(f'Acurácia do modelo: {accuracy:.2f}')
