import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

# 2) Regressão Linear Manual

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

x_mean, y_mean = np.mean(x), np.mean(y)
b1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
b0 = y_mean - b1 * x_mean
print(f"Equação da reta: y = {b0:.2f} + {b1:.2f}x")

# 6) Regressão Linear - Custo de Energia

df = pd.DataFrame({
    'aparelhos': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'custo': [30, 50, 70, 85, 105, 120, 140, 160, 185, 200]
})
X = df[['aparelhos']]
y = df['custo']
modelo = LinearRegression().fit(X, y)
preds = modelo.predict(X)
mse = mean_squared_error(y, preds)
print(f"Erro quadrático médio: {mse:.2f}")

# 7) Regressão Logística - Doença Cardíaca

df = pd.DataFrame({
    'horas_exercicio': np.random.randint(0, 10, 100),
    'risco': np.random.choice([0, 1], 100)
})
X = df[['horas_exercicio']]
y = df['risco']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo = LogisticRegression().fit(X_train, y_train)
preds = modelo.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Acurácia: {acc:.2f}")

# 8) K-Means - Agrupamento de Alunos

df = pd.DataFrame({
    'matematica': np.random.randint(50, 100, 30),
    'ciencias': np.random.randint(50, 100, 30)
})
modelo = KMeans(n_clusters=3).fit(df)
df['grupo'] = modelo.labels_
plt.scatter(df['matematica'], df['ciencias'], c=df['grupo'])
plt.xlabel('Matemática')
plt.ylabel('Ciências')
plt.title('Agrupamento de Alunos')
plt.show()

# 9) Árvore de Decisão - Classificação de Veículos

df = pd.DataFrame({
    'potencia': np.random.randint(50, 200, 20),
    'peso': np.random.randint(800, 2000, 20),
    'economico': np.random.choice(['Econômico', 'Não Econômico'], 20)
})
X = df[['potencia', 'peso']]
y = df['economico']
modelo = DecisionTreeClassifier().fit(X, y)
plt.figure(figsize=(10, 5))
plot_tree(modelo, feature_names=['potencia', 'peso'], class_names=['Econômico', 'Não Econômico'], filled=True)
plt.show()