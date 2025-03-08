  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # 1) Processamento de Dados Meteorológicos
  
  data = {
      'Data': ['15/01/2025'] * 5,
      'Cidade': ['São Paulo', 'Rio de Janeiro', 'Curitiba', 'Porto Alegre', 'Salvador'],
      'Temperatura Máxima (°C)': [30.5, 35.0, 24.0, 28.0, 31.0],
      'Temperatura Mínima (°C)': [22.0, 25.0, 18.0, 20.0, 24.5],
      'Precipitação (mm)': [12.0, np.nan, 8.0, 15.0, np.nan],
      'Umidade Relativa (%)': [78, 70, np.nan, 82, 80]
  }
  
  df = pd.DataFrame(data)
  
  # Substituir valores ausentes pela média/mediana
  
  df['Precipitação (mm)'].fillna(df['Precipitação (mm)'].mean(), inplace=True)
  df['Umidade Relativa (%)'].fillna(df['Umidade Relativa (%)'].median(), inplace=True)
  
  # Adicionar a coluna Amplitude Térmica
  df['Amplitude Térmica'] = df['Temperatura Máxima (°C)'] - df['Temperatura Mínima (°C)']
  
  # Criar DataFrame com cidades com Temperatura Máxima acima de 30°C
  
  df_temp_alta = df[df['Temperatura Máxima (°C)'] > 30]
  
  # Reordenar as colunas
  
  df = df[['Data', 'Cidade', 'Temperatura Máxima (°C)', 'Temperatura Mínima (°C)',
           'Amplitude Térmica', 'Precipitação (mm)', 'Umidade Relativa (%)']]
  
  print("DataFrame Processado:")
  print(df)
  
  # 2) Gráfico de Temperatura ao longo do Dia
  
  horas = list(range(25))
  temperaturas = [15 + (i / 12) * 15 if i <= 12 else 30 - ((i - 12) / 12) * 12 for i in horas]
  
  plt.figure(figsize=(10, 5))
  plt.plot(horas, temperaturas, marker='o', linestyle='-', color='b', label='Temperatura (°C)')
plt.xlabel('Horário (h)')
plt.ylabel('Temperatura (°C)')
plt.title('Evolução da Temperatura ao Longo do Dia')
plt.grid(True)
plt.legend()
plt.show()

# 3) Análise de Vendas com Seaborn

vendas_data = {
    'Dia': ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'],
    'Vendas': [200, 220, 180, 250, 300, 400, 380],
    'Clientes': [50, 55, 45, 65, 80, 120, 110],
    'Lucro': [1000, 1100, 900, 1300, 1800, 2500, 2300]
}

df_vendas = pd.DataFrame(vendas_data)

# Gráfico de Barras - Total de Vendas por Dia
plt.figure(figsize=(8, 5))
sns.barplot(x='Dia', y='Vendas', data=df_vendas, palette='Blues')
plt.title('Total de Vendas por Dia')
plt.xlabel('Dia da Semana')
plt.ylabel('Total de Vendas')
plt.show()

# Gráfico de Dispersão - Número de Clientes vs. Total de Vendas

plt.figure(figsize=(8, 5))
sns.scatterplot(x='Clientes', y='Vendas', data=df_vendas, color='red')
plt.title('Número de Clientes vs. Total de Vendas')
plt.xlabel('Número de Clientes')
plt.ylabel('Total de Vendas')
plt.show()

# Heatmap - Correlação entre Vendas, Clientes e Lucro

plt.figure(figsize=(6, 4))
sns.heatmap(df_vendas[['Vendas', 'Clientes', 'Lucro']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlação entre Vendas, Clientes e Lucro')
plt.show()
