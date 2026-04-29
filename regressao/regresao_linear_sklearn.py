from sklearn import linear_model
import pandas as pd
import numpy as np

# Definir x e y
mes = list(range(1, 13))
print(f'mes: {mes}')

faturamento = [10, 20, 30, 10, 40, 30, 50, 40, 60, 50, 40, 70]
print(f'faturamento: {faturamento}')

data_dict = {'mes': mes, 'faturamento': faturamento}
print(f'Data_dict: {data_dict}')

data = pd.DataFrame.from_dict(data_dict)
print('Data Frame')
print(data)


## Criar Gráfico de dispeersão
x = data['mes']
y = data['faturamento']


# instanciar o modelo
lm = linear_model.LinearRegression()

# Reshape dos dados
x = np.array(x).reshape(-1, 1)
print(f'x reshape: {x}')


# Treinar o modelo
sk_model = lm.fit(x, y)


# Gerar predições
sk_model_pred = sk_model.predict(x)
print(f'Predições: {sk_model_pred}')


# R²
print(f'R²: {sk_model.score(x, y)}')


# Coeficiente M
print(f'Coeficiente M: {sk_model.coef_}')

# Coeficiente B
print(f'Coeficiente B: {sk_model.intercept_}')