import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

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


# Adicionar constante
x = sm.add_constant(x)

# Criar modelo
model = sm.OLS(y, x).fit()

# Gerar predições
sm_pred = model.predict(x)

print(model.summary())

# RMSE
rmse = rmse(y, sm_pred)
print(f'RMSE: {rmse}')