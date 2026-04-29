import pandas as pd
import matplotlib.pyplot as plt

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

plt.scatter(x, y)
# plt.show()


## Criando modelo preditivo
# y = mx + b

# m
sx = x.sum()
print(f'Soma de x: {sx}')

sy = y.sum()
print(f'Soma de y: {sy}')

xy = (x*y).sum()
print(f'Soma de xy: {xy}')

x_squared = (x**2).sum()
print(f'Soma de x²: {x_squared}')

n = len(x)
print(f'tamanho: {n}')

sx2 = (sx**2)
print(f'Quadrado do Somatorio de x: {sx2}')


m = (n * xy  - sx * sy) / (n * x_squared - sx2)
print(m)

print(f'y = {m.round(4)}x + b')



# b

b = (sy - m * sx) / (n)

print(f'y = {m.round(4)}x + {b.round(4)}')


# Predição
xpred = 3
ypred = m * xpred + b
print(xpred, ypred)

## Criar Predição para os próximos 12 meses

pred_manual = []

for elemento in x:
    ypred = m * elemento + b
    pred_manual.append(ypred)

print(pred_manual)

data['predições'] = pred_manual
print()
print(data)


## Visualizar regresão linear
plt.scatter(x, y)
plt.plot(x, pred_manual, color='red')

# lt.show()


# Coeficiente de determinação R²

data['residuos'] = data['faturamento'] - data['predições']
print(data)

SQres = (data['residuos'] ** 2).sum()
print(f'Soma dos residuos ao quadrado: {SQres}')


media = data['faturamento'].mean()

data['faturamento medio'] = media
print(data)

SQtot = ((data['faturamento'] - data['faturamento medio']) ** 2).sum()
print(f'Soma total ao quadrado: {SQtot}')

r_squared = 1 - (SQres / SQtot)
print(f'R²: {r_squared.round(4)}')


# RMSE
rmse = (SQres / n) ** 0.5
print(f'RMSE: {rmse.round(4)}')


# Regra empírica, desvio padão de 68%
# plt.plot(x, pred_manual, color='red')
# plt.scatter(x, y)
# plt.plot(x, pred_manual + rmse, color='green', linestyle='dashed')
# plt.plot(x, pred_manual - rmse, color='green', linestyle='dashed')
# plt.show()

# Dois desvio padão de 95
# plt.plot(x, pred_manual, color='red')
# plt.scatter(x, y)
# plt.plot(x, pred_manual + rmse, color='green', linestyle='dashed')
# plt.plot(x, pred_manual - rmse, color='green', linestyle='dashed')
# plt.show()

# Dois desvio padão de 95
# plt.plot(x, pred_manual, color='red')
# plt.scatter(x, y)
# plt.plot(x, pred_manual, color='red')
# plt.plot(x, pred_manual + rmse, color='green', linestyle='dashed')
# plt.plot(x, pred_manual - rmse, color='green', linestyle='dashed')
# plt.plot(x, pred_manual + 2*rmse, color='purple', linestyle='dashed')
# plt.plot(x, pred_manual - 2*rmse, color='purple', linestyle='dashed')
# plt.show()

# três desvio padão de 99,7%
# plt.plot(x, pred_manual, color='red')
# plt.scatter(x, y)
# plt.plot(x, pred_manual, color='red')
# plt.plot(x, pred_manual + rmse, color='green', linestyle='dashed')
# plt.plot(x, pred_manual - rmse, color='green', linestyle='dashed')
# plt.plot(x, pred_manual + 2*rmse, color='purple', linestyle='dashed')
# plt.plot(x, pred_manual - 2*rmse, color='purple', linestyle='dashed')
# plt.plot(x, pred_manual + 3*rmse, color='orange', linestyle='dashed')
# plt.plot(x, pred_manual - 3*rmse, color='orange', linestyle='dashed')
# plt.show()