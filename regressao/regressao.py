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

plt.show()