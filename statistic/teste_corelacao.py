def pct(x):
    return "{:.2f}%".format(x * 100)

import pandas as pd
import numpy as np

### Correlação da Área de um Círculo com seu Raio

## Gerar dados
raio = np.array([1, 2, 3, 4, 5])
area = np.pi * raio ** 2
print(f'Raio: {raio}')
print(f'Área: {area}')

# Desvio Padrão Amostral - raio
desvio_padrao_raio = np.sqrt(np.sum((raio - raio.mean()) ** 2) / (len(raio) - 1))
print(f'Desvio Padrão Amostral do Raio: {desvio_padrao_raio}')

# Desvio Padrão Amostral - área
desvio_padrao_area = np.sqrt(np.sum((area - area.mean()) ** 2) / (len(area) - 1))
print(f'Desvio Padrão Amostral da Área: {desvio_padrao_area}')

# Covariância
covariancia = np.sum((raio - raio.mean()) * (area - area.mean())) / (len(area) - 1)
print(f'Covariância: {covariancia}')

# Correlação
correlacao = covariancia / (desvio_padrao_raio * desvio_padrao_area)
print(f'Correlação: {pct(correlacao)}')
