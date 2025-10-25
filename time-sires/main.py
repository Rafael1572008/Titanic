# ===================================================
# Módulo para o estudo de dados do tipo time
# ===================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  # use apenas se estiver em um Jupyter Notebook

# ===========================
# Tempo
import datetime
# ===========================


# locale
import locale

# Data e hora atual
print(datetime.datetime.now())

# Diferença de tempo (timedelta)
antes = datetime.datetime.now()
depois = datetime.datetime.now()

diferenca = depois-antes # me dá a diferença dos tempos
print(diferenca.seconds) # tempo em segundos

print(antes, depois)

print(datetime.timedelta(seconds=10))  # Descrever o intevalo de tempo (secunds, day, houers)


# String para data
