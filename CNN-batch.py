import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow import keras
from keras import Model
from keras.utils import plot_model
from keras.models import Model
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.layers import Input 

caminho_pasta = os.getcwd()
dados = pd.read_csv(caminho_pasta + "/BaseCidade2019.csv", sep=";")


# Selecionando as features numéricas e categóricas
numericos = ['month', 'weekday', 'day', 'hour', 'temperature', 'r_temperature', 'wind', 'humidity', 'dew_point', 'pressure']
categoricos = ['season', 'workday']

# Aplicando normalização às features numéricas
scaler = StandardScaler()
dados_numericos = pd.DataFrame(scaler.fit_transform(dados[numericos]), columns=numericos)

# Aplicando LabelEncoder às features categóricas
encoder = LabelEncoder()
dados_categoricos = dados[categoricos].apply(encoder.fit_transform)

# Concatenando os resultados
processados = pd.concat([dados_numericos, dados_categoricos, dados['qtd']], axis=1)
df = processados

# Separando treinamento e teste
X = df.drop(columns=['qtd'])
y = df['qtd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Ajuste da forma dos dados de entrada para a CNN
X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Definir lista de tamanhos de lote
batch_sizes = [4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]

# Dicionário para armazenar histórico de perdas para cada tamanho de lote
r2_histories = {}
loss_histories = {}

# Loop sobre tamanhos de lote
for batch_size in batch_sizes:
    print("Testando com batch size "+str(batch_size) +"\n")
    # Criar modelo
    model = Sequential()
    model.add(Input(shape=(X_train_reshaped.shape[1], 1)))
    model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='relu'))
    model.add(Conv1D(filters=528, kernel_size=4, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # Achatar os dados para conectá-los à camada densa
    model.add(Flatten())
    model.add(Dense(units=528, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=1))  # Camada de saída para regressão
    model.compile(optimizer='adam', loss='mse')
    qntEpochs = 50
    history = model.fit(X_train_reshaped, y_train, epochs=qntEpochs, batch_size=batch_size, validation_data=(X_test_reshaped, y_test))
    # Calcular R²
    y_pred = model.predict(X_test_reshaped)
    r2 = r2_score(y_test, y_pred)
    # Armazenar R²
    r2_histories[batch_size] = r2
    loss_histories[batch_size] = history.history['loss'][-1]  # Pegando o último valor de perda
    
# Salvar resultados em um arquivo TXT
with open('Resultados/loss_vs_batch_size-7.txt', 'w') as f:
    for batch_size in batch_sizes:
        f.write(f'Batch Size: {batch_size}, R²: {r2_histories[batch_size]}, Loss: {loss_histories[batch_size]}\n')

# Plotar gráfico de R² para diferentes tamanhos de lote
plt.figure(figsize=(5, 5))
plt.plot(batch_sizes, [r2_histories[batch_size] for batch_size in batch_sizes], marker='o')
plt.xlabel('Batch Size')
plt.ylabel('R²')
plt.ylim(0.0, 1.0)
plt.title('R² for Different Batch Sizes')
plt.grid(True)
plt.tight_layout()
plt.savefig('Imagens/r2_vs_batch_size-5a50.png')
plt.show()


# Plotar gráfico de Loss para diferentes tamanhos de lote
plt.figure(figsize=(5, 5))
plt.plot(batch_sizes, [loss_histories[batch_size] for batch_size in batch_sizes], marker='o', color='r')
plt.xlabel('Batch Size')
plt.ylabel('Loss')
plt.title('Loss for Different Batch Sizes')
plt.grid(True)
plt.tight_layout()
plt.savefig('Imagens/loss_vs_batch_size-5a50.png')  # Salvar gráfico como imagem
plt.show()