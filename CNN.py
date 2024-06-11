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
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.layers import Input
from keras.callbacks import EarlyStopping
from tensorflow.keras.activations import swish

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

# Função para calcular o RMSE ponderado
def weighted_rmse(y_true, y_pred, weight_factor):
    errors = y_true - y_pred
    weighted_errors = np.where(errors < 0, weight_factor * errors ** 2, errors ** 2)
    weighted_rmse = np.sqrt(np.mean(weighted_errors))
    return weighted_rmse

# Função para calcular o MAE ponderado
def weighted_mae(y_true, y_pred, weight_factor):
    errors = y_true - y_pred
    weighted_errors = np.where(errors < 0, weight_factor * np.abs(errors), np.abs(errors))
    weighted_mae = np.mean(weighted_errors)
    return weighted_mae

#  Ajuste da forma dos dados de entrada para a CNN
X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential()

# model.add(Input(shape=(X_train_reshaped.shape[1], 1)))
# model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
# model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
# model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
# model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
# model.add(Dropout(0.10)) 
# model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
# model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='relu'))
# model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='relu'))
# model.add(Conv1D(filters=528, kernel_size=4, padding='same', activation='relu'))
# model.add(Conv1D(filters=528, kernel_size=4, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# # Achatar os dados para conectá-los à camada densa
# model.add(Flatten())
# model.add(Dense(units=528, activation='relu'))
# model.add(Dense(units=256, activation='relu'))
# model.add(Dense(units=64, activation='relu'))
# model.add(Dense(units=32, activation='relu'))
# model.add(Dense(units=32, activation='relu'))
# model.add(Dense(units=16, activation='relu'))
# model.add(Dense(units=16, activation='relu'))
# model.add(Dense(units=8, activation='relu'))
# model.add(Dense(units=4, activation='relu'))
# model.add(Dense(units=1))  # Camada de saída para regressão


model.add(Input(shape=(X_train_reshaped.shape[1], 1)))
model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='relu'))
model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='relu'))
model.add(Conv1D(filters=528, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
# Achatar os dados para conectá-los à camada densa
model.add(Flatten())
model.add(Dense(units=528, activation='swish'))
model.add(Dense(units=256, activation='swish'))
model.add(Dense(units=64, activation='swish'))
model.add(Dense(units=32, activation='swish'))
model.add(Dense(units=32, activation='swish'))
model.add(Dense(units=16, activation='swish'))
model.add(Dense(units=16, activation='swish'))
model.add(Dense(units=8, activation='swish'))
model.add(Dense(units=4, activation='swish'))
model.add(Dense(units=1))  # Camada de saída para regressão


# Compilação do modelo
model.compile(optimizer='adam', loss='mse')  # Utilizando o erro quadrático médio como função de perda


qntEpochs = 50

# Gerando e salvando a visualização do modelo
epoch_range = range(1, qntEpochs + 1)
tipo = "CNN-Relu-Swish-Maior-" + str(qntEpochs) +"Batch25"
nomeArquivo='Estruturas/Estrutura'+tipo+'.txt'
with open(nomeArquivo, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

#  Treinamento do modelo

# Definindo o Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_reshaped, y_train, epochs=qntEpochs,
                    batch_size=25,
                    validation_data=(X_test_reshaped, y_test),
                    # callbacks=[early_stopping]
                    )

# Plot da loss ao decorrer das épocas
plt.figure(figsize=(4.5, 3))
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], label='train')
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.xlim(1, len(history.history['loss']))  # Definindo os limites do eixo x de acordo com o número real de épocas
plt.legend()

# Ajustando o layout para evitar que a imagem fique cortada
plt.tight_layout()

# Salvando a imagem como .png
plt.savefig(caminho_pasta+'/Imagens/'+tipo+'.png')
plt.show()

y_pred = model.predict(X_test)

# Transforma y_pred em um array unidimensional, se necessário
if len(y_pred.shape) > 1:
    y_pred = y_pred.flatten()

# Cálculo das métricas
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
weight_factor = 2  # Peso maior para erros negativos
rmseP = weighted_rmse(y_test, y_pred, weight_factor)
maeP = weighted_mae(y_test, y_pred, weight_factor)


# Verificando se o Early Stopping ocorreu
if len(history.epoch) < qntEpochs:
    ocorreu = "Sim"
    qnt_epocas = len(history.epoch)
else:
    ocorreu = "Não"
    qnt_epocas = qntEpochs

# Criando o dicionário com os resultados
resultados = {
    "Tipo": [tipo],
    "R²": [r2],
    "MAE": [mae],
    "MAE Ponderado": [maeP],
    "RMSE": [rmse],
    "RMSE Ponderado": [rmseP],
    "Early Stopping": [ocorreu],
    "Quantidade de Épocas": [qnt_epocas]

}

# Convertendo o dicionário para um DataFrame
df_resultados = pd.DataFrame(resultados)

# Salvando os resultados no arquivo CSV
df_resultados.to_csv("resultados.csv", mode='a', header=False, index=False)