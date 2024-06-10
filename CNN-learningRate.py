import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib

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

# Ajuste da forma dos dados de entrada para a CNN
X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))


# Definindo as taxas de aprendizado a serem testadas
learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
qntEpochs = 100

# Create an empty list to store dictionaries
results_list = []

# Create lists to store losses for plotting
train_loss_list = []
val_loss_list = []

for lr in learning_rates:
    model = Sequential()
    model.add(Input(shape=(X_train_reshaped.shape[1], 1)))
    model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # Flatten the data to connect to a dense layer
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1))  # Output layer for regression

    # Compilation of the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')  # Using mean squared error as loss function

    history = model.fit(X_train_reshaped, y_train, epochs=qntEpochs,
                        batch_size=25,
                        validation_data=(X_test_reshaped, y_test))
    
     # Save the results of each epoch
    for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss']), 1):
        results_list.append({'Learning Rate': lr, 'Epoch': epoch, 'Loss': loss, 'Validation Loss': val_loss})
        train_loss_list.append(loss)
        val_loss_list.append(val_loss)

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(results_list)

# Save the results to a CSV file
results_df.to_csv('Resultados_Epocas.csv', index=False)

# --------------------------------------------
# Define a colormap# Define a colormap
colormap = matplotlib.cm.get_cmap('tab10', len(learning_rates))

# Plotting the training loss
for i, lr in enumerate(learning_rates):
    plt.plot(results_df[results_df['Learning Rate'] == lr]['Epoch'],
             results_df[results_df['Learning Rate'] == lr]['Loss'],
             label='Learning Rate = ' + f'{lr:.0e}',
             color=colormap(i))
plt.xlabel('Epochs')
plt.xlim(1, qntEpochs)
plt.ylabel('Training Loss')
plt.legend()
plt.tight_layout()
plt.savefig('Imagens/TrainingLoss.png')
plt.show()
plt.close()

# Plotting the validation loss
for i, lr in enumerate(learning_rates):
    plt.plot(results_df[results_df['Learning Rate'] == lr]['Epoch'],
             results_df[results_df['Learning Rate'] == lr]['Validation Loss'],
             label='Learning Rate = ' + f'{lr:.0e}',
             color=colormap(i))

plt.xlabel('Epochs')
plt.xlim(1, qntEpochs)
plt.ylabel('Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('Imagens/ValidationLoss.png')
plt.show()
plt.close()