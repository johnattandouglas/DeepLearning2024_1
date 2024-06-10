import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Lendo os dados do arquivo CSV
results_df = pd.read_csv("Resultados/Resultados_Epocas.csv")

# Obtendo os valores únicos de learning rates e epochs
learning_rates = results_df['Learning Rate'].unique()
qntEpochs = results_df['Epoch'].max()

# --------------------------------------------
# Define a colormap# Define a colormap
colormap = matplotlib.cm.get_cmap('tab10', len(learning_rates))
size = (6, 4)


# Plotting the training loss
plt.figure(figsize=size)
for i, lr in enumerate(learning_rates):
    plt.plot(results_df[results_df['Learning Rate'] == lr]['Epoch'],
             results_df[results_df['Learning Rate'] == lr]['Loss'],
             label='LR = ' + f'{lr:.0e}',
             color=colormap(i))
plt.xlabel('Epochs')
plt.xlim(1, qntEpochs)
plt.ylabel('Training Loss')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('Imagens/TrainingLoss2.png')
plt.show()
plt.close()

# Plotting the validation loss
plt.figure(figsize=size)
for i, lr in enumerate(learning_rates):
    plt.plot(results_df[results_df['Learning Rate'] == lr]['Epoch'],
             results_df[results_df['Learning Rate'] == lr]['Validation Loss'],
             label='LR = ' + f'{lr:.0e}',
             color=colormap(i))

plt.xlabel('Epochs')
plt.xlim(1, qntEpochs)
plt.ylabel('Validation Loss')
plt.yscale('log')
# plt.legend(loc='upper right')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('Imagens/ValidationLoss2.png')
plt.show()
plt.close()