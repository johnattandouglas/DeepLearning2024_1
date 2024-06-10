import matplotlib.pyplot as plt

# Função para ler os dados do arquivo txt
def ler_dados(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    batch_sizes = []
    r2_scores = []
    losses = []

    for line in lines:
        parts = line.split(',')
        batch_size = int(parts[0].split(':')[1].strip())
        r2_score = float(parts[1].split(':')[1].strip())
        loss = float(parts[2].split(':')[1].strip())
        
        batch_sizes.append(batch_size)
        r2_scores.append(r2_score)
        losses.append(loss)
    
    return batch_sizes, r2_scores, losses

# Caminho para o arquivo txt
filename = 'Resultados/loss_vs_batch_size-5a50.txt'

# Lendo os dados do arquivo
batch_sizes, r2_scores, losses = ler_dados(filename)


# Gerando e salvando o gráfico R² vs Batch Size
plt.figure(figsize=(5.5, 3.5))
plt.plot(batch_sizes, r2_scores, marker='o')
plt.xlabel('Batch Size')
plt.ylabel('R²')
plt.ylim(0.85, 1.0)
plt.grid(True)
# Salvando o gráfico R²
plt.savefig('Imagens/NEW2r2_vs_batch_size-5a50.png',bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()

# Gráfico Loss
plt.figure(figsize=(5.5, 3.5))
plt.plot(batch_sizes, losses, marker='o', color='red')
plt.xlabel('Batch Size')
plt.ylabel('Loss')
plt.grid(True)
# Salvando o gráfico Loss
plt.savefig('Imagens/NEW2loss_vs_batch_size-5a50.png',bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()