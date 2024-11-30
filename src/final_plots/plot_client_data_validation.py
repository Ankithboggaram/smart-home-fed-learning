import matplotlib.pyplot as plt

# Training and validation loss data
data = {
    1: [1980.309,1410.3668,990.8185,621.5934,315.5289,123.5966,26.9165,4.0805,2.312,1.7301,1.1995,0.7287,0.2171,0.2334,0.144,0.1391,0.4702,0.133,0.1314,0.1337],
    2: [2592.3378,1827.0007,995.0906,1128.773,92.15,90.44,13.9649,2.7097,1.8654,1.8738,0.886,0.3959,0.2255,0.1208,0.087,0.0606,0.0984,0.0752,0.1588, 0.0559],
    3: [3667.8155,1381.7182,839.7368,540.3345,148.9166,110.1008,24.5757,4.3596,1.8971,1.6264,0.9321,0.4849,0.2738,0.1867,0.1974,0.0898,0.071,0.1821,0.1713,0.2117],
    4: [2817.981,1219.0473,1142.1698,626.9621,148.7916,133.3995,25.1641,2.9523,2.3126,1.1467,0.9344,0.5759,0.2374,0.1611,0.1867,0.0978,0.1984,0.1211,0.0915,0.0975]
}

validation_data = {
    1:  [71208.9968,3259.6571,9758.1033,42910.1542,7270.5239,824.268,1533.3747,482.2531,356.3806,321.7475,87.8196,39.6163,37.8568,30.4816,37.7387,29.745,79.6777,30.8801,29.9423,29.4047],
    2: [18855.892,74712.6459,41139.0599,46397.9693,9465.7788,2660.4504,525.1512,122.9978,175.9807,330.0764,97.1975,41.3629,42.9928,30.7218,31.1636,30.3301,33.4875,29.5494,56.2136,30.823],
    3: [59861.5506,146203.4054,7930.5072,13306.8561,1064.6371,4718.1432,118.2785,666.7306,55.6372,205.2057,77.0193,32.2131,39.1332,29.6488,31.0009,29.6054,30.2576,43.0719,32.318,40.5308],
    4: [142080.7803,161839.4927,36705.7219,12129.1712,289.9356,2148.0095,366.2013,240.1281,150.8256,93.8323,356.1994,29.7204,33.4838,30.573,44.4993,32.2162,40.6871,53.8871,29.2127,29.2125]
}

# Function to plot train and validation loss
def plot_and_save(client_id, train_loss, val_loss):
    epochs = list(range(len(train_loss)))
    plt.figure(figsize=(10, 12))  # Increase height for two plots
    
    # Train loss plot
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    plt.plot(epochs, train_loss, marker='o', label=f'Train Loss (Client {client_id})', color='blue')
    plt.title(f"Train Loss vs. Epochs (Client {client_id})", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Train Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Validation loss plot
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    plt.plot(epochs, val_loss, marker='x', label=f'Validation Loss (Client {client_id})', color='orange')
    plt.title(f"Validation Loss vs. Epochs (Client {client_id})", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = f"client_{client_id}_train_val_loss.png"
    plt.savefig(filename)
    plt.close()

for client_id in data.keys():
    plot_and_save(client_id, data[client_id], validation_data[client_id])


print("Plots saved as client_n_loss.png for all clients.")
