import matplotlib.pyplot as plt

# Train Loss for each client
data = {
    1: [15361.7363, 8096.0428, 4626.0184, 3300.7576, 2637.4792, 1713.2466, 
        1359.4513, 1316.6869, 509.8453, 614.8479, 187.5010, 114.0374, 
        42.8255, 12.1522, 8.5517, 5.5603, 3.3403, 3.9174, 2.4378, 5.3581],
    2: [18329.5858, 4981.2638, 2224.2365, 4247.1487, 2236.6547, 1969.5735, 
        1703.9630, 1251.2980, 565.1522, 276.2796, 244.4076, 80.8499, 
        44.5454, 15.7949, 7.3902, 4.9691, 3.5027, 6.9600, 1.3925, 4.4286],
    3: [16492.5940, 4431.8050, 3788.6454, 3284.5280, 2200.6123, 1724.7867, 
        2052.5530, 1066.3281, 1038.1011, 627.5579, 304.4678, 129.1185,
        41.2693, 20.2098, 6.2773, 4.9097, 3.9005, 2.1355, 1.7149, 3.5349],
    4: [15358.6902, 6549.4494, 4375.6836, 2851.2919, 3432.6349, 2136.7913, 
        1982.6625, 1380.4287, 721.2446, 413.7379, 283.9529, 108.5089,
        48.8658, 28.4230, 10.3050, 7.4228, 2.2920, 1.6114, 2.1187, 2.0974]
}

def plot_and_save(client_id, train_loss):
    epochs = list(range(len(train_loss)))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, marker='o', label=f'Train Loss (Client {client_id})', color='blue')

    plt.title(f"Train Loss vs. Epochs (Client {client_id})", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Train Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    filename = f"client_{client_id}_loss.png"
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory

# Generate and save plots for all clients
for client_id, train_loss in data.items():
    plot_and_save(client_id, train_loss)

print("Plots saved as client_n.png for all clients.")
