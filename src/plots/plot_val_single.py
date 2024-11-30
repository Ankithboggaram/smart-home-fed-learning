import matplotlib.pyplot as plt

data = {
    # 1: [15361.7363, 8096.0428, 4626.0184, 3300.7576, 2637.4792, 1713.2466, 
      1:[1316.6869, 509.8453, 614.8479, 187.5010, 114.0374, 
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

validation_data = {
    # 1: [67691.3710, 288742.6442, 109508.1820, 190517.0342, 112088.3294, 147858.6198,
    1:  [35721.0988, 22544.5627, 1726.3138,
        774.5928, 74.1893, 90.9800, 93.8145, 194.2773, 141.4130,
        110.7157, 338.1427, 96.5438, 39.8007],
    2: [73414.1010, 254729.5248, 101924.7622, 163658.8214, 109342.2791,
        134855.0013, 61945.9997, 33991.1120, 20918.4631, 1872.2179,
        689.6583, 70.4321, 87.0144, 93.5562, 199.0143, 127.6342,
        105.9122, 321.9897, 97.3187, 40.5010],
    3: [66032.2198, 284211.5130, 118723.0481, 177715.0390, 115865.1379,
        152411.0105, 69000.3401, 35602.0042, 21456.2291, 1634.1100,
        758.0021, 72.3001, 89.1100, 93.2113, 198.9102, 140.1110,
        110.1180, 329.1247, 95.8721, 39.9980],
    4: [67914.4312, 287134.6231, 119562.1292, 191201.1201, 114981.2310,
        150923.1004, 69010.3023, 35001.1101, 21015.8931, 1711.0023,
        770.8931, 76.3210, 91.0012, 93.7120, 196.4500, 142.0023,
        112.3410, 335.7891, 96.4512, 39.9082]
}

def plot_and_save(client_id, train_loss, val_loss):
    epochs = list(range(len(train_loss)))
    plt.figure(figsize=(10, 6))
    
    # Plotting train loss
    plt.plot(epochs, train_loss, marker='o', label=f'Train Loss (Client {client_id})', color='blue')
    
    # Plotting validation loss
    plt.plot(epochs, val_loss, marker='x', label=f'Validation Loss (Client {client_id})', color='orange')
    
    # Adding titles and labels
    plt.title(f"Train and Validation Loss vs. Epochs (Client {client_id})", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = f"client_{client_id}_loss.png"
    plt.savefig(filename)
    plt.close()

# Generate plots for each client
# for client_id in data.keys():
    # plot_and_save(client_id, data[client_id], validation_data[client_id])

plot_and_save(1, data[1], validation_data[1])

print("Plots saved as client_n_loss.png for all clients.")