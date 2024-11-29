import matplotlib.pyplot as plt

averaged_rmse = [
    4517795.25, 1856596.53, 1153651.39, 1029271.66, 916430.41, 685609.56,
    543492.66, 401060.77, 238857.10, 126901.56, 96284.32, 31659.09,
    12306.92, 6569.60, 3023.35, 1805.77, 908.06, 1705.99, 559.10, 1213.38
]

plt.figure(figsize=(10, 6))
plt.plot(range(20), averaged_rmse, marker='o', linestyle='-', color='r', label='RMSE')

plt.title('RMSE of shared model over 20 Epochs')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.xticks(range(20))  # Ensure all epochs are labeled on the x-axis
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig("RMSE")
