import matplotlib.pyplot as plt

averaged_rmse = [857150.859375, 459952.171875, 315546.75, 215718.964575, 62489.222675, 
39798.931675, 9054.105625, 1095.9475, 625.4255, 435.538, 254.496725, 
165.062425, 67.8709, 55.915125, 46.13885, 36.043475, 74.299525, 
48.608625, 45.7073, 45.810475]

plt.figure(figsize=(10, 6))
plt.plot(range(20), averaged_rmse, marker='o', linestyle='-', color='r', label='RMSE')

plt.title('RMSE of model over shared test set')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.xticks(range(20))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig("RMSE")
