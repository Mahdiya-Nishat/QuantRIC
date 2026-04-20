import matplotlib.pyplot as plt

epochs  = list(range(1, 21))
total   = [0.2918, 0.2040, 0.1933, 0.1642, 0.1543, 0.1490, 0.1518, 0.1326,
           0.1256, 0.1226, 0.1216, 0.1070, 0.1030, 0.0994, 0.0962, 0.0969,
           0.1010, 0.0946, 0.0879, 0.0899]
sensing = [0.1447, 0.1051, 0.0973, 0.0841, 0.0831, 0.0776, 0.0765, 0.0735,
           0.0706, 0.0665, 0.0654, 0.0613, 0.0580, 0.0571, 0.0533, 0.0551,
           0.0549, 0.0508, 0.0471, 0.0489]
comm    = [0.1471, 0.0989, 0.0960, 0.0801, 0.0712, 0.0714, 0.0754, 0.0591,
           0.0550, 0.0561, 0.0562, 0.0457, 0.0451, 0.0423, 0.0429, 0.0418,
           0.0461, 0.0438, 0.0408, 0.0410]

plt.figure(figsize=(10, 5))
plt.plot(epochs, total,   'k-o',  label='Total Loss',    linewidth=2)
plt.plot(epochs, sensing, 'b-s',  label='Sensing Loss',  linewidth=2)
plt.plot(epochs, comm,    'r-^',  label='Comm Loss',     linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('QuantRIC — Sensing vs Comm Branch Loss Separation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curves.png', dpi=150)
plt.show()
print("Saved loss_curves.png")