import matplotlib.pyplot as plt

# Nilai metrik
metrics = ["Akurasi", "Macro F1", "Weighted F1"]
values = [0.7100, 0.7098, 0.7125]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, values, color=["skyblue", "lightgreen", "salmon"])
plt.ylim(0.6, 0.75)
plt.title("Hasil Evaluasi Sistem Resolusi Konflik", fontsize=14)
plt.ylabel("Skor")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Tampilkan nilai di atas bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()
