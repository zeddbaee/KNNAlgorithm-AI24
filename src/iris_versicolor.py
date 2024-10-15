import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans

# Data latih tanpa label
data_train = np.array([
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5],
    [5.7, 2.8, 4.5, 1.3],
    [6.3, 3.3, 4.7, 1.6],
    [4.9, 2.4, 3.3, 1.0],
    [6.6, 2.9, 4.6, 1.3],
    [5.2, 2.7, 3.9, 1.4]
])

# Langkah awal: Melakukan clustering pada data latih untuk membuat label sementara
n_clusters = 2  # Misalnya kita asumsikan 2 kelompok
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data_train)

# Label sementara berdasarkan hasil clustering
labels = kmeans.labels_

# Data uji
data_test = np.array([5.0, 2.0, 3.5, 1.0])

# Jumlah tetangga terdekat (K)
K = 3

# Hitung jarak antara data uji dengan semua data latih menggunakan jarak Euclidean
distances = np.sqrt(np.sum((data_train - data_test) ** 2, axis=1))

# Buat DataFrame untuk menampilkan hasil dalam bentuk tabel
data_with_labels = pd.DataFrame(data_train, columns=['x1', 'x2', 'x3', 'x4'])
data_with_labels['label'] = labels
data_with_labels['jarak'] = distances
data_with_labels['no'] = range(1, len(data_train) + 1)

# Urutkan berdasarkan jarak dari yang terdekat ke yang terjauh
data_with_labels = data_with_labels.sort_values(by='jarak').reset_index(drop=True)

# Pilih K data dengan jarak terdekat
nearest_data = data_with_labels.head(K)

# Hitung frekuensi masing-masing kategori pada K data yang telah dipilih
nearest_labels = nearest_data['label']
label_counts = Counter(nearest_labels)

# Tentukan label yang paling banyak muncul di antara K tetangga
predicted_label = label_counts.most_common(1)[0][0]

# Tambahkan data uji ke dalam tabel sebagai entri terakhir
data_test_row = pd.DataFrame({
    'no': [len(data_train) + 1],
    'x1': [data_test[0]],
    'x2': [data_test[1]],
    'x3': [data_test[2]],
    'x4': [data_test[3]],
    'label': [predicted_label],
    'jarak': [0]  # Jarak ke diri sendiri adalah 0
})

# Gabungkan data latih dengan data uji
final_table = pd.concat([data_with_labels, data_test_row], ignore_index=True)

# Output tabel tanpa penomoran otomatis
print("Tabel Data Latih dengan Jarak ke Data Uji (termasuk data uji):")
print(final_table[['no', 'x1', 'x2', 'x3', 'x4', 'label', 'jarak']].to_string(index=False))
print("\nLabel prediksi untuk data uji:", predicted_label)