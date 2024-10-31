import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# DATASET IRIS
iris = load_iris()
x = iris.data
y = iris.target

# MEMBAGI DATASET MENJADI DATA LATIH DAN DATA UJI
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# INISIALISASI MODEL KNN
k = 10
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(x_train, y_train)

# PREDIKSI MODEL
y_pred = knn.predict(x_test)

# HITUNG AKURASI MODEL
accuracy = accuracy_score(y_test, y_pred)
print(f"AKURASI MODEL DENGAN (K = {k}) : {accuracy:.2f}")

# DATAFRAME HASIL PREDIKSI
results_df = pd.DataFrame({
    'Sepal Length': x_test[:, 0],
    'Sepal Width': x_test[:, 1],
    'Petal Length': x_test[:, 2],
    'Petal Width': x_test[:, 3],
    'Spesies Prediksi': [iris.target_names[i] for i in y_pred],
    'Spesies Asli': [iris.target_names[i] for i in y_test]
})

print(f"\nTABEL PREDIKSI :")
print(results_df)
