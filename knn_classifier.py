from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. Split dataset jadi train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Buat model KNN
knn = KNeighborsClassifier(n_neighbors=3)

# 4. Train model
knn.fit(X_train, y_train)

# 5. Prediksi data test
y_pred = knn.predict(X_test)

# 6. Evaluasi akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')
