from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap

# Generar datos Swiss Roll
X, color = make_swiss_roll(n_samples=1500, noise=0.05)

# Visualización en 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Swiss Roll en 3D")
plt.show()

# Aplicar Isomap
isomap = Isomap(n_neighbors=10, n_components=2)
X_iso = isomap.fit_transform(X)

# Visualización en 2D
plt.figure(figsize=(8, 6))
plt.scatter(X_iso[:, 0], X_iso[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Proyección en 2D usando Isomap")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()