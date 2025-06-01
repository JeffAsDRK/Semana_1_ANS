import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap

# Cargar datos
X, y = load_digits(return_X_y=True)

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# ICA
ica = FastICA(n_components=3, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X)

# Isomap
isomap = Isomap(n_components=3, n_neighbors=5, n_jobs=-1)
X_iso = isomap.fit_transform(X)

# Función para graficar
def plot_3d(X_transformed, title):
    fig = px.scatter_3d(
        x=X_transformed[:, 0], y=X_transformed[:, 1], z=X_transformed[:, 2],
        color=y.astype(str), title=title,
        labels={'color': 'Digit'}
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    fig.show()

# Mostrar resultados
plot_3d(X_pca, "PCA - Componentes principales")
plot_3d(X_ica, "ICA - Componentes independientes")
plot_3d(X_iso, "Isomap - Geometría no lineal")