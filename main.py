import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Discrétisation
N = 10  # Nombre de nœuds
L = 1.0  # Longueur du domaine
x = np.linspace(0, L, N)

# Construction de la matrice de rigidité et du vecteur force
A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N)).tocsr()
A = A * (N-1) / L  # Ajustement pour la discrétisation
f = np.ones(N) * (L / (N-1))  # Source uniforme

# Conditions aux limites (u(0)=0, u(L)=0)
A[0, :] = 0; A[0, 0] = 1; f[0] = 0
A[-1, :] = 0; A[-1, -1] = 1; f[-1] = 0

# Résolution
u = spsolve(A, f)

# Visualisation
plt.plot(x, u)
plt.xlabel('x'); plt.ylabel('u(x)')
plt.grid(True)
plt.show()


