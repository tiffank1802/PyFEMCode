from fenics import *

# Créer un maillage du domaine
mesh = UnitSquareMesh(32, 32)

# Définir l'espace fonctionnel
V = FunctionSpace(mesh, 'P', 1)

# Conditions aux limites
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Définir le problème variationnel
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)  # -Δu = -6
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Résoudre
u = Function(V)
solve(a == L, u, bc)

# Visualiser
import matplotlib.pyplot as plt
plot(u)
plt.show()