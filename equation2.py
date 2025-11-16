from fenics import *
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx
u = Function(V)
solve(a == L, u)
import matplotlib.pyplot as plt
plot(u)
plt.show()
