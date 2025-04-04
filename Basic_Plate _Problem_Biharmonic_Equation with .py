from dolfin import *

# Create a mesh over the unit square
mesh = UnitSquareMesh(8, 8)

# Define a function space
V = FunctionSpace(mesh, "P", 1)

# Define boundary condition
u_D = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define the variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Compute the solution
u = Function(V)
solve(a == L, u, bc)

# Save and plot the solution
plot(u)
import matplotlib.pyplot as plt
plt.show()
