import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dolfin import *

# Define the mesh (plate geometry)
N = 32  # Number of elements in each direction
mesh = UnitSquareMesh(N, N)  # Create a unit square mesh

# Define the function space (H^2 space for biharmonic equation)
V = FunctionSpace(mesh, "Lagrange", 2)  # Quadratic elements

# Define boundary conditions (clamped edges: w = 0 and ∂w/∂n = 0)
def clamped_boundary(x, on_boundary):
    return on_boundary

bc = [DirichletBC(V, Constant(0), clamped_boundary)]

# Define the problem
w = TrialFunction(V)  # Unknown function
v = TestFunction(V)  # Test function
q0 = Constant(1.0)  # Load on the plate
D = Constant(1.0)  # Flexural rigidity

# Weak form: Integrating by parts twice
a = inner(div(grad(w)), div(grad(v))) * dx
L = (q0 / D) * v * dx

# Solve the problem
w_sol = Function(V)
solve(a == L, w_sol, bc)

# Extract values for plotting
coords = mesh.coordinates()
values = np.array([w_sol(Point(x, y)) for x, y in coords])

# Reshape into a grid for 3D plotting
X = coords[:, 0].reshape((N+1, N+1))
Y = coords[:, 1].reshape((N+1, N+1))
Z = values.reshape((N+1, N+1))

# Plot the 3D surface
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap="viridis")

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Deflection w(x,y)")
ax.set_title("3D Plot of Clamped Plate Deflection")

plt.show()
