from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

# Define mesh
mesh = UnitSquareMesh(64, 64)

# Define function space with higher-order elements
V = FunctionSpace(mesh, "Lagrange", 3)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define forcing function (Uniform load f = 1)
f = Constant(1.0)

# Regularization term to stabilize normal derivative enforcement
epsilon = Constant(1e-6)

# Define bilinear and linear forms
a = inner(grad(grad(u)), grad(grad(v))) * dx + epsilon * inner(grad(u), grad(v)) * ds
L = f * v * dx

# Define boundary condition (clamped: u = 0)
def clamped_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0.0), clamped_boundary)

# Solve using a direct solver (MUMPS)
u_solution = Function(V)
solve(a == L, u_solution, bc, solver_parameters={"linear_solver": "mumps"})

# Check for NaN or Inf values before plotting
u_values = u_solution.vector().get_local()
if np.isnan(u_values).any() or np.isinf(u_values).any():
    print("⚠️ Warning: The solution contains NaN or Inf values!")
else:
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a triangulation for plotting
    mesh_coords = mesh.coordinates()
    triang = tri.Triangulation(mesh_coords[:, 0], mesh_coords[:, 1], mesh.cells())

    # Plot using tripcolor (which supports colorbars)
    tpc = ax.tripcolor(triang, u_solution.compute_vertex_values(mesh), shading='gouraud', cmap='viridis')

    # Add contour lines for better visualization
    levels = np.linspace(min(u_values), max(u_values), 15)  # Define contour levels
    ax.tricontour(triang, u_solution.compute_vertex_values(mesh), levels=levels, colors='k', linewidths=0.8)

    # Add colorbar
    fig.colorbar(tpc, ax=ax, label="Displacement")

    # Set title and show plot
    ax.set_title("Clamped Plate Solution (Biharmonic, f=1) with Contours")
    plt.show()
