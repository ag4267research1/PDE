from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

# Define mesh
mesh = UnitSquareMesh(64, 64)

# Define function spaces for the mixed formulation
V = FunctionSpace(mesh, "Lagrange", 2)  # Displacement u
W = FunctionSpace(mesh, "Lagrange", 1)  # Laplacian w

# Create a mixed finite element
element = MixedElement([V.ufl_element(), W.ufl_element()])
V2 = FunctionSpace(mesh, element)  # Mixed function space for (u, w)

# Define trial and test functions
U = TrialFunction(V2)
V_test = TestFunction(V2)
(u, w) = split(U)
(v, z) = split(V_test)

# Define parameters tau and g
tau = Constant(5.0)  # Positive parameter tau
g = Constant(1.0)    # Forcing function

# Define bilinear and linear forms for the mixed system
a = (inner(grad(u), grad(v)) * dx - w * v * dx + tau * u * v * dx +  # (Δ + τ) u = w
     inner(grad(w), grad(z)) * dx + tau * w * z * dx)  # (Δ + τ) w = g

L = g * z * dx  # Load term applies only to second equation

# Define boundary conditions
def clamped_boundary(x, on_boundary):
    return on_boundary

bc_u = DirichletBC(V2.sub(0), Constant(0.0), clamped_boundary)  # u = 0
bc_w = DirichletBC(V2.sub(1), Constant(0.0), clamped_boundary)  # w = 0

# Solve the system using a direct solver
U_sol = Function(V2)
solve(a == L, U_sol, [bc_u, bc_w], solver_parameters={"linear_solver": "mumps"})

# Extract solutions
u_solution, w_solution = U_sol.split()

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

    # Set title with LaTeX formatting
    ax.set_title(r"Solution of $(\Delta + \tau)(\Delta + \tau) u = g$ with $u = 0$")

    # Show plot
    plt.show()
