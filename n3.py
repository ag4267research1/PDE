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
element = MixedElement([V.ufl_element(), V.ufl_element()])
V2 = FunctionSpace(mesh, element)  # Mixed function space for (u, w)

# Define trial and test functions
(u, w) = TrialFunctions(V2)
(v, z) = TestFunctions(V2)

# Define uniform forcing function f = 1
f = Constant(1.0)

# Define bilinear and linear forms for the mixed system
a = (inner(grad(u), grad(v)) * dx - w * v * dx +
     inner(grad(w), grad(z)) * dx)  # Mixed formulation
L = f * z * dx  # Load term applies only to second equation

# # Define parameters q_0 and D
# q_0 = Constant(1.0)  # Load intensity (modifiable)
# D = Constant(10.0)   # Rigidity parameter (modifiable)

# # Define general forcing function f = q_0 / D (you can replace it with any function)
# f = q_0 / D  # Generalized form

# # Define bilinear and linear forms for the mixed system
# a = (inner(grad(u), grad(v)) * dx - w * v * dx + inner(grad(w), grad(z)) * dx)  # Mixed formulation
# L = f * z * dx  # Load term applies only to second equation

# Define boundary conditions
def clamped_boundary(x, on_boundary):
    return on_boundary

bc_u = DirichletBC(V2.sub(0), Constant(0.0), clamped_boundary)  # u = 0
bc_w = DirichletBC(V2.sub(1), Constant(0.0), clamped_boundary)  # w = 0

# Solve the system using a direct solver
U = Function(V2)
solve(a == L, U, [bc_u, bc_w], solver_parameters={"linear_solver": "mumps"})

# Extract solutions
u_solution, w_solution = U.split()

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
    ax.set_title(r"Biharmonic Solution with $u = 0$ and $\frac{\partial u}{\partial n} = 0$, $g = \frac{q_0}{D}$")
    #ax.set_title(r"Biharmonic Solution with $u = 0$ and $\frac{\partial u}{\partial n} = 0$, $g = 1$")

    # Show plot
    plt.show()
