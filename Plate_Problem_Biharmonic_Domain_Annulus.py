from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from mshr import *

# Define annular domain (outer radius R2, inner radius R1)
R1, R2 = 0.3, 1.0  # Inner and outer radii
domain = Circle(Point(0, 0), R2) - Circle(Point(0, 0), R1)  # Annulus as a difference of circles

# Create mesh
mesh = generate_mesh(domain, 64)

# Define function spaces for the mixed formulation
V = FunctionSpace(mesh, "Lagrange", 2)  # Displacement u
W = FunctionSpace(mesh, "Lagrange", 1)  # Laplacian w

# Create a mixed finite element
element = MixedElement([V.ufl_element(), W.ufl_element()])
V2 = FunctionSpace(mesh, element)  # Mixed function space for (u, w)

# Define trial and test functions
(u, w) = TrialFunctions(V2)
(v, z) = TestFunctions(V2)

# Define parameters q_0 and D
q_0 = Constant(1.0)  # Load intensity (modifiable)
D = Constant(10.0)   # Rigidity parameter (modifiable)

# Define general forcing function f = q_0 / D
f = q_0 / D  # Uniform load over the annulus

# Define bilinear and linear forms for the mixed system
a = (inner(grad(u), grad(v)) * dx - w * v * dx + inner(grad(w), grad(z)) * dx)  # Mixed formulation
L = f * z * dx  # Load term applies only to second equation

# Define boundary conditions
class InnerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(np.sqrt(x[0]**2 + x[1]**2), R1, 1e-3)

class OuterBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(np.sqrt(x[0]**2 + x[1]**2), R2, 1e-3)

inner_boundary = InnerBoundary()
outer_boundary = OuterBoundary()

bc_u_inner = DirichletBC(V2.sub(0), Constant(0.0), inner_boundary)  # u = 0 on inner boundary
bc_w_inner = DirichletBC(V2.sub(1), Constant(0.0), inner_boundary)  # w = 0 on inner boundary

bc_u_outer = DirichletBC(V2.sub(0), Constant(0.0), outer_boundary)  # u = 0 on outer boundary
bc_w_outer = DirichletBC(V2.sub(1), Constant(0.0), outer_boundary)  # w = 0 on outer boundary

bcs = [bc_u_inner, bc_w_inner, bc_u_outer, bc_w_outer]

# Solve the system using a direct solver
U = Function(V2)
solve(a == L, U, bcs, solver_parameters={"linear_solver": "mumps"})

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

    # Set title with LaTeX formatting
    ax.set_title(r"Biharmonic Solution on Annulus with $u = 0$ and $\frac{\partial u}{\partial n} = 0$")

    # Show plot
    plt.show()
