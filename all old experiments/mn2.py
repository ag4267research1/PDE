from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Define annular domain parameters
R1, R2 = 0.3, 1.0  # Inner and outer radii
num_radial = 30  # Number of radial divisions
num_circum = 100  # Number of circumferential divisions

# Generate a structured annular mesh manually
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, "triangle", 2, 2)

# Corrected number of vertices
editor.init_vertices((num_radial + 1) * num_circum)
editor.init_cells(2 * num_radial * num_circum)

# Create mesh points
for i in range(num_radial + 1):
    r = R1 + (R2 - R1) * i / num_radial
    for j in range(num_circum):
        theta = 2 * np.pi * j / num_circum
        editor.add_vertex(i * num_circum + j, Point(r * np.cos(theta), r * np.sin(theta)))

# Generate triangular cells
for i in range(num_radial):
    for j in range(num_circum):
        v1 = i * num_circum + j
        v2 = i * num_circum + (j + 1) % num_circum
        v3 = (i + 1) * num_circum + j
        v4 = (i + 1) * num_circum + (j + 1) % num_circum
        editor.add_cell(2 * (i * num_circum + j), [v1, v2, v3])
        editor.add_cell(2 * (i * num_circum + j) + 1, [v2, v4, v3])

editor.close()

# Define function spaces for the mixed formulation
V = FunctionSpace(mesh, "Lagrange", 2)  # Displacement u
W = FunctionSpace(mesh, "Lagrange", 1)  # Laplacian w

# Create a mixed finite element
element = MixedElement([V.ufl_element(), W.ufl_element()])
V2 = FunctionSpace(mesh, element)

# Define trial and test functions
U = TrialFunction(V2)
V_test = TestFunction(V2)
(u, w) = split(U)
(v, z) = split(V_test)

# Define parameters tau and g(x, y)
tau = Constant(5.0)
g = Expression("sin(pi*x[0])*cos(pi*x[1])", degree=2)  # Non-uniform forcing function

# Define bilinear and linear forms
a = (inner(grad(u), grad(v)) * dx - w * v * dx + tau * u * v * dx + 
     inner(grad(w), grad(z)) * dx + tau * w * z * dx)
L = g * z * dx

# Define boundary conditions
class InnerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(np.sqrt(x[0]**2 + x[1]**2), R1, 1e-3)

class OuterBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(np.sqrt(x[0]**2 + x[1]**2), R2, 1e-3)

inner_boundary = InnerBoundary()
outer_boundary = OuterBoundary()

bc_u_inner = DirichletBC(V2.sub(0), Constant(0.0), inner_boundary)
bc_w_inner = DirichletBC(V2.sub(1), Constant(0.0), inner_boundary)
bc_u_outer = DirichletBC(V2.sub(0), Constant(0.0), outer_boundary)
bc_w_outer = DirichletBC(V2.sub(1), Constant(0.0), outer_boundary)

bcs = [bc_u_inner, bc_w_inner, bc_u_outer, bc_w_outer]

# Solve the system
U_sol = Function(V2)
solve(a == L, U_sol, bcs, solver_parameters={"linear_solver": "mumps"})

# Extract solutions
u_solution, w_solution = U_sol.split()

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))
mesh_coords = mesh.coordinates()
triang = tri.Triangulation(mesh_coords[:, 0], mesh_coords[:, 1])

tpc = ax.tripcolor(triang, u_solution.compute_vertex_values(mesh), shading='gouraud', cmap='viridis')
levels = np.linspace(min(u_solution.vector().get_local()), max(u_solution.vector().get_local()), 15)
ax.tricontour(triang, u_solution.compute_vertex_values(mesh), levels=levels, colors='k', linewidths=0.8)
fig.colorbar(tpc, ax=ax, label="Displacement")
ax.set_title(r"Solution of $(\Delta + \tau)(\Delta + \tau) u = g(x, y)$ on Annulus")
plt.show()
