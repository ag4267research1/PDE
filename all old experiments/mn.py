# from dolfin import *
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as tri

# # Define annular domain parameters
# R1, R2 = 0.3, 1.0  # Inner and outer radii
# num_radial = 30  # Number of radial divisions
# num_circum = 100  # Number of circumferential divisions

# # Generate a structured annular mesh manually
# mesh = Mesh()
# editor = MeshEditor()
# editor.open(mesh, "triangle", 2, 2)

# # Corrected number of vertices
# editor.init_vertices((num_radial + 1) * num_circum)  # ✅ Fix
# editor.init_cells(2 * num_radial * num_circum)  # Number of elements

# # Create mesh points
# for i in range(num_radial + 1):
#     r = R1 + (R2 - R1) * i / num_radial
#     for j in range(num_circum):
#         theta = 2 * np.pi * j / num_circum
#         editor.add_vertex(i * num_circum + j, Point(r * np.cos(theta), r * np.sin(theta)))  # ✅ Use correct index

# # Generate triangular cells (✅ Fix `add_cell()` to use lists)
# for i in range(num_radial):
#     for j in range(num_circum):
#         v1 = i * num_circum + j
#         v2 = i * num_circum + (j + 1) % num_circum
#         v3 = (i + 1) * num_circum + j
#         v4 = (i + 1) * num_circum + (j + 1) % num_circum
#         editor.add_cell(2 * (i * num_circum + j), [v1, v2, v3])  # ✅ Fix
#         editor.add_cell(2 * (i * num_circum + j) + 1, [v2, v4, v3])  # ✅ Fix

# editor.close()

# # Define function spaces for the mixed formulation
# V = FunctionSpace(mesh, "Lagrange", 2)  # Displacement u
# W = FunctionSpace(mesh, "Lagrange", 1)  # Laplacian w

# # Create a mixed finite element
# element = MixedElement([V.ufl_element(), W.ufl_element()])
# V2 = FunctionSpace(mesh, element)  # Mixed function space for (u, w)

# # Define trial and test functions
# U = TrialFunction(V2)
# V_test = TestFunction(V2)
# (u, w) = split(U)
# (v, z) = split(V_test)

# # Define parameters tau and g
# tau = Constant(5.0)  # Positive parameter tau
# g = Constant(1.0)    # Forcing function

# # Define bilinear and linear forms for the mixed system (natural boundary conditions)
# a = (inner(grad(u), grad(v)) * dx - w * v * dx + tau * u * v * dx +  # (Δ + τ) u = w
#      inner(grad(w), grad(z)) * dx + tau * w * z * dx)  # (Δ + τ) w = g

# L = g * z * dx  # Load term applies only to second equation

# # Define boundary conditions for clamped boundary (Dirichlet: u = 0, w = 0)
# class InnerBoundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and near(np.sqrt(x[0]**2 + x[1]**2), R1, 1e-3)

# class OuterBoundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and near(np.sqrt(x[0]**2 + x[1]**2), R2, 1e-3)

# inner_boundary = InnerBoundary()
# outer_boundary = OuterBoundary()

# bc_u_inner = DirichletBC(V2.sub(0), Constant(0.0), inner_boundary)  # u = 0 on inner boundary
# bc_w_inner = DirichletBC(V2.sub(1), Constant(0.0), inner_boundary)  # w = 0 on inner boundary

# bc_u_outer = DirichletBC(V2.sub(0), Constant(0.0), outer_boundary)  # u = 0 on outer boundary
# bc_w_outer = DirichletBC(V2.sub(1), Constant(0.0), outer_boundary)  # w = 0 on outer boundary

# bcs = [bc_u_inner, bc_w_inner, bc_u_outer, bc_w_outer]

# # Solve the system using a direct solver
# U_sol = Function(V2)
# solve(a == L, U_sol, bcs, solver_parameters={"linear_solver": "mumps"})

# # Extract solutions
# u_solution, w_solution = U_sol.split()

# # Check for NaN or Inf values before plotting
# u_values = u_solution.vector().get_local()
# if np.isnan(u_values).any() or np.isinf(u_values).any():
#     print("⚠️ Warning: The solution contains NaN or Inf values!")
# else:
#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # Create a triangulation for plotting
#     mesh_coords = mesh.coordinates()
#     triang = tri.Triangulation(mesh_coords[:, 0], mesh_coords[:, 1])

#     # Plot using tripcolor (which supports colorbars)
#     tpc = ax.tripcolor(triang, u_solution.compute_vertex_values(mesh), shading='gouraud', cmap='viridis')

#     # Add contour lines for better visualization
#     levels = np.linspace(min(u_values), max(u_values), 15)  # Define contour levels
#     ax.tricontour(triang, u_solution.compute_vertex_values(mesh), levels=levels, colors='k', linewidths=0.8)

#     # Add colorbar
#     fig.colorbar(tpc, ax=ax, label="Displacement")

#     # Set title with LaTeX formatting
#     ax.set_title(r"Solution of $(\Delta + \tau)(\Delta + \tau) u = g$ on Annulus with $u = 0, \frac{\partial u}{\partial n} = 0$")

#     # Show plot
#     plt.show()
