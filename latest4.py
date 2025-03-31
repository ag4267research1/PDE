#!/usr/bin/env python
"""
Dolfin-X mixed formulation example on an annulus.

We wish to solve the fourth-order PDE:
      (Δ+τ)^2 u = g   in Ω,
by splitting it into a system:
      (Δ+τ)(u₀+u_h) = w   in Ω,
      (Δ+τ) w = g        in Ω.
We enforce non-homogeneous Dirichlet data on the inner boundary by using a lifting function:
      u₀(x) = f + h*(sqrt(x[0]^2+x[1]^2) - R_inner),
so that u = u₀ on the inner boundary.
We then solve for the correction u_h (with u_h = 0 on the inner boundary) and recover the full solution:
      u = u₀ + u_h.
The outer boundary is left with natural (Neumann) conditions.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# --- Gmsh and Meshio imports ---
import gmsh
import meshio

# --- Dolfin-X and MPI imports ---
from mpi4py import MPI
from dolfinx import mesh, fem, io
import ufl
from ufl import div, grad, dx, inner
from ufl.finiteelement import FiniteElement  # Import FiniteElement from ufl.finiteelement

comm = MPI.COMM_WORLD

# =============================================================================
# 1. Generate the Annular Mesh Using Gmsh and Convert to XDMF
# =============================================================================
lc      = 0.05         # mesh size
R_inner = 1.0          # inner radius
R_outer = 2.0          # outer radius

gmsh.initialize()
gmsh.model.add("annulus")

# Define center point.
p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)

# Inner circle points.
p1 = gmsh.model.geo.addPoint(R_inner, 0.0, 0.0, lc)
p2 = gmsh.model.geo.addPoint(0.0, R_inner, 0.0, lc)
p3 = gmsh.model.geo.addPoint(-R_inner, 0.0, 0.0, lc)
p4 = gmsh.model.geo.addPoint(0.0, -R_inner, 0.0, lc)

# Outer circle points.
q1 = gmsh.model.geo.addPoint(R_outer, 0.0, 0.0, lc)
q2 = gmsh.model.geo.addPoint(0.0, R_outer, 0.0, lc)
q3 = gmsh.model.geo.addPoint(-R_outer, 0.0, 0.0, lc)
q4 = gmsh.model.geo.addPoint(0.0, -R_outer, 0.0, lc)

# Create inner arcs.
arc1 = gmsh.model.geo.addCircleArc(p1, p0, p2)
arc2 = gmsh.model.geo.addCircleArc(p2, p0, p3)
arc3 = gmsh.model.geo.addCircleArc(p3, p0, p4)
arc4 = gmsh.model.geo.addCircleArc(p4, p0, p1)

# Create outer arcs.
arc5 = gmsh.model.geo.addCircleArc(q1, p0, q2)
arc6 = gmsh.model.geo.addCircleArc(q2, p0, q3)
arc7 = gmsh.model.geo.addCircleArc(q3, p0, q4)
arc8 = gmsh.model.geo.addCircleArc(q4, p0, q1)

# Create curve loops.
inner_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
outer_loop = gmsh.model.geo.addCurveLoop([arc5, arc6, arc7, arc8])
surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("annulus.msh")
gmsh.finalize()
print("Mesh generated and saved as annulus.msh.")

# Convert mesh to XDMF.
msh_data = meshio.read("annulus.msh")
triangle_cells = msh_data.get_cells_type("triangle")
meshio_mesh = meshio.Mesh(points=msh_data.points, cells=[("triangle", triangle_cells)])
meshio.write("annulus.xdmf", meshio_mesh)
print("Mesh converted to XDMF format as annulus.xdmf.")

# =============================================================================
# 2. Define Problem Parameters
# =============================================================================
tau_val = 1.0      # parameter τ
f_val   = 1.0      # prescribed u on inner boundary
h_val   = 0.5      # prescribed ∂ₙu on inner boundary
g_val   = 1.0      # source term

# =============================================================================
# 3. Create Function Spaces and Mixed Space
# =============================================================================
# Create a standard H¹–conforming space using Lagrange elements of degree 2.
cell = mesh.ufl_cell()  # get the UFL cell type from the mesh
element_CG = FiniteElement("Lagrange", cell, 2,
                           reference_value_shape=(),
                           pullback="identity",
                           sobolev_space="H1")
V1 = fem.FunctionSpace(mesh, element_CG)
# Use the same space for the auxiliary variable.
W1 = fem.FunctionSpace(mesh, element_CG)
# Create a mixed element: W = V1 x W1.
mixed_elem = ufl.MixedElement([V1.ufl_element(), W1.ufl_element()])
W = fem.FunctionSpace(mesh, mixed_elem)

# =============================================================================
# 4. Define the Mixed Variational Problem
# =============================================================================
# We write u = u0 + u_h, where u0 is the lifting that enforces u = f on the inner boundary.
# Then we solve for (u_h, w) such that:
#    (Δ+τ)(u0 + u_h) = w   in Ω,
#    (Δ+τ) w = g          in Ω.
# Let (v, q) be test functions.
(u_h, w) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

# Define the bilinear form.
a1 = inner(grad(u_h), grad(v))*dx + tau_val*u_h*v*dx - w*v*dx
a2 = inner(grad(w), grad(q))*dx + tau_val*w*q*dx
a_mixed = a1 + a2

# -----------------------------------------------------------------------------
# 5. Construct the Lifting Function u0
# -----------------------------------------------------------------------------
# We choose a radial lifting:
#   u0(x) = f + h*(sqrt(x[0]^2+x[1]^2) - R_inner)
u0_expr = ufl.Expression("f + h*(sqrt(x[0]*x[0] + x[1]*x[1]) - R_inner)",
                           f=f_val, h=h_val, R_inner=R_inner, degree=2)
u0 = fem.Function(V1)
u0.interpolate(u0_expr)

# Define the right-hand side:
L1 = - inner(grad(u0), grad(v))*dx - tau_val*u0*v*dx
L2 = g_val*q*dx
L_mixed = L1 + L2

# -----------------------------------------------------------------------------
# 6. Impose Dirichlet BC on the Inner Boundary for u_h
# -----------------------------------------------------------------------------
# We enforce that on the inner boundary, u = u0; hence, we set u_h = 0 there.
def inner_boundary(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    return np.isclose(r, R_inner, atol=1e-3)

inner_facets = mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, inner_boundary)
dofs_inner = fem.locate_dofs_topological(V1, mesh.topology.dim - 1, inner_facets)
bc_u_h = fem.dirichletbc(np.array(0, dtype=np.float64), dofs_inner)
bcs_mixed = [bc_u_h]

# =============================================================================
# 7. Solve the Mixed Problem
# =============================================================================
problem = fem.petsc.LinearProblem(a_mixed, L_mixed, bcs=bcs_mixed,
                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
sol = problem.solve()
(u_h_sol, w_sol) = sol.split()

# The full solution is u = u0 + u_h.
u_full = fem.Function(V1)
u_full.vector.set_local(u0.vector.get_local() + u_h_sol.vector.get_local())

# =============================================================================
# 8. Post-Processing: Plot and Save the Solution
# =============================================================================
dof_coords = V1.tabulate_dof_coordinates().reshape((-1, mesh.geometry.dim))
u_values = u_full.vector.get_local()

plt.figure(figsize=(8,6))
contour = plt.tricontourf(dof_coords[:, 0], dof_coords[:, 1], u_values, 50, cmap="viridis")
cbar = plt.colorbar(contour)
cbar.set_label("u(x)", fontsize=12)
plt.title("Mixed-Formulation Solution of $(\\Delta+\\tau)^2 u = g$")
plt.xlabel("x")
plt.ylabel("y")
textstr = (f"Inner boundary (r = {R_inner:.2f}):\n  u = f (non-homogeneous), ∂ₙu = h\n"
           "Outer boundary: natural (Neumann) BC")
plt.gcf().text(0.65, 0.75, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()

with io.XDMFFile(comm, "solution_mixed.xdmf", "w") as xdmf_file:
    xdmf_file.write_mesh(mesh)
    xdmf_file.write_function(u_full)
print("Solution saved to solution_mixed.xdmf")
