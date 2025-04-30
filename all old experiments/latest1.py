#!/usr/bin/env python
from dolfin import *
import gmsh
import meshio
import math
import os
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 1. Mesh Generation with Gmsh
# ===============================
# Mesh parameters
lc      = 0.05      # characteristic mesh size
R_inner = 1.0       # inner radius
R_outer = 2.0       # outer radius

gmsh.initialize()
gmsh.model.add("annulus")

# Define center point (for circle arcs)
p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)

# Define points on the inner circle
p1 = gmsh.model.geo.addPoint(R_inner, 0.0, 0.0, lc)
p2 = gmsh.model.geo.addPoint(0.0, R_inner, 0.0, lc)
p3 = gmsh.model.geo.addPoint(-R_inner, 0.0, 0.0, lc)
p4 = gmsh.model.geo.addPoint(0.0, -R_inner, 0.0, lc)

# Define points on the outer circle
q1 = gmsh.model.geo.addPoint(R_outer, 0.0, 0.0, lc)
q2 = gmsh.model.geo.addPoint(0.0, R_outer, 0.0, lc)
q3 = gmsh.model.geo.addPoint(-R_outer, 0.0, 0.0, lc)
q4 = gmsh.model.geo.addPoint(0.0, -R_outer, 0.0, lc)

# Create inner circle arcs (using p0 as center)
arc1 = gmsh.model.geo.addCircleArc(p1, p0, p2)
arc2 = gmsh.model.geo.addCircleArc(p2, p0, p3)
arc3 = gmsh.model.geo.addCircleArc(p3, p0, p4)
arc4 = gmsh.model.geo.addCircleArc(p4, p0, p1)

# Create outer circle arcs (using p0 as center)
arc5 = gmsh.model.geo.addCircleArc(q1, p0, q2)
arc6 = gmsh.model.geo.addCircleArc(q2, p0, q3)
arc7 = gmsh.model.geo.addCircleArc(q3, p0, q4)
arc8 = gmsh.model.geo.addCircleArc(q4, p0, q1)

# Create curve loops for inner and outer boundaries
inner_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
outer_loop = gmsh.model.geo.addCurveLoop([arc5, arc6, arc7, arc8])

# Create a plane surface with a hole (inner_loop is the hole)
surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("annulus.msh")
gmsh.finalize()

# ===============================
# 2. Convert the Mesh to XDMF Using Meshio
# ===============================
msh = meshio.read("annulus.msh")
triangle_cells = msh.get_cells_type("triangle")
meshio_mesh = meshio.Mesh(points=msh.points, cells=[("triangle", triangle_cells)])
meshio.write("annulus.xdmf", meshio_mesh)

# ===============================
# 3. Load the Mesh in FEniCS
# ===============================
mesh = Mesh()
with XDMFFile("annulus.xdmf") as infile:
    infile.read(mesh)

# ===============================
# 4. Define the Inner Boundary for BCs
# ===============================
class InnerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        r = math.sqrt(x[0]**2 + x[1]**2)
        # Select points on the inner circle (with tolerance)
        return on_boundary and near(r, R_inner, 1e-3)

inner_boundary = InnerBoundary()

# ===============================
# 5. Define Function Spaces and Mixed Space
# ===============================
V = FunctionSpace(mesh, "Lagrange", 2)
# Create a mixed element from two copies of V
mixed_elem = MixedElement([V.ufl_element(), V.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# ===============================
# 6. Define Problem Parameters and Non-Homogeneous Data
# ===============================
tau    = Constant(7.0)
g_expr = Constant(1.0)
f_val  = 18.0    # u = f on inner boundary
h_val  = 0.5    # ∂n u = h on inner boundary

# ===============================
# 7. Construct a Lifting Function u₀
# ===============================
# We choose a radial lifting function:
#   u₀(x) = f + h*(r - R_inner)
# so that at r = R_inner, u₀ = f and its radial derivative approximates h.
u0_expr = Expression("f + h*(sqrt(x[0]*x[0] + x[1]*x[1]) - R_inner)",
                     f=f_val, h=h_val, R_inner=R_inner, degree=2)
u0 = interpolate(u0_expr, V)

# ===============================
# 8. Formulate the Mixed Variational Problem
# ===============================
# Write u = u₀ + u_h and introduce an auxiliary variable w such that:
#    (Δ + τ)(u₀ + u_h) = w.
(u_h, w) = TrialFunctions(W)
(v, q)   = TestFunctions(W)

# Equation (i): weak form for (Δ + τ)(u₀ + u_h) = w
a1 = inner(grad(u_h), grad(v))*dx + tau*u_h*v*dx - w*v*dx
# Equation (ii): weak form for (Δ + τ)w = g
a2 = inner(grad(w), grad(q))*dx + tau*w*q*dx
a = a1 + a2

# Right-hand side: move u₀ terms to RHS
L1 = - inner(grad(u0), grad(v))*dx - tau*u0*v*dx
L2 = g_expr*q*dx
L = L1 + L2

# ===============================
# 9. Impose Homogeneous Dirichlet Condition for u_h on the Inner Boundary
# ===============================
# This ensures that u = u₀ + u_h satisfies u = u₀ = f on the inner boundary.
bc = DirichletBC(W.sub(0), Constant(0.0), inner_boundary)

# ===============================
# 10. Solve the Mixed Problem and Recover Full Solution
# ===============================
W_sol = Function(W)
solve(a == L, W_sol, bc)
(u_h_sol, w_sol) = W_sol.split()

# Make sure u_h_sol is in the same space as u0 by projecting into V
u_h_sol_V = interpolate(u_h_sol, V)
u_sol = Function(V)
u_sol.assign(u0 + u_h_sol_V)

# ===============================
# 11. Plot the Solution using Matplotlib (Custom Plotting)
# ===============================
# Extract degrees of freedom coordinates. Use mesh.geometry().dim() to determine dimension.
dof_coords = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
u_values = u_sol.vector().get_local()

plt.figure()
plt.tricontourf(dof_coords[:, 0], dof_coords[:, 1], u_values, 50, cmap="viridis")
plt.colorbar()
plt.title("Solution u with non-homogeneous inner BC (u = f, u' ≈ h)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
