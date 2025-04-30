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
# 6. Define a Solver Function for the Mixed Problem
# ===============================
def solve_problem(V, W, inner_boundary, tau, f_param, h_param, g_expr, R_inner):
    # Define the lifting function u0 that imposes the non-homogeneous BCs on the inner boundary.
    # Here we assume the lifting formula: u0(x) = f(x) + h(x) * (r - R_inner)
    u0_expr = Expression("f + h*(sqrt(x[0]*x[0] + x[1]*x[1]) - R_inner)",
                         f=f_param, h=h_param, R_inner=R_inner, degree=2)
    u0 = interpolate(u0_expr, V)
    
    # Split u into u = u0 + u_h, and introduce an auxiliary variable w such that:
    # (Δ + τ)(u0 + u_h) = w.
    (u_h, w) = TrialFunctions(W)
    (v, q)   = TestFunctions(W)
    
    # Equation (i): weak form for (Δ + τ)(u0 + u_h) = w.
    a1 = inner(grad(u_h), grad(v))*dx + tau*u_h*v*dx - w*v*dx
    # Equation (ii): weak form for (Δ + τ)w = g.
    a2 = inner(grad(w), grad(q))*dx + tau*w*q*dx
    a = a1 + a2
    
    # Move the u0 terms to the RHS.
    L1 = - inner(grad(u0), grad(v))*dx - tau*u0*v*dx
    L2 = g_expr*q*dx
    L = L1 + L2
    
    # Impose u_h = 0 on the inner boundary.
    bc = DirichletBC(W.sub(0), Constant(0.0), inner_boundary)
    
    # Solve the mixed problem.
    W_sol = Function(W)
    solve(a == L, W_sol, bc)
    (u_h_sol, w_sol) = W_sol.split()
    
    # Project u_h_sol into V (to ensure both are in the same space) and recover u.
    u_h_sol_V = interpolate(u_h_sol, V)
    u_sol = Function(V)
    u_sol.assign(u0 + u_h_sol_V)
    return u_sol

# ===============================
# 7. Define Examples with Different f, h, and g Functions
# ===============================
# Example 1: Constant data
example1 = {
    "f": Constant(1.0),
    "h": Constant(0.5),
    "g": Constant(1.0),
    "label": "Example 1: f=1, h=0.5, g=1"
}
# Example 2: Linear and sinusoidal expressions
example2 = {
    "f": Expression("1 + 0.5*x[0]", degree=2),
    "h": Expression("0.3*x[1]", degree=2),
    "g": Expression("sin(pi*x[0])", degree=2),
    "label": "Example 2: f=1+0.5*x[0], h=0.3*x[1], g=sin(pi*x[0])"
}
# Example 3: Quadratic and exponential expressions
example3 = {
    "f": Expression("1 + x[0]*x[1]", degree=2),
    "h": Expression("0.2 + 0.1*x[0]", degree=2),
    "g": Expression("exp(- (x[0]*x[0] + x[1]*x[1]))", degree=2),
    "label": "Example 3: f=1+x[0]*x[1], h=0.2+0.1*x[0], g=exp(-|x|^2)"
}

examples = [example1, example2, example3]

# ===============================
# 8. Solve and Plot Each Example
# ===============================
for ex in examples:
    u_sol = solve_problem(V, W, inner_boundary, Constant(1.0), ex["f"], ex["h"], ex["g"], R_inner)
    
    # Extract dof coordinates from V.
    dof_coords = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
    u_values = u_sol.vector().get_local()
    
    plt.figure()
    # Use tricontourf for a 2D contour plot.
    plt.tricontourf(dof_coords[:, 0], dof_coords[:, 1], u_values, 50, cmap="viridis")
    plt.colorbar()
    plt.title(ex["label"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
