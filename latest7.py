#!/usr/bin/env python
"""
mixed_annulus_2d.py

Legacy FEniCS code that:
  1. Generates an annular mesh using gmsh (converted to XDMF via meshio),
  2. Uses a mixed formulation to solve a fourth-order PDE,
  3. Loops over 20 different choices for the functions f, h, and g,
  4. For each test case, enforces u = u0 (with u0 = f + h*(sqrt(x[0]^2+x[1]^2)-R_inner))
     on the inner boundary and solves for u_h (with u_h = 0 on the inner boundary),
  5. Recovers the full solution u = u0 + u_h,
  6. Generates and saves a 2D contour plot for each test case.
  
Usage (in a legacy FEniCS environment):
    python mixed_annulus_2d.py
"""

from fenics import *
import meshio
import gmsh
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# ----------------------------
# 1. Generate Annular Mesh using gmsh and convert to XDMF
# ----------------------------
lc = 0.05      # mesh size
R_inner = 3.0  # inner radius
R_outer = 8.0  # outer radius

gmsh.initialize()
gmsh.model.add("annulus")

p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
p1 = gmsh.model.geo.addPoint(R_inner, 0.0, 0.0, lc)
p2 = gmsh.model.geo.addPoint(0.0, R_inner, 0.0, lc)
p3 = gmsh.model.geo.addPoint(-R_inner, 0.0, 0.0, lc)
p4 = gmsh.model.geo.addPoint(0.0, -R_inner, 0.0, lc)
q1 = gmsh.model.geo.addPoint(R_outer, 0.0, 0.0, lc)
q2 = gmsh.model.geo.addPoint(0.0, R_outer, 0.0, lc)
q3 = gmsh.model.geo.addPoint(-R_outer, 0.0, 0.0, lc)
q4 = gmsh.model.geo.addPoint(0.0, -R_outer, 0.0, lc)

arc1 = gmsh.model.geo.addCircleArc(p1, p0, p2)
arc2 = gmsh.model.geo.addCircleArc(p2, p0, p3)
arc3 = gmsh.model.geo.addCircleArc(p3, p0, p4)
arc4 = gmsh.model.geo.addCircleArc(p4, p0, p1)
arc5 = gmsh.model.geo.addCircleArc(q1, p0, q2)
arc6 = gmsh.model.geo.addCircleArc(q2, p0, q3)
arc7 = gmsh.model.geo.addCircleArc(q3, p0, q4)
arc8 = gmsh.model.geo.addCircleArc(q4, p0, q1)

inner_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
outer_loop = gmsh.model.geo.addCurveLoop([arc5, arc6, arc7, arc8])
surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("annulus.msh")
gmsh.finalize()
print("Mesh generated and saved as annulus.msh.")

msh_data = meshio.read("annulus.msh")
triangle_cells = msh_data.get_cells_type("triangle")
meshio_mesh = meshio.Mesh(points=msh_data.points, cells=[("triangle", triangle_cells)])
meshio.write("annulus.xdmf", meshio_mesh)
print("Mesh converted to XDMF format as annulus.xdmf.")

# ----------------------------
# 2. Import Mesh in FEniCS
# ----------------------------
mesh = Mesh()
with XDMFFile("annulus.xdmf") as infile:
    infile.read(mesh)
print("Mesh imported into FEniCS.")

# ----------------------------
# 3. Define Problem Parameters and Test Cases
# ----------------------------
tau_val = 1.0  # parameter τ

# Define 20 test cases (each with f, h, g as strings).
# test_cases = [
#     {"f": "1.0", "h": "0.5", "g": "1.0"},
#     {"f": "1.0 + x[0]", "h": "0.5 + 0.2*x[1]", "g": "1.0"},
#     {"f": "sin(x[0])", "h": "cos(x[1])", "g": "1.0"},
#     {"f": "exp(-x[0]*x[0])", "h": "x[1]", "g": "1.0"},
#     {"f": "1.0 + sin(x[1])", "h": "0.5 + cos(x[0])", "g": "1.0"},
#     {"f": "2.0", "h": "0.0", "g": "1.0"},
#     {"f": "1.0 + 0.5*x[0]*x[1]", "h": "0.5 - 0.1*x[0]", "g": "1.0 + 0.1*x[1]"},
#     {"f": "1.0 + x[0]*x[0]", "h": "0.5 + x[1]*x[1]", "g": "1.0"},
#     {"f": "cos(x[0])", "h": "sin(x[1])", "g": "1.0"},
#     {"f": "1.0", "h": "0.5*sin(x[0])", "g": "1.0 + cos(x[1])"},
#     {"f": "1.0 + 0.2*x[0]", "h": "0.5 + 0.2*x[0]", "g": "1.0 + 0.2*x[1]"},
#     {"f": "exp(-x[0])", "h": "exp(-x[1])", "g": "1.0"},
#     {"f": "1.0 + x[1]", "h": "0.5 + x[0]", "g": "1.0"},
#     {"f": "1.0 + sin(pi*x[0])", "h": "0.5 + cos(pi*x[1])", "g": "1.0"},
#     {"f": "1.0 + 0.5*x[0]*sin(x[1])", "h": "0.5 + 0.5*x[1]*cos(x[0])", "g": "1.0"},
#     {"f": "2.0 - x[0]", "h": "0.5 + x[1]", "g": "1.0"},
#     {"f": "1.0", "h": "0.5", "g": "1.0 + sin(x[0])"},
#     {"f": "1.0 + cos(x[1])", "h": "0.5 + sin(x[0])", "g": "1.0 + cos(x[0])"},
#     {"f": "1.0 + 0.3*x[0]", "h": "0.5 + 0.3*x[1]", "g": "1.0 + 0.3*x[0]*x[1]"},
#     {"f": "1.0 + sin(pi*x[0]) + cos(pi*x[1])", "h": "0.5 + cos(pi*x[0]) - sin(pi*x[1])", "g": "1.0"}
# ]
test_cases = [
    {"f": "1.0", "h": "0.5", "g": "0.0"},
    {"f": "1.0 + x[0]", "h": "0.5 + 0.2*x[1]", "g": "0.0"},
    {"f": "sin(x[0])", "h": "cos(x[1])", "g": "0.0"},
    {"f": "exp(-x[0]*x[0])", "h": "x[1]", "g": "0.0"},
    {"f": "1.0 + sin(x[1])", "h": "0.5 + cos(x[0])", "g": "0.0"},
    {"f": "2.0", "h": "0.0", "g": "0.0"},
    {"f": "1.0 + 0.5*x[0]*x[1]", "h": "0.5 - 0.1*x[0]", "g": "0"},
    {"f": "1.0 + x[0]*x[0]", "h": "0.5 + x[1]*x[1]", "g": "0"},
    {"f": "cos(x[0])", "h": "sin(x[1])", "g": "0"},
    {"f": "1.0", "h": "0.5*sin(x[0])", "g": "0"},
    {"f": "1.0 + 0.2*x[0]", "h": "0.5 + 0.2*x[0]", "g": "0"},
    {"f": "exp(-x[0])", "h": "exp(-x[1])", "g": "0"},
    {"f": "1.0 + x[1]", "h": "0.5 + x[0]", "g": "0"},
    {"f": "1.0 + sin(pi*x[0])", "h": "0.5 + cos(pi*x[1])", "g": "0"},
    {"f": "1.0 + 0.5*x[0]*sin(x[1])", "h": "0.5 + 0.5*x[1]*cos(x[0])", "g": "0"},
    {"f": "2.0 - x[0]", "h": "0.5 + x[1]", "g": "0"},
    {"f": "1.0", "h": "0.5", "g": "0"},
    {"f": "1.0 + cos(x[1])", "h": "0.5 + sin(x[0])", "g": "0"},
    {"f": "1.0 + 0.3*x[0]", "h": "0.5 + 0.3*x[1]", "g": "0"},
    {"f": "1.0 + sin(pi*x[0]) + cos(pi*x[1])", "h": "0.5 + cos(pi*x[0]) - sin(pi*x[1])", "g": "0.0"}
]

# ----------------------------
# 4. Create Function Spaces and Mixed Space (same as above)
# ----------------------------
cell = mesh.ufl_cell()
element_CG = FiniteElement("Lagrange", cell, 2)
V1 = FunctionSpace(mesh, element_CG)
W1 = FunctionSpace(mesh, element_CG)
mixed_elem = MixedElement([V1.ufl_element(), V1.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# ----------------------------
# 5. Loop over test cases, solve, and generate 2D contour plots
# ----------------------------
for i, test in enumerate(test_cases):
    # Update the parameters for this test case.
    f_expr_str = test["f"]
    h_expr_str = test["h"]
    g_expr_str = test["g"]
    
    # Create the lifting function u0.
    u0_expr = Expression(f_expr_str + " + " + h_expr_str +
                           "*(sqrt(x[0]*x[0]+x[1]*x[1])-R_inner)",
                           degree=2, f=1.0, h=0.0, R_inner=R_inner)  # dummy parameters; override below
    # To properly incorporate the expression strings, we pass the parameters as numbers:
    u0_expr.f = float(test["f"]) if test["f"].replace(" ", "").isnumeric() else 1.0
    u0_expr.h = float(test["h"]) if test["h"].replace(" ", "").isnumeric() else 0.0
    u0_expr.R_inner = R_inner
    # For more general expressions, we use the string directly:
    u0_expr = Expression("("+f_expr_str+") + ("+h_expr_str+")*(sqrt(x[0]*x[0]+x[1]*x[1])-R_inner)",
                         degree=2, R_inner=R_inner)
    u0 = interpolate(u0_expr, V1)
    
    # Define the mixed problem.
    (u_h, w) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    a1 = inner(grad(u_h), grad(v))*dx + tau_val*u_h*v*dx - w*v*dx
    a2 = inner(grad(w), grad(q))*dx + tau_val*w*q*dx
    a_mixed = a1 + a2
    L1 = - inner(grad(u0), grad(v))*dx - tau_val*u0*v*dx
    L2 = Expression(g_expr_str, degree=2) * q * dx  # g as an expression
    L = L1 + L2
    
    # Impose Dirichlet BC for u_h (u_h = 0 on inner boundary)
    class InnerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            r = sqrt(x[0]**2 + x[1]**2)
            return on_boundary and near(r, R_inner, 1e-3)
    inner_b = InnerBoundary()
    bc = DirichletBC(W.sub(0), Constant(0.0), inner_b)
    bcs_mixed = [bc]
    
    solution = Function(W)
    solve(a_mixed == L, solution, bcs_mixed)
    (u_h_sol, w_sol) = solution.split(deepcopy=True)
    u_full = Function(V1)
    u_full.vector()[:] = u0.vector()[:] + u_h_sol.vector()[:]
    
    # 2D contour plot using vertex values
    vertex_values = u_full.compute_vertex_values(mesh)
    coordinates = mesh.coordinates()
    plt.figure()
    cont = plt.tricontourf(coordinates[:, 0], coordinates[:, 1],
                           vertex_values, 50, cmap="viridis")
    plt.colorbar(cont, label="u(x)")
    plt.title("Test case %d: u = u0 + u_h" % (i+1))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.figtext(0.65, 0.75, "f: %s\nh: %s\ng: %s" % (f_expr_str, h_expr_str, g_expr_str),
                bbox=dict(facecolor="white", alpha=0.8), fontsize=10)
    plt.tight_layout()
    plt.savefig("solution_case_%02d_2d.png" % (i+1))
    plt.close()
    print("Test case %d: 2D plot saved as solution_case_%02d_2d.png" % (i+1, i+1))

print("All 2D test case plots generated and saved.")
