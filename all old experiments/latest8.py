#!/usr/bin/env python
"""
mixed_annulus_3d.py

Legacy FEniCS code that:
  1. Generates an annular mesh using gmsh (converted to XDMF via meshio),
  2. Uses a mixed formulation to solve a fourth-order PDE on the annulus,
  3. Loops over 20 different test cases for functions f, h, and g,
  4. For each test case, solves for the full solution u = u0 + u_h,
  5. Produces a 3D surface plot of the solution using matplotlib's mplot3d,
  6. Saves each 3D plot to a separate file.

Usage (in legacy FEniCS):
    python mixed_annulus_3d.py
"""

from fenics import *
import meshio
import gmsh
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
import os

# ----------------------------
# 1. Generate Annular Mesh using gmsh and convert to XDMF
# ----------------------------
lc = 0.05      # mesh size
R_inner = 5.0  # inner radius
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
# 4. Create Function Spaces and Mixed Space
# ----------------------------
cell = mesh.ufl_cell()
element_CG = FiniteElement("Lagrange", cell, 2)
V1 = FunctionSpace(mesh, element_CG)
W1 = FunctionSpace(mesh, element_CG)
mixed_elem = MixedElement([V1.ufl_element(), V1.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# ----------------------------
# 5. Loop over test cases, solve, and generate 3D surface plots
# ----------------------------
for i, test in enumerate(test_cases):
    # Define the lifting function u0 for this test case.
    u0_expr = Expression("("+test['f']+") + ("+test['h']+")*(sqrt(x[0]*x[0]+x[1]*x[1])-R_inner)",
                           degree=2, R_inner=R_inner)
    u0 = interpolate(u0_expr, V1)
    
    # Define mixed problem: solve for (u_h, w)
    (u_h, w) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    a1 = inner(grad(u_h), grad(v))*dx + tau_val*u_h*v*dx - w*v*dx
    a2 = inner(grad(w), grad(q))*dx + tau_val*w*q*dx
    a_mixed = a1 + a2
    
    L1 = - inner(grad(u0), grad(v))*dx - tau_val*u0*v*dx
    # Define g as an Expression for this test case.
    g_expr = Expression(test['g'], degree=2)
    L2 = g_expr*q*dx
    L_mixed = L1 + L2
    
    # Impose u_h = 0 on inner boundary.
    class InnerBoundary(SubDomain):
        def inside(self, x, on_boundary):
            r = sqrt(x[0]**2 + x[1]**2)
            return on_boundary and near(r, R_inner, 1e-3)
    inner_b = InnerBoundary()
    bc = DirichletBC(V1, Constant(0.0), inner_b)  # On mixed space, use W.sub(0)
    bc_mixed = DirichletBC(W.sub(0), Constant(0.0), inner_b)
    bcs = [bc_mixed]
    
    solution = Function(W)
    solve(a_mixed == L_mixed, solution, bcs)
    (u_h_sol, w_sol) = solution.split(deepcopy=True)
    u_full = Function(V1)
    u_full.vector()[:] = u0.vector()[:] + u_h_sol.vector()[:]
    
    # 3D plot:
    # Get vertex coordinates and solution values.
    coords = mesh.coordinates()
    u_vertex = u_full.compute_vertex_values(mesh)
    # Get cell connectivity.
    # In legacy FEniCS, mesh.cells() returns connectivity.
    triangles = mesh.cells()
    triangles = np.array(triangles, dtype=np.int32)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")
    triang = Triangulation(coords[:, 0], coords[:, 1], triangles)
    surf = ax.plot_trisurf(triang, u_vertex, cmap="viridis", edgecolor="none")
    ax.set_title("Test case %d: 3D Surface Plot of u" % (i+1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x)")
    ax.text2D(0.65, 0.75, "f: %s\nh: %s\ng: %s" % (test['f'], test['h'], test['g']),
              transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.8), fontsize=10)
    plt.tight_layout()
    plt.savefig("solution_case_%02d_3d.png" % (i+1))
    plt.close()
    print("Test case %d: 3D plot saved as solution_case_%02d_3d.png" % (i+1, i+1))

print("All 3D test case plots generated and saved.")
