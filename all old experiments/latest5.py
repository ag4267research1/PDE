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
R_inner = 1.0  # inner radius
R_outer = 2.0  # outer radius

gmsh.initialize()
gmsh.model.add("annulus")

# Define center point
p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)

# Define points on inner circle
p1 = gmsh.model.geo.addPoint(R_inner, 0.0, 0.0, lc)
p2 = gmsh.model.geo.addPoint(0.0, R_inner, 0.0, lc)
p3 = gmsh.model.geo.addPoint(-R_inner, 0.0, 0.0, lc)
p4 = gmsh.model.geo.addPoint(0.0, -R_inner, 0.0, lc)

# Define points on outer circle
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

# Create a plane surface with a hole (inner_loop defines the hole)
surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("annulus.msh")
gmsh.finalize()
print("Mesh generated and saved as annulus.msh.")

# Convert gmsh mesh to XDMF using meshio
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
# 3. Define Problem Parameters
# ----------------------------
tau_val = 1.0      # parameter τ
f_val = 1.0        # prescribed u on inner boundary
h_val = 0.5        # prescribed ∂ₙu on inner boundary
g_val = 1.0        # source term

# ----------------------------
# 4. Create Function Spaces and Mixed Space
# ----------------------------
# Create a standard H¹ space using Lagrange elements (degree 2)
cell = mesh.ufl_cell()  # get the UFL cell type
element_CG = FiniteElement("Lagrange", cell, 2)
V1 = FunctionSpace(mesh, element_CG)
W1 = FunctionSpace(mesh, element_CG)
# Create a mixed space W = V1 x V1
mixed_elem = MixedElement([V1.ufl_element(), V1.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# ----------------------------
# 5. Define the Mixed Variational Problem
# ----------------------------
# Decompose u = u0 + u_h, where u0 is a lifting that satisfies u = f on the inner boundary.
# Then, we solve for (u_h, w) such that:
#   (Δ+τ)(u0 + u_h) = w   in Ω,
#   (Δ+τ)w = g           in Ω.
(u_h, w) = TrialFunctions(W)
(v, q) = TestFunctions(W)

a1 = inner(grad(u_h), grad(v))*dx + tau_val*u_h*v*dx - w*v*dx
a2 = inner(grad(w), grad(q))*dx + tau_val*w*q*dx
a_mixed = a1 + a2

# ----------------------------
# 6. Construct the Lifting Function u0
# ----------------------------
# Define a radial lifting function:
#   u0(x) = f + h*(sqrt(x[0]^2+x[1]^2)-R_inner)
u0_expr = Expression("f + h*(sqrt(x[0]*x[0] + x[1]*x[1]) - R_inner)",
                     degree=2, f=f_val, h=h_val, R_inner=R_inner)
u0 = interpolate(u0_expr, V1)

# Right-hand side:
L1 = - inner(grad(u0), grad(v))*dx - tau_val*u0*v*dx
L2 = g_val*q*dx
L_mixed = L1 + L2

# ----------------------------
# 7. Impose Dirichlet BC on the Inner Boundary for u_h
# ----------------------------
# Enforce u = u0 on the inner boundary by setting u_h = 0.
class InnerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        r = sqrt(x[0]**2 + x[1]**2)
        return on_boundary and near(r, R_inner, 1e-3)
inner_b = InnerBoundary()
bc = DirichletBC(W.sub(0), Constant(0.0), inner_b)
bcs_mixed = [bc]

# ----------------------------
# 8. Solve the Mixed Problem
# ----------------------------
solution = Function(W)
solve(a_mixed == L_mixed, solution, bcs_mixed)
(u_h_sol, w_sol) = solution.split(deepcopy=True)

# Full solution: u = u0 + u_h.
u_full = Function(V1)
u_full.vector()[:] = u0.vector()[:] + u_h_sol.vector()[:]

# ----------------------------
# 9. Post-Processing: Plot and Save the Solution
# ----------------------------
# Instead of using dolfin's plot (which may attempt a 3D projection), we extract vertex values.
vertex_values = u_full.compute_vertex_values(mesh)
coordinates = mesh.coordinates()

plt.figure(figsize=(8,6))
contour = plt.tricontourf(coordinates[:, 0], coordinates[:, 1], vertex_values, 50, cmap="viridis")
plt.colorbar(contour, label="u(x)")
plt.title("Mixed-Formulation Solution of $(\\Delta+\\tau)^2 u = g$")
plt.xlabel("x")
plt.ylabel("y")
plt.figtext(0.65, 0.75, 
            "Inner boundary (r = 1.00):\n  u = f (non-homogeneous), ∂ₙu = h\n"
            "Outer boundary: natural (Neumann) BC", 
            bbox=dict(facecolor="white", alpha=0.8))
plt.tight_layout()
plt.show()

File("solution_mixed.pvd") << u_full
print("Solution saved to solution_mixed.pvd")
