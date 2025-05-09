# from dolfinx import *
# from ufl import FiniteElement
# from ufl.core.element import MixedElement
# from slepc4py import SLEPc
# import numpy as np

# # --- 1. Load mesh + boundary markers ---

# # Build two Lagrange P1 elements on triangles
# elem = FiniteElement("CG", mesh.ufl_cell(), 1)
# mixed_elem = MixedElement([elem, elem])

# # Create the mixed FunctionSpace
# from dolfinx.fem import FunctionSpace
# W = FunctionSpace(mesh, mixed_elem)

# (u, v)    = TrialFunctions(W)
# (phi, psi)= TestFunctions(W)


# # --- 2. Function spaces ---


# # 2) Mix them by multiplication
#         # now W is H¹×H¹       
# (u, v)    = TrialFunctions(W)
# (phi, psi)= TestFunctions(W)

# # --- 3. Refractive index n(x) ---
# # assume n= n1 inside ellipse, n=1 outside
# n1 = 4.0
# n = Expression(" (x[0]/a)*(x[0]/a)+(x[1]/b)*(x[1]/b) < 1 ? n1 : 1.0",
#                degree=0, a=1.0, b=0.7, n1=n1)

# # --- 4. Bilinear forms a and b ---
# a_form = ( dot(grad(u),grad(phi)) - dot(grad(v),grad(psi)) )*dx
# b_form = ( n*u*phi - v*psi )*dx

# # --- 5. Dirichlet BC: u - v = 0 on inner ellipse ---
# # We impose u=v by setting u - v to zero via a boundary condition
# zero = Constant(0.0)
# # BC on u - v requires “coupling”:
# # we specify u = v by two conditions: u = w, v = w, with w free—but easiest is Lagrange multiplier.
# # Here we cheat by enforcing u = v = 0 on inner, then shift later; 
# # for true u=v one uses multipliers.  
# bc_u = DirichletBC(W.sub(0), zero, boundaries, inner_id)
# bc_v = DirichletBC(W.sub(1), zero, boundaries, inner_id)
# bcs = [bc_u, bc_v]

# # --- 6. Assemble PETSc matrices ---
# A = PETScMatrix(); assemble(a_form, tensor=A)
# B = PETScMatrix(); assemble(b_form, tensor=B)
# for bc in bcs:
#     bc.apply(A)
#     bc.apply(B)

# # --- 7. Solve generalized eigenproblem A x = λ B x ---
# eps = SLEPc.EPS().create(MPI.comm_world)
# eps.setOperators(A.mat(), B.mat())
# eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
# eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
# eps.solve()

# # --- 8. Extract and print k = sqrt(λ) ---
# nconv = eps.getConverged()
# ks = []
# for i in range(nconv):
#     lam, _ = eps.getEigenpair(i)
#     if lam.real > DOLFIN_EPS:
#         ks.append(np.sqrt(lam.real))
# print("Computed transmission eigenvalues k:", ks)


#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
import meshio

from dolfinx.mesh import create_mesh, CellType, meshtags
from dolfinx.fem   import (
    FunctionSpace, dirichletbc, locate_dofs_topological,
    assemble_matrix, form, Constant
)
from petsc4py       import PETSc
from slepc4py       import SLEPc

import ufl
from ufl import dot, grad, dx, TrialFunctions, TestFunctions
from ufl.finiteelement import FiniteElement, MixedElement

# -----------------------------------------------------------------------------
# 1) Read the Gmsh .msh via meshio and split out triangles & lines
# -----------------------------------------------------------------------------
m = meshio.read("Mesh/mesh_ellipse_circle.msh")

# Extract 2D triangles and 1D line facets
tri_cells = next(block.data for block in m.cells       if block.type == "triangle")
line_cells= next(block.data for block in m.cells       if block.type == "line")

# Extract the corresponding 'physical' tags
tri_markers = next(data for key,data in zip(m.cell_data["gmsh:physical"], m.cell_data["gmsh:physical"])
                   if key[0] == "triangle")
line_markers= next(data for key,data in zip(m.cell_data["gmsh:physical"], m.cell_data["gmsh:physical"])
                   if key[0] == "line")

# Build the DOLFINx mesh (points → Nx2 array, triangles → connectivity)
points = m.points[:, :2]
mesh = create_mesh(MPI.COMM_WORLD,
                   tri_cells.astype(np.int32),
                   points.astype(np.float64),
                   CellType.triangle)

# Build facet tags on the mesh
facet_tags = meshtags(mesh,
                      mesh.topology.dim - 1,
                      line_cells.astype(np.int32),
                      line_markers.astype(np.int32))

# Inspect available boundary‐tags and choose the inner‐ellipse tag
tags = np.unique(facet_tags.values)
print("Boundary tags found:", tags)
inner_tag = int(tags[1])  # e.g. usually the second tag is your ellipse group

# -----------------------------------------------------------------------------
# 2) Define the mixed H1×H1 space via UFL’s new API
# -----------------------------------------------------------------------------
# Two scalar P1 Lagrange elements
P1      = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
mixed_el= MixedElement([P1, P1])

# Create the mixed FunctionSpace
W = FunctionSpace(mesh, mixed_el)

# Split into (u,v) and (φ,ψ)
(u, v)       = TrialFunctions(W)
(phi, psi)   = TestFunctions(W)

# -----------------------------------------------------------------------------
# 3) Define the refractive‐index coefficient n(x)
# -----------------------------------------------------------------------------
# inside ellipse → n1, outside → 1.0
n1   = 4.0
a,b  = 1.0, 0.7
x    = ufl.SpatialCoordinate(mesh)
n    = ufl.conditional(x[0]**2/a**2 + x[1]**2/b**2 < 1, n1, 1.0)

# -----------------------------------------------------------------------------
# 4) Assemble the bilinear forms a((u,v),(φ,ψ)) and b((u,v),(φ,ψ))
# -----------------------------------------------------------------------------
a_form = (dot(grad(u), grad(phi)) - dot(grad(v), grad(psi))) * dx
b_form = (    n*u*phi      -      v*psi      ) * dx

# -----------------------------------------------------------------------------
# 5) Enforce u = v on the inner ellipse via two Dirichlet BCs (u=0, v=0)
# -----------------------------------------------------------------------------
fdim      = mesh.topology.dim - 1
inner_facets = facet_tags.indices[facet_tags.values == inner_tag]

dofs_u = locate_dofs_topological(W.sub(0), fdim, inner_facets)
dofs_v = locate_dofs_topological(W.sub(1), fdim, inner_facets)

zero = Constant(mesh, 0.0)
bc_u = dirichletbc(zero, dofs_u, W.sub(0))
bc_v = dirichletbc(zero, dofs_v, W.sub(1))
bcs  = [bc_u, bc_v]

# -----------------------------------------------------------------------------
# 6) Assemble PETSc matrices A and B
# -----------------------------------------------------------------------------
A = assemble_matrix(form(a_form), bcs=bcs)
A.assemble()
B = assemble_matrix(form(b_form), bcs=bcs)
B.assemble()

# -----------------------------------------------------------------------------
# 7) Solve A x = λ B x with SLEPc
# -----------------------------------------------------------------------------
eigensolver = SLEPc.EPS().create(comm=MPI.COMM_WORLD)
eigensolver.setOperators(A, B)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
eigensolver.solve()

# -----------------------------------------------------------------------------
# 8) Extract and print the transmission wavenumbers k = sqrt(λ)
# -----------------------------------------------------------------------------
nconv = eigensolver.getConverged()
ks = []
for i in range(nconv):
    lam, _ = eigensolver.getEigenpair(i)
    if lam.real > 1e-8:
        ks.append(np.sqrt(lam.real))
print("Computed transmission eigenvalues k:", ks)
