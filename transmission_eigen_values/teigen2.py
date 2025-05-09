#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py   import SLEPc
import meshio

import ufl
from ufl import (
    SpatialCoordinate, dot, grad, dx, conditional,
    TrialFunctions, TestFunctions
)

from dolfinx.io   import gmshio
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem  import (
    FunctionSpace,
    dirichletbc,
    locate_dofs_topological,
    assemble_matrix,
    form,
    Constant,
)
from basix import create_element, CellType as BasixCellType, ElementFamily



# 1) Read mesh + facet tags in one go
mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "Mesh/mesh_ellipse_circle.msh", MPI.COMM_WORLD, gdim=2
)

# 2) Auto-detect inner‐ellipse facets by checking x^2/a^2 + y^2/b^2 < 1
a, b, tol = 1.0, 0.7, 1e-6
fdim = mesh.topology.dim - 1

def on_inner(x):
    return x[0]**2/a**2 + x[1]**2/b**2 < 1 + tol

inner_facets = locate_entities_boundary(mesh, fdim, on_inner)

# 3) Build mixed H¹×H¹ space via V×V
element = create_element(ElementFamily.P, BasixCellType.triangle, 1)
V       = FunctionSpace(mesh, element)

W = V * V
(u, v)    = TrialFunctions(W)
(phi, psi)= TestFunctions(W)

# 4) Refractive index n(x): n1 inside the ellipse, 1 outside
x  = SpatialCoordinate(mesh)
n1 = 4.0
n  = conditional(x[0]**2/a**2 + x[1]**2/b**2 < 1, n1, 1.0)

# 5) Weak forms
a_form = (dot(grad(u), grad(phi)) - dot(grad(v), grad(psi))) * dx
b_form = (    n*u*phi       -      v*psi      ) * dx

# 6) Enforce u=v on the inner boundary via u=0 and v=0
dofs_u = locate_dofs_topological(W.sub(0), fdim, inner_facets)
dofs_v = locate_dofs_topological(W.sub(1), fdim, inner_facets)
bc_u   = dirichletbc(Constant(mesh, 0.0), dofs_u, W.sub(0))
bc_v   = dirichletbc(Constant(mesh, 0.0), dofs_v, W.sub(1))
bcs    = [bc_u, bc_v]

# 7) Assemble PETSc matrices A and B
A = assemble_matrix(form(a_form), bcs=bcs); A.assemble()
B = assemble_matrix(form(b_form), bcs=bcs); B.assemble()

# 8) Solve A x = λ B x with SLEPc
eps = SLEPc.EPS().create(comm=MPI.COMM_WORLD)
eps.setOperators(A, B)
eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
eps.solve()

# 9) Extract and print k = sqrt(λ)
ks = []
for i in range(eps.getConverged()):
    lam, _ = eps.getEigenpair(i)
    if lam.real > 1e-8:
        ks.append(np.sqrt(lam.real))
print("Computed transmission eigenvalues k:", ks)
