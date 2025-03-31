from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Define a refined mesh for better stability
mesh = UnitSquareMesh(128, 128)

# Define function space (Higher-order Lagrange elements for smoothness)
V = FunctionSpace(mesh, "Lagrange", 3)

# Define a mixed finite element
element = MixedElement([V.ufl_element(), V.ufl_element()])
V2 = FunctionSpace(mesh, element)

# Define trial and test functions
(u, w) = TrialFunctions(V2)
(v, z) = TestFunctions(V2)

# Define forcing function
f = Constant(1.0)  # Uniform force applied on the plate

# Define bilinear and linear forms
a = inner(grad(grad(u)), grad(grad(v))) * dx + inner(w, z) * ds
L = f * v * dx

# Define boundary conditions
def clamped_boundary(x, on_boundary):
    return on_boundary

bc_u = DirichletBC(V2.sub(0), Constant(0.0), clamped_boundary)  # Enforce u = 0
bc_w = DirichletBC(V2.sub(1), Constant(0.0), clamped_boundary)  # Enforce w = 0

# Compute the solution using a robust solver
U = Function(V2)
solve(a == L, U, [bc_u, bc_w], solver_parameters={"linear_solver": "gmres", "preconditioner": "ilu"})

# Extract u from the mixed solution
u_solution, _ = U.split()

# Check for NaNs before plotting
u_values = u_solution.vector().get_local()
if np.isnan(u_values).any() or np.isinf(u_values).any():
    print("⚠️ Warning: The solution contains NaN or Inf values!")
else:
    plt.figure(figsize=(8, 6))
    plot(u_solution)
    plt.title("Clamped Plate Solution (Biharmonic)")
    plt.colorbar(plt.cm.ScalarMappable(), label="Displacement")
    plt.show()
