from dolfin import *
from mshr import Circle, generate_mesh
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

# --------------------------------------------------
# 1. Problem Parameters and Mesh Generation
# --------------------------------------------------

# Simulation parameters
tau_value = 1.0           # Parameter τ in the equation
inner_radius = 1.0        # Inner radius of the annulus
outer_radius = 2.0        # Outer radius of the annulus
mesh_resolution = 64      # Controls the fineness of the mesh

# Set τ as a FEniCS Constant
tau = Constant(tau_value)

# Define the annular domain using mshr:
outer_circle = Circle(Point(0, 0), outer_radius)
inner_circle = Circle(Point(0, 0), inner_radius)
domain = outer_circle - inner_circle

# Generate a mesh for the annular domain.
mesh = generate_mesh(domain, mesh_resolution)

# --------------------------------------------------
# 2. Mixed Finite Element Function Space
# --------------------------------------------------

# Define a quadratic Lagrange finite element (H^1–conforming)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)

# Create a mixed element for the pair (u, w)
mixed_elem = MixedElement([P1, P1])

# Build the mixed function space on the mesh
W = FunctionSpace(mesh, mixed_elem)

# --------------------------------------------------
# 3. Define Trial/Test Functions and Variational Forms
# --------------------------------------------------

# In the mixed formulation we introduce an auxiliary variable:
#    w = Δu + τ u
# The unknowns are (u, w) and the corresponding test functions are (v, q)
(u, w) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Derive the weak forms:
#
# From the definition w = Δu + τ u, after integration by parts:
#    -∫_Ω grad(u)·grad(v) dx + τ∫_Ω u v dx - ∫_Ω w v dx = 0.
#
# From the original PDE, after splitting and integration by parts:
#    -∫_Ω grad(w)·grad(q) dx + τ∫_Ω w q dx = ∫_Ω g q dx.
#
# The combined bilinear form is then:
a = (
    - inner(grad(u), grad(v)) * dx
    + tau * u * v * dx
    - w * v * dx
    - inner(grad(w), grad(q)) * dx
    + tau * w * q * dx
)

# The right–hand side (linear form) comes only from the second equation:
#   L((v,q)) = ∫_Ω g q dx

# --------------------------------------------------
# 4. Define Multiple Source Functions g
# --------------------------------------------------

# We define a list of (g, description) pairs.
# The six examples are:
#   1. Constant: g = 1.0.
#   2. Polynomial: g = 100*(r² - 1)*(4 - r²), with r² = x²+y².
#      (This g vanishes on the boundaries: r=1 and r=2.)
#   3. Gaussian: g = exp(-50*((x-1.5)² + y²)).
#   4. Sinusoidal (radial): g = sin(3π r), with r = √(x²+y²).
#   5. Cosine product: g = cos(2πx)*cos(2πy).
#   6. Sin–Cos product: g = sin(2πx)*cos(2πy).

g_list = [
    (Constant(1.0), "g = 1.0 (Constant)"),
    (
        Expression("100*(x[0]*x[0] + x[1]*x[1] - 1.0)*(4.0 - (x[0]*x[0] + x[1]*x[1]))", degree=2),
        "g = 100*(r² - 1)*(4 - r²)"
    ),
    (
        Expression("exp(-50*((x[0]-1.5)*(x[0]-1.5) + x[1]*x[1]))", degree=2),
        "g = exp(-50*((x-1.5)² + y²))"
    ),
    (
        Expression("sin(3*pi*sqrt(x[0]*x[0] + x[1]*x[1]))", degree=2, pi=np.pi),
        "g = sin(3πr)"
    ),
    (
        Expression("cos(2*pi*x[0])*cos(2*pi*x[1])", degree=2, pi=np.pi),
        "g = cos(2πx)*cos(2πy)"
    ),
    (
        Expression("sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2, pi=np.pi),
        "g = sin(2πx)*cos(2πy)"
    )
]

# --------------------------------------------------
# 5. Apply Boundary Conditions
# --------------------------------------------------

# For clamped boundary conditions of the original fourth–order problem,
# we enforce u = 0. In the mixed formulation a common choice is also to set w = 0.
bc_u = DirichletBC(W.sub(0), Constant(0.0), "on_boundary")
bc_w = DirichletBC(W.sub(1), Constant(0.0), "on_boundary")
bcs = [bc_u, bc_w]

# --------------------------------------------------
# 6. Solve the Mixed Problem for Each g
# --------------------------------------------------

u_solutions = []    # To store the computed u-component for each g
descriptions = []   # To store the corresponding description text

for (g_expr, desc) in g_list:
    # Define the linear form: L((v,q)) = ∫_Ω g q dx.
    L = g_expr * q * dx

    # Create a Function in the mixed space to hold the solution (u, w)
    solution = Function(W)
    
    # Solve the variational problem: a((u,w),(v,q)) = L((v,q)) with the boundary conditions.
    solve(a == L, solution, bcs)
    
    # Split the solution into its components (u, w)
    (u_sol, w_sol) = solution.split()
    
    # Store the u-component and the description for later plotting.
    u_solutions.append(u_sol)
    descriptions.append(desc)

# --------------------------------------------------
# 7. Plot the u Solutions with Simulation Parameters Annotated
# --------------------------------------------------

# For six solutions, we arrange the plots in a 2x3 grid.
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

for i, u_sol in enumerate(u_solutions):
    # Get vertex values and coordinates for plotting.
    u_vals = u_sol.compute_vertex_values(mesh)
    coords = mesh.coordinates()
    
    # Create a triangulation for contour plotting.
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], mesh.cells())
    
    # Create a filled contour plot of u.
    tpc = axs[i].tricontourf(triangulation, u_vals, levels=50, cmap="viridis")
    
    axs[i].set_title(descriptions[i], fontsize=12)
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("y")
    axs[i].set_aspect('equal')
    fig.colorbar(tpc, ax=axs[i])

# Create a text box with all simulation parameters and paste it at the bottom of the figure.
params_text = (
    f"Parameters:\n"
    f"Inner radius = {inner_radius}\n"
    f"Outer radius = {outer_radius}\n"
    f"Tau = {tau_value}\n"
    f"Mesh resolution = {mesh_resolution}"
)
fig.text(0.5, 0.01, params_text, wrap=True, horizontalalignment='center', fontsize=12)

# Set a super-title for the entire figure.
fig.suptitle("Solution u for Different g in the Mixed Formulation of $(\\Delta+\\tau)^2 u = g$", fontsize=16)
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.show()
