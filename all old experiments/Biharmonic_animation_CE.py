import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from dolfin import *
from tqdm import tqdm  # Import progress bar

# Suppress excessive output from FEniCS
set_log_level(LogLevel.ERROR)

# Define the mesh (plate geometry)
N = 32  # Number of elements in each direction
mesh = UnitSquareMesh(N, N)  # Create a unit square mesh

# Define the function space (H^2 space for biharmonic equation)
V = FunctionSpace(mesh, "Lagrange", 2)  # Quadratic elements

# Define boundary conditions (clamped edges: w = 0 and ∂w/∂n = 0)
def clamped_boundary(x, on_boundary):
    return on_boundary

bc = [DirichletBC(V, Constant(0), clamped_boundary)]

# Define test and trial functions
w = TrialFunction(V)  # Unknown function
v = TestFunction(V)  # Test function
D = Constant(1.0)  # Flexural rigidity

# Animation parameters
frames = 50  # Number of animation frames
deflections = []  # Store deflections for each time step

# Generate deflection solutions over time with progress bar
for t in tqdm(np.linspace(0, 1, frames), desc="Solving PDE"):
    q0 = Constant(t)  # Load increases from 0 to 1
    a = inner(div(grad(w)), div(grad(v))) * dx
    L = (q0 / D) * v * dx

    # Solve the PDE
    w_sol = Function(V)
    solve(a == L, w_sol, bc)

    # Extract solution values
    coords = mesh.coordinates()
    values = np.array([w_sol(Point(x, y)) for x, y in coords])
    
    # Reshape into a grid for 3D plotting
    X = coords[:, 0].reshape((N+1, N+1))
    Y = coords[:, 1].reshape((N+1, N+1))
    Z = values.reshape((N+1, N+1))

    deflections.append(Z)

# Set up figure for animation
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Function to update plot in animation
def update(frame):
    ax.clear()
    ax.set_zlim(-0.1, 0.1)  # Set consistent z-axis limits
    ax.plot_surface(X, Y, deflections[frame], cmap="viridis", edgecolor='none')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Deflection w(x,y)")
    ax.set_title(f"Plate Deformation - Frame {frame+1}/{frames}")

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)

# Show animation
plt.show()
