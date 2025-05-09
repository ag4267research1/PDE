from dolfin import *

# 1) Load **only** the 2D triangle mesh
mesh = Mesh("Mesh/mesh_ellipse_circle.xml")
print("CELL TYPE:", mesh.ufl_cell())        # should print "triangle"
print("# vertices:", mesh.num_vertices())

# 2) Try building your scalar H1 space
V = FunctionSpace(mesh, "CG", 2)
print("FunctionSpace built successfully!")
exit()
