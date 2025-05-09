import meshio

# 1) Read the Gmsh file
m = meshio.read("Mesh/mesh_ellipse_circle.msh")

# 2) Separate out triangles and lines + their physical tags
tri_cells = []
tri_data  = []
line_cells = []
line_data  = []

for block, phys in zip(m.cells, m.cell_data["gmsh:physical"]):
    if block.type == "triangle":
        tri_cells.append(block)
        tri_data.append(phys)
    elif block.type == "line":
        line_cells.append(block)
        line_data.append(phys)

# 3) Write the 2D mesh (triangles only)
mesh2d = meshio.Mesh(
    points=m.points,
    cells=[("triangle", tri_cells[0].data)],
    cell_data={"name_to_read": [tri_data[0]]},
)
meshio.write("Mesh/mesh_ellipse_circle_2d.xdmf", mesh2d)

# 4) Write the 1D facet‐markers (lines only)
facets = meshio.Mesh(
    points=m.points,
    cells=[("line", line_cells[0].data)],
    cell_data={"name_to_read": [line_data[0]]},
)
meshio.write("Mesh/mesh_ellipse_circle_facet.xdmf", facets)
