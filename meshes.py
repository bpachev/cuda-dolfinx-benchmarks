from mpi4py import MPI
from dolfinx import mesh

def make_cubic_mesh(res=10):
    return mesh.create_box(
        comm = MPI.COMM_WORLD,
        points = ((0,0,0), (1, 1, 1)),
        n = (res, res, res),
        cell_type = mesh.CellType.tetrahedron
    )
