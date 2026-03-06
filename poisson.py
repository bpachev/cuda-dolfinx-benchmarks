import basix.ufl
from ufl import (Coefficient, FunctionSpace, TestFunction, TrialFunction, Mesh, action, ds,
                 dx, grad, inner, tetrahedron, exp, sin, SpatialCoordinate)
from dolfinx import fem as fe

def get_forms(domain, degree=1):
    """Return Poisson forms for a given polynomial degree."""

    element = basix.ufl.element("Lagrange", "tetrahedron", degree)
    #domain = Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
    space = fe.functionspace(domain, element)

    u = TrialFunction(space)
    v = TestFunction(space)
    x = SpatialCoordinate(domain)
    f = 10*exp(-((x[0]-.05)**2 + (x[1]-.05)**2 + (x[2]-.05)**2) / .02)
    g = sin(5*x[0])*sin(5*x[1])
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    return a, L
