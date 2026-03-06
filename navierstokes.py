"""Contains forms representing the most intensive portion of Jorgen's 3-step Navier-Stokes solver - the tentative step."""

import basix.ufl
import ufl
from ufl import dx
from dolfinx import fem as fe
from dolfinx import default_scalar_type

def get_forms(domain, degree=2):
    """Return Poisson forms for a given polynomial degree."""

    if degree < 2:
        raise ValueError("Degree must be at least 2 for Navier-Stokes!")

    #domain = Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
    V_element = basix.ufl.element("Lagrange", "tetrahedron", degree, shape=(domain.geometry.dim,))
    Q_element = basix.ufl.element("Lagrange", "tetrahedron", degree-1)
    V = fe.functionspace(domain, V_element)
    Q = fe.functionspace(domain, Q_element)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    uh = fe.Function(V)
    u_old = fe.Function(V)
    ph = fe.Function(Q)
    dt = 0.01
    nu = 0.001
    f = fe.Constant(domain, default_scalar_type((0,) * domain.geometry.dim))
    w_time = fe.Constant(domain, 3 / (2 * dt))
    w_diffusion = fe.Constant(domain, default_scalar_type(nu))
    a_tent = w_time * ufl.inner(u, v) * dx + w_diffusion * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L_tent = (ph*ufl.div(v) + ufl.inner(f, v)) * dx
    L_tent += fe.Constant(domain, 1 / (2 * dt)) * ufl.inner(4 * uh - u_old, v) * dx
    # BDF2 with implicit Adams-Bashforth
    bs = 2 * uh - u_old
    a_tent += ufl.inner(ufl.grad(u) * bs, v) * dx
    # Temam-device
    a_tent += 0.5 * ufl.div(bs) * ufl.inner(u, v) * dx

    return a_tent, L_tent



