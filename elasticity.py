# Copyright (C) 2017-2022 Chris N. Richardson and Garth N. Wells
#
# This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT

import basix.ufl
from ufl import (Coefficient, Identity, FunctionSpace, Mesh, TestFunction, TrialFunction,
                 dx, grad, inner, tetrahedron, tr, SpatialCoordinate, as_vector)
from dolfinx import fem as fe

# Elasticity parameters
E = 1.0e6
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
cell = tetrahedron

def get_forms(domain, degree=1):
    """Get UFL forms for the elasticity problem for a given polynomial degree."""
    element = basix.ufl.element("Lagrange", "tetrahedron", degree, shape=(3, ))
    #domain = Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3, )))
    space = fe.functionspace(domain, element)

    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(domain)
    f = as_vector([x[0], x[1], x[2]])

    def eps(v):
        return 0.5*(grad(v) + grad(v).T)

    def sigma(v):
        return 2.0*mu*eps(v) + lmbda*tr(eps(v))*Identity(3)

    a = inner(sigma(u), eps(v))*dx
    L = inner(f, v)*dx
    return a, L
