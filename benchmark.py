import meshes
import poisson
import elasticity
import navierstokes
import argparse as ap
import time
import cudolfinx as cufem

problems = {
  "poisson": poisson,
  "elasticity": elasticity,
  "navierstokes": navierstokes,
}

def main(problem, reps, degree, no_quadrature=False, cuda=True, res=100):
    """Perform benchmarking of cuDOLFINx assembly routines for a given linear form."""

    # create mesh
    # TODO add options for mesh size/type
    domain = meshes.make_cubic_mesh(res=res)
    
    problem_module = problems[problem]

    a, L = problem_module.get_forms(domain, degree=degree)

    # need to assign a mesh somehow to the form

    if cuda:
        a = cufem.form(a, form_compiler_options={"disable_tabulate_tensors": no_quadrature})
        L = cufem.form(L, form_compiler_options={"disable_tabulate_tensors": no_quadrature})
        asm = cufem.CUDAAssembler()
        cuda_A = asm.create_matrix(a)
        cuda_b = asm.create_vector(L)
        A = cuda_A.mat
        b = cuda_b.vector
    else:
        a = fe.form(a, jit_options = {"cffi_extra_compile_args":["-O3", "-mcpu=neoverse-v2"]})
        L = fe.form(L, jit_options = {"cffi_extra_compile_args":["-O3", "-mcpu=neoverse-v2"]})
        A = fe_petsc.create_matrix(a)
        b = fe_petsc.create_vector(L)
    print(A.size)
    timing = {"mat_assemble": 0.0, "vec_assemble": 0.0}
    for i in range(reps):
        start = time.time()
        if cuda:
            asm.assemble_matrix(a, cuda_A)
            cuda_A.assemble()
        else:
            A.zeroEntries()
            fe_petsc.assemble_matrix(A, a)
            A.assemble()
        timing["mat_assemble"] += time.time()-start
        start = time.time()
        if cuda:
            asm.assemble_vector(L, cuda_b)
        else:
            fe_petsc.assemble_vector(b, L)
        timing["vec_assemble"] += time.time()-start

    for k,v in timing.items():
        print(f"Average timing for '{k}' over {reps} reps: {v/reps}s")

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--problem", choices=sorted(list(problems.keys())), default="poisson")
    parser.add_argument("--reps", type=int, default=1, help="Number of trials to average results over.")
    parser.add_argument("--degree", type=int, default=1, help="Polynomial degree of elements.")
    parser.add_argument("--no-quadrature", action="store_true", default=False, help="Disable quadrature (assembly loop only).")
    parser.add_argument("--cpu", action="store_true", default=False, help="Test DOLFINx as a baseline (no cuDOLFINx)")
    parser.add_argument("--res", type=int, default=100, help="Number of cells in each direction for cubic mesh.")
    args = parser.parse_args()

    main(
        problem=args.problem,
        reps=args.reps,
        degree=args.degree,
        no_quadrature=args.no_quadrature,
        cuda = not args.cpu,
        res = args.res,
    )
