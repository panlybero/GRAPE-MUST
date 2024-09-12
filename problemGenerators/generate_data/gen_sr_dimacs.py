# This file is adapted from https://github.com/dselsam/neurosat
# Below is the original license statement:
##############################################################
#  Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import copy
import math
import numpy as np
import random
import argparse
from io import StringIO

# import PyMiniSolvers.minisolvers as minisolvers
from pysat.solvers import Glucose3


def write_dimacs_to(n_vars, iclauses, out_filename):
    with open(out_filename, "w") as f:
        f.write("p cnf %d %d\n" % (n_vars, len(iclauses)))
        for c in iclauses:
            for x in c:
                f.write("%d " % x)
            f.write("0\n")


def write_dimacs_to_IO(n_vars, iclauses):
    f = StringIO()
    f.write(f"p cnf {n_vars} {len(iclauses)}\n")
    for c in iclauses:
        for x in c:
            f.write(f"{x} ")
        f.write("0\n")

    f.seek(0)
    return f


def mk_out_filenames(opts, n_vars, t):
    prefix = "%s/sr_n=%.4d_pk2=%.2f_pg=%.2f_t=%d" % (
        opts.out_dir,
        n_vars,
        opts.p_k_2,
        opts.p_geo,
        t,
    )
    return ("%s_sat=0.dimacs" % prefix, "%s_sat=1.dimacs" % prefix)


def gen_iclause_pair_easy(opts):

    n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(opts)

    extra_var = n_vars + 1
    # print(iclauses)
    solver = Glucose3()
    for iclause in iclauses:
        solver.add_clause(iclause)

    before_last = solver.solve()
    iclauses.append([extra_var])
    iclause_unsat = [-extra_var]
    iclause_sat = [extra_var]
    solver.add_clause(iclause_sat)
    with_sat = solver.solve()

    solver = Glucose3()
    for iclause in iclauses:
        solver.add_clause(iclause)
    solver.add_clause(iclause_unsat)
    with_unsat = solver.solve()

    assert before_last, "before_last should be sat"
    assert with_sat, "with_sat should be sat"
    assert not with_unsat, "with_unsat should be unsat"

    return n_vars + 1, iclauses, iclause_unsat, iclause_sat


def gen_iclause_pair_with_sanity_check(opts):

    if opts.easy:
        n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair_easy(
            opts)
    else:
        n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(opts)

    # print(iclauses)
    solver = Glucose3()
    for iclause in iclauses:
        solver.add_clause(iclause)

    before_last = solver.solve()
    solver.add_clause(iclause_sat)
    with_sat = solver.solve()

    solver = Glucose3()
    for iclause in iclauses:
        solver.add_clause(iclause)
    solver.add_clause(iclause_unsat)
    with_unsat = solver.solve()

    assert before_last, "before_last should be sat"
    assert with_sat, "with_sat should be sat"
    assert not with_unsat, "with_unsat should be unsat"

    return n_vars, iclauses, iclause_unsat, iclause_sat


def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]


def gen_iclause_pair(opts):
    n = random.randint(opts.min_n, opts.max_n)

    solver = Glucose3()  # minisolvers.MinisatSolver()
    # for i in range(n):
    #     solver.new_var(dvar=True)

    iclauses = []

    while True:
        k_base = 1 if random.random() < opts.p_k_2 else 2
        k = k_base + np.random.geometric(opts.p_geo)
        iclause = generate_k_iclause(n, k)
        iclause = [int(x) for x in iclause]
        solver.add_clause(iclause)
        is_sat = solver.solve()
        if is_sat:
            iclauses.append(iclause)
        else:
            break

    iclause_unsat = iclause
    iclause_sat = [-iclause_unsat[0]] + iclause_unsat[1:]
    return n, iclauses, iclause_unsat, iclause_sat


def generate_data_online(
    n_pairs, min_n, max_n, p_k_2=0.3, p_geo=0.4, out_dir="./", easy=False
):
    opts = argparse.Namespace()
    opts.min_n = min_n
    opts.max_n = max_n
    opts.p_k_2 = p_k_2
    opts.p_geo = p_geo
    opts.out_dir = out_dir
    opts.easy = easy
    data = []
    for t in range(n_pairs):
        (
            n_vars,
            iclauses,
            iclause_unsat,
            iclause_sat,
        ) = gen_iclause_pair_with_sanity_check(opts)

        unsat = copy.deepcopy(iclauses) + [iclause_unsat]
        sat = copy.deepcopy(iclauses) + [iclause_sat]
        unsat_str = write_dimacs_to_IO(n_vars, unsat)
        sat_str = write_dimacs_to_IO(n_vars, sat)
        data.append((unsat_str, sat_str))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", action="store", type=str)
    parser.add_argument("n_pairs", action="store", type=int)

    parser.add_argument("--min_n", action="store",
                        dest="min_n", type=int, default=40)
    parser.add_argument("--max_n", action="store",
                        dest="max_n", type=int, default=40)

    parser.add_argument(
        "--p_k_2", action="store", dest="p_k_2", type=float, default=0.3
    )
    parser.add_argument(
        "--p_geo", action="store", dest="p_geo", type=float, default=0.4
    )

    parser.add_argument(
        "--py_seed", action="store", dest="py_seed", type=int, default=None
    )
    parser.add_argument(
        "--np_seed", action="store", dest="np_seed", type=int, default=None
    )

    parser.add_argument(
        "--print_interval", action="store", dest="print_interval", type=int, default=100
    )
    parser.add_argument("--easy", action="store", default=False)

    opts = parser.parse_args()

    if opts.py_seed is not None:
        random.seed(opts.py_seed)
    if opts.np_seed is not None:
        np.random.seed(opts.np_seed)

    for pair in range(opts.n_pairs):
        if pair % opts.print_interval == 0:
            print("[%d]" % pair)
        (
            n_vars,
            iclauses,
            iclause_unsat,
            iclause_sat,
        ) = gen_iclause_pair_with_sanity_check(opts)
        out_filenames = mk_out_filenames(opts, n_vars, pair)

        iclauses.append(iclause_unsat)
        write_dimacs_to(n_vars, iclauses, out_filenames[0])

        iclauses[-1] = iclause_sat
        write_dimacs_to(n_vars, iclauses, out_filenames[1])
