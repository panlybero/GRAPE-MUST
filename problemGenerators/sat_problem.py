import datetime
import multiprocessing
from typing import NamedTuple, Optional

import networkx as nx
import torch_geometric as pyg
from torch_geometric.data import HeteroData, Data
import torch_geometric.utils as pyg_utils
import torch
import matplotlib.pyplot as plt
from pysat.solvers import Glucose3
import numpy as np
import sys
import scipy as sp
import warnings
from problemGenerators.generate_data.gen_sr_dimacs import generate_data_online
from problemGenerators.generate_data.graph_color2cnf import random_graph_color, graph2cnf, random_erdos_graph_color
from problemGenerators.generate_data.plan_to_sat import random_grid_planning_cnf, random_logistics_cnf
from problemGenerators.generate_data.random_ksat import random_ksat
from torch_geometric.utils import to_networkx
import copy
import subprocess
import re
import bz2
from threading import Timer
# @torch.jit.script


def make_graph(n_vars, clauses, spectral_dim, sat_problem):
    """Convert the problem to a graph."""
    n_var_nodes = n_vars * 2  # one node for each variable and its negation
    n_clauses = len(clauses)
    data = HeteroData()
    vars_x = torch.zeros((n_var_nodes, 1))
    clauses_x = torch.zeros((n_clauses, 1))

    edges = []
    clause_to_var_edges = []
    for i, clause in enumerate(clauses):

        for var in clause:
            var = int(var)
            if var > 0:
                # var-1 because variables are 1-indexed
                edges.append((var - 1, i))
                clause_to_var_edges.append((i, var - 1))
            else:
                # negated variable is n_vars+var-1
                edges.append((n_vars + abs(var) - 1, i))
                clause_to_var_edges.append((i, n_vars + abs(var) - 1))

    vars_clauses = torch.tensor(
        edges).t().contiguous()
    clauses_vars = (
        torch.tensor(clause_to_var_edges).t().contiguous()
    )
    # connect each variable to its negation
    edges = []
    for i in range(n_vars):
        edges.append((i, n_vars + i))
        edges.append((n_vars + i, i))
    vars_vars = torch.tensor(edges).t().contiguous()

    return vars_x, clauses_x, vars_vars, vars_clauses, clauses_vars


class SatProblem:
    """A SAT problem."""

    def __init__(self, n_vars: int, clauses: list, isSat=None):
        self.n_vars = n_vars
        self.clauses = clauses
        self._isSat = isSat
        # if isSat is None:
        #     self._isSat = self.solve()
        # else:
        #     self._isSat = isSat

    def __hash__(self) -> int:
        hashable_clauses = tuple(tuple(clause) for clause in self.clauses)
        return hash((self.n_vars, hashable_clauses))

    def run_muser2(self, get_time=False) -> int:
        """Run the muser2 algorithm on the problem."""
        path = "/home/plymper/unsat-detection/solvers/marco/marco_py-3.0/src/marco/muser2-para"
        formula_dimacs = self.to_dimacs()
        fname = f"/tmp/tmp_cnf_muser_{np.random.randint(0, 1000000)}_{np.random.randint(0, 1000000)}.cnf"
        with open(fname, "w") as f:
            f.write(formula_dimacs)

        cmd = f"{path} {fname}"
        try:
            p = subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as e:

            p = e.output

        p = p.decode("utf-8")
        p = p.split("\n")
        for line in p:
            if "MUS size:" not in line:
                continue

            match = re.findall("\(([^\)]+)\)", line)
            if match is None:
                raise ValueError("Could not parse muser2 output.")

        percentage = float(match[-1][:-1])

        return percentage/100

    @property
    def isSat(self):
        return self.solve()

    @property
    def n_clauses(self):
        return len(self.clauses)

    @classmethod
    def from_lines(cls, lines: list[str]):
        """Create a problem from a list of lines."""
        # known sat?
        isSat = None

        clauses = []
        for l in lines:

            if l.startswith("c"):  # this is a comment
                continue
            if l.startswith("p"):
                n_variables, n_clauses = l.split()[-2:]  # this is the header
                continue
            # remove spaces and the last 0
            elems = l.strip().split()[:-1]

            elems = [int(e) for e in elems]
            clauses.append(elems)

        return cls(int(n_variables), clauses, isSat=isSat)

    @classmethod
    def combine_problems(cls, problem1, problem2):
        """Combine two problems into one."""
        p1_max_var = problem1.n_vars
        # renumber the variables in problem2
        new_clauses = []
        for clause in problem2.clauses:
            new_clause = []
            for var in clause:
                if var > 0:
                    new_clause.append(var+p1_max_var)
                else:
                    new_clause.append(var-p1_max_var)
            new_clauses.append(new_clause)

        total_n_vars = problem1.n_vars + problem2.n_vars
        return cls(total_n_vars,  problem1.clauses + new_clauses)

    @classmethod
    def from_file(cls, filename: str):
        """Read a CNF file and return a SatProblem instance."""

        with open(filename) as f:
            lines = f.readlines()

        return cls.from_lines(lines)

    @classmethod
    def random_problem(cls, min_vars: int, max_vars: int, sat: Optional[bool] = None, return_both=False, p_geo: float = 0.4):
        pair = generate_data_online(1, min_vars, max_vars, p_geo=p_geo)[0]

        # pos = cls.from_lines(pair[0].readlines())
        # neg = cls.from_lines(pair[1].readlines())

        if return_both:
            return cls.from_lines(pair[1].readlines()), cls.from_lines(pair[0].readlines())

        if sat is None:
            sat = np.random.choice([True, False])

        if sat:
            res = cls.from_lines(pair[1].readlines())
            return res
        else:
            res = cls.from_lines(pair[0].readlines())
            return res

    @classmethod
    def random_graph_color_problem(cls, n_nodes: int, min_n_colors: int, max_n_colors: int = None, sat: Optional[bool] = None, edge_prob: Optional[float] = None):
        """Create a random graph coloring problem.

        If sat is None, the problem will be randomly generated as sat or unsat.
        """
        if sat is None:
            sat = np.random.choice([True, False])

        # * (n_nodes - 1) / 2)

        if max_n_colors is None:
            max_n_colors = min_n_colors

        n_colors = np.random.randint(low=min_n_colors, high=max_n_colors+1)

        max_edges = n_nodes*(n_nodes-1)/2
        n_edges = np.random.randint(n_nodes, max_edges)

        if edge_prob is None:
            cnf = random_graph_color(n_nodes, n_colors, n_edges=n_edges)
        else:
            cnf = random_erdos_graph_color(n_nodes, n_colors, edge_prob)

        problem = SatProblem.from_lines(cnf)
        count = 0
        # print("looking for problem")
        while (problem.isSat != sat):
            # print(f"Problem isSat!={sat}, generating a new one")
            n_edges = np.random.randint(n_nodes, max_edges)
            n_colors = np.random.randint(low=min_n_colors, high=max_n_colors+1)
            if edge_prob is None:
                cnf = random_graph_color(n_nodes, n_colors, n_edges=n_edges)
            else:
                # print("Erdos", edge_prob)
                cnf = random_erdos_graph_color(n_nodes, n_colors, edge_prob)

            # problem = SatProblem.from_lines(
            #     random_graph_color(n_nodes, n_colors, n_edges=n_edges))
            problem = SatProblem.from_lines(cnf)
            count += 1

        # print("Done", count)
        return problem

    @classmethod
    def random_ksat_problem(cls, n_vars: int, k: int = 3, ratio: float = 5, sat: Optional[bool] = None):
        """Create a random k-SAT problem.

        If sat is None, the problem will be randomly generated as sat or unsat.
        """
        if sat is None:
            sat = np.random.choice([True, False])

        problem = SatProblem.from_lines(random_ksat(n_vars, ratio, k))
        count = 0
        while (problem.isSat != sat):
            problem = SatProblem.from_lines(random_ksat(n_vars, ratio, k))
            count += 1

        return problem

    @classmethod
    def random_grid_planning_problem(cls, grid_size: tuple[int, int], traj_length: int = 10, sat: Optional[bool] = None):
        if sat is None:
            sat = np.random.choice([True, False])
        nv, clauses = random_grid_planning_cnf(grid_size, traj_length)
        problem = cls(nv, clauses)
        count = 0
        while (problem.isSat != sat):
            nv, clauses = random_grid_planning_cnf(grid_size, traj_length)
            # print("Problem isSat!=", sat, "generating a new one")
            problem = cls(nv, clauses)
            count += 1
        #print("Done", count)
        return problem

    @classmethod
    def random_logistics_planning_problem(cls, n_planes: int = 1, n_airports: int = 2, n_locations: int = 2, n_cities: int = 2, n_trucks: int = 2, n_packages: int = 10, trajectory_length: int = 5, sat: bool = False):
        if sat is None:
            sat = np.random.choice([True, False])

        # np.random.randint(n_packages, n_packages*2)
        n_packages_s = n_packages

        nv, clauses = random_logistics_cnf(
            n_planes, n_airports, n_locations, n_cities, n_trucks, n_packages_s, trajectory_length)
        problem = cls(nv, clauses)
        count = 0
        while (problem.isSat != sat):

            # np.random.randint(n_packages, n_packages*2)
            n_packages_s = np.random.randint(n_packages//2, n_packages)
            nv, clauses = random_logistics_cnf(
                n_planes, n_airports, n_locations, n_cities, n_trucks, n_packages_s, trajectory_length)
            # print("Problem isSat!=", sat, "generating a new one")
            problem = cls(nv, clauses)
            count += 1
        return problem

    @classmethod
    def random_formula_from_stats(cls, stats, sat: bool = False):
        "Generate a random formula according to the given stats"
        clause_length_dist = stats['clause_length_dist']
        lit_degree_dist = stats['lit_degree_dist']
        ratios = stats['ratios']

        num_clauses = np.random.randint(50, 10000)
        ratio = np.random.choice(ratios)
        num_vars = int(num_clauses * ratio)

        clause_length_dist = {int(k): v for k, v in clause_length_dist.items()}
        clause_length_dist = sorted(
            clause_length_dist.items(), key=lambda x: x[0])
        clause_length_dist = np.array(clause_length_dist)

        clause_lengths = np.random.choice(
            clause_length_dist[:, 0], num_clauses, p=clause_length_dist[:, 1])

        # build the clauses
        clauses = cls._build_clauses(num_vars, clause_lengths)
        problem = cls(num_vars, clauses)
        while problem.isSat != sat:
            clauses = cls._build_clauses(num_vars, clause_lengths)
            problem = cls(num_vars, clauses)

        return problem

    @classmethod
    def random_formula_from_stats_sequential(cls, stats, sat: bool = False, num_vars_range = (50, 1000)):
        "Generate a random formula according to the given stats"
        clause_length_dist = stats['clause_length_dist']
        ratios = stats['ratios']

        #num_clauses = np.random.randint(50, 10000)
        ratio = np.random.choice(ratios)
        # 100  # int(num_clauses * ratio)
        num_vars = np.random.randint(*num_vars_range)
        #num_vars = 2000

        clause_length_dist = {int(k): v for k, v in clause_length_dist.items()}
        clause_length_dist = sorted(
            clause_length_dist.items(), key=lambda x: x[0])
        clause_length_dist = np.array(clause_length_dist)

        problem = SatProblem(1, [[1]])
        clauses = []
        while problem.isSat:  # or len(clauses) < num_vars*ratio:
            #print(len(clauses), num_vars*ratio)

            clause_length = np.random.choice(
                clause_length_dist[:, 0], p=clause_length_dist[:, 1])

            clause = cls._build_clauses(num_vars, [clause_length])[0]
            clauses.append(clause)
            new_problem = cls(num_vars, clauses)
            #print(len(clauses))
            if not new_problem.isSat and len(clauses) < num_vars*ratio:
                clauses.pop()
            else:
                problem = new_problem

        return problem

    @classmethod
    def _build_clauses(cls, num_vars, clause_lengths):
        clauses = []
        for clause_length in clause_lengths:
            clause = []
            for i in range(int(clause_length)):
                lit = int(np.random.choice([-1, 1]) *
                          np.random.randint(1, num_vars+1))
                while -lit in clause:
                    lit = int(np.random.choice(
                        [-1, 1]) * np.random.randint(1, num_vars+1))

                clause.append(lit)
            clauses.append(clause)
        return clauses

    @classmethod
    def from_bz2(cls, bz2_path: str):
        """Load a problem from a bz2 file."""

        with bz2.open(bz2_path, "rt") as f:
            lines = f.readlines()

            return cls.from_lines(lines)

    def solve_interruptible(self, timeout=10):
        solver = Glucose3()
        for clause in self.clauses:
            solver.add_clause(clause)

        def interrupt(s):
            s.interrupt()

        timer = Timer(timeout, interrupt, [solver])
        timer.start()
        isSat = solver.solve_limited(expect_interrupt=True)
        solver.delete()
        return isSat

    def solve(self):
        """Solve the problem using pysat."""

        if self._isSat is None:

            with Glucose3() as g:
                for clause in self.clauses:
                    g.add_clause(clause)
                self._isSat = g.solve()

        return self._isSat

    def toHeteroGraph(self, spectral_dim=10, use_spectral_emb=True):
        """Convert the problem to a graph."""

        # vars_x, clauses_x, vars_vars, vars_clauses, clauses_vars= make_graph(self.n_vars, self.clauses, spectral_dim, self)

        # return data
        n_var_nodes = self.n_vars * 2  # one node for each variable and its negation
        n_clauses = len(self.clauses)
        data = HeteroData()
        data["vars"].x = torch.zeros((n_var_nodes, 1))
        data["clauses"].x = torch.zeros((n_clauses, 1))

        edges = []
        clause_to_var_edges = []
        # print("Making Edges")
        for i, clause in enumerate(self.clauses):

            for var in clause:
                var = int(var)
                if var > 0:
                    # var-1 because variables are 1-indexed
                    edges.append((var - 1, i))
                    clause_to_var_edges.append((i, var - 1))
                else:
                    # negated variable is n_vars+var-1
                    edges.append((self.n_vars + abs(var) - 1, i))
                    clause_to_var_edges.append((i, self.n_vars + abs(var) - 1))

        data["vars", "clauses"].edge_index = torch.tensor(
            edges).t().contiguous()
        data["clauses", "vars"].edge_index = (
            torch.tensor(clause_to_var_edges).t().contiguous()
        )

        # connect each variable to its negation
        edges = []
        for i in range(self.n_vars):
            edges.append((i, self.n_vars + i))
            edges.append((self.n_vars + i, i))
        data["vars", "vars"].edge_index = torch.tensor(edges).t().contiguous()
        # print("Done")
        # add dense connectivity between clauses
        # edges = []
        # for i in range(n_clauses):
        #     for j in range(i + 1, n_clauses):
        #         edges.append((i, j))
        #         edges.append((j, i))

        # connect only to clauses with at least one variable in common
        # edges = []
        # for i in range(n_clauses):
        #     for j in range(i + 1, n_clauses):
        #         if len(set(self.clauses[i]) & set(self.clauses[j])) > 0:
        #             edges.append((i, j))
        #             edges.append((j, i))

        # data["clauses", "clauses"].edge_index = torch.tensor(
        #     edges).t().contiguous().long()

        data.sat_problem_hash = self.__hash__()
        data.sat_problem = self
        data.isSat = self.solve()
        # print("Calculating Spectral Embeddings")
        if use_spectral_emb:
            spec = self.spectral_embeddings(data, spectral_dim=spectral_dim)
        else:  # random
            spec = {"vars": torch.randn(
                (n_var_nodes, spectral_dim))*0.01, "clauses": torch.randn((n_clauses, spectral_dim))*0.01}

        # print("Done")
        data["vars"].x = spec["vars"]
        data["clauses"].x = spec["clauses"]

        return data

    @ classmethod
    def spectral_embeddings(cls, graph: HeteroData, spectral_dim: int = 10):
        """
        Embeds the graph using first 'spectral_dim' eigenvectors of the Laplacian matrix.
        If n_nodes < spectral_dim, then the eigenvectors are padded with zeros.
        """

        homog = graph.to_homogeneous()

        node_types = homog.node_type

        homog.edge_index = pyg_utils.to_undirected(
            homog.edge_index, homog.edge_attr, homog.num_nodes
        )

        graphx = pyg_utils.to_networkx(homog)
        graphx = nx.to_undirected(graphx)

        # spec_ = np.vstack(
        #     list(nx.spectral_layout(
        #         graphx, dim=graphx.number_of_nodes() - 1).values())
        # )

        padded_matrix = np.zeros((graphx.number_of_nodes(), spectral_dim))
        true_spec_dim = min(spectral_dim, graphx.number_of_nodes() - 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            norm_lap = nx.normalized_laplacian_matrix(graphx)
            eigvals, eigvecs = sp.sparse.linalg.eigsh(
                norm_lap,
                k=true_spec_dim,
                which="SM",
                return_eigenvectors=True,
                tol=1e-3,
            )
            spec_ = eigvecs[:, :spectral_dim]

        padded_matrix[: spec_.shape[0], : spec_.shape[1]] = spec_
        spec_ = padded_matrix
        # spec_ = spec_[:, :spectral_dim]
        return {
            "vars": torch.tensor(spec_[node_types == 0]),
            "clauses": torch.tensor(spec_[node_types == 1]),
        }

    def toSequence(self):
        """Convert the problem to a sequence of clauses."""

        formula_str = ""
        for clause in self.clauses:

            formula_str += "(" + "|".join([str(l) for l in clause]) + ")&"

        formula_str = formula_str[:-1]

        return formula_str

    def toGraph(self, emb_dim=64):
        """Convert the problem to a homogeneous undirected graph.

        Each node is a variable or a clause.
        Two types of edges reflected as edge attributes:
        - clause -> variable
        - variable -> variable (negation)

        Node embeddings are None and should be implemented by the model
        """

        n_nodes = self.n_vars * 2  # one node for each variable and its negation
        n_clauses = len(self.clauses)
        data = Data()
        edges = []
        edge_types = []
        node_type = []
        for i, clause in enumerate(self.clauses):
            for var in clause[:-1]:  # ignore the last 0
                var = int(var)
                if var > 0:
                    # var-1 because variables are 1-indexed
                    edges.append((var - 1, i))
                else:
                    # negated variable is n_vars+var-1
                    edges.append((self.n_vars + (-var) - 1, i))
                edge_types.append([1, 0])
                node_type.append([0, 1])

        for i in range(self.n_vars):
            edges.append((i, self.n_vars + i))
            edge_types.append([0, 1])

        data.edge_index = torch.tensor(edges).t().contiguous()
        data.edge_attr = torch.tensor(edge_types).float()
        data.isSat = self.solve()

        return data

    @ classmethod
    def removeClauses(cls, cnf: "SatProblem", clauses: list):
        """
        Return a new SAT problem with the given clauses removed.
        Caution: The clauses and variables are renumbered.

        cnf: the original problem
        clauses: the clauses to remove

        """

        new_clauses = copy.deepcopy(cnf.clauses)

        sorted_rem = sorted(clauses.cpu().numpy(), reverse=True)

        # remove them
        for i in sorted_rem:
            del new_clauses[i]

        # for i, clause in enumerate(cnf.clauses):
        #     if i not in clauses:

        #         new_clauses.append(clause)

        all_vars = set([abs(int(lit))
                       for clause in new_clauses for lit in clause])
        new_vars = {v: i + 1 for i, v in enumerate(sorted(all_vars))}
        new_clauses_renumbered = [
            [
                (new_vars[abs(int(lit))])
                if int(lit) > 0
                else -(new_vars[abs(int(lit))])
                for lit in clause
            ]
            for clause in new_clauses
        ]
        return cls(len(new_vars), new_clauses_renumbered)

    @classmethod
    def getSubProblem(cls, cnf: "SatProblem", clauses: list):
        all_clauses = copy.deepcopy(cnf.clauses)
        new_clauses = []
        sorted_rem = sorted(clauses.cpu().numpy(), reverse=True)

        # remove them
        for i in sorted_rem:
            new_clauses.append(all_clauses[i])

        all_vars = set([abs(int(lit))
                       for clause in new_clauses for lit in clause])
        new_vars = {v: i + 1 for i, v in enumerate(sorted(all_vars))}
        new_clauses_renumbered = [
            [
                (new_vars[abs(int(lit))])
                if int(lit) > 0
                else -(new_vars[abs(int(lit))])
                for lit in clause
            ]
            for clause in new_clauses
        ]
        return cls(len(new_vars), new_clauses_renumbered)

    @classmethod
    def splitProblem(cls, cnf: "SatProblem", clauses: list):

        p1 = cls.removeClauses(cnf, clauses)
        p2 = cls.getSubProblem(cnf, clauses)

        return p1, p2

    def __repr__(self) -> str:
        return f"n_vars={self.n_vars}, n_clauses={self.n_clauses}"

    @ classmethod
    def fromGraph(cls, data: Data):
        n_vars = data["vars"].x.shape[0] / 2
        assert n_vars == int(n_vars)
        n_vars = int(n_vars)
        n_clauses = data["clauses"].x.shape[0]
        vars_to_clauses = list(zip(*data["vars", "clauses"].edge_index))
        vars_to_clauses = [(v.item(), c.item()) for v, c in vars_to_clauses]

        vars_to_vars = list(zip(*data["vars", "vars"].edge_index))
        vars_to_vars = [(v.item(), c.item()) for v, c in vars_to_vars]

        # map variables to original variable numbers
        # Use edges to find vars and their negation, as those should be consistent
        map_lit_to_var = {}
        for i in range(n_vars):
            assert (i, n_vars + i) in vars_to_vars
            assert (n_vars + i, i) in vars_to_vars
            map_lit_to_var[i] = i + 1
            map_lit_to_var[n_vars + i] = -(i + 1)

        # map clauses to their literals
        clauses = [[] for _ in range(n_clauses)]
        for v, c in vars_to_clauses:
            clauses[c].append(map_lit_to_var[v])

        return cls(n_vars, clauses)

    def to_dimacs(self):
        """
        Convert to DIMACS format
        """
        dimacs = f"p cnf {self.n_vars} {len(self.clauses)}\n"
        for clause in self.clauses:
            clause_str = ''
            for lit in clause:
                clause_str += f"{lit} "
            clause_str += '0\n'
            dimacs += clause_str
        return dimacs

    def to_dimacs_file(self, filename):
        """
        Write to a DIMACS file
        """
        with open(filename, 'w') as f:
            f.write(self.to_dimacs())

    def to_bz2_dimacs_file(self, filename):
        """
        Write to a bz2-compressed DIMACS file
        """
        with bz2.open(filename, 'wt') as f:
            f.write(self.to_dimacs())

    def to_networkx(self):

        heterograph = self.toHeteroGraph()
        homograph = heterograph.to_homogeneous()

        # nodetypes = #homograph.node_type.numpy().tolist()

        graph = to_networkx(homograph, node_attrs=[
                            "node_types"]).to_undirected()

        return graph


def check_consistency(cnf, plot_graph=False):
    heterograph = cnf.toHeteroGraph()

    homograph = heterograph.to_homogeneous()
    nodetypes = homograph.node_type
    graph = to_networkx(homograph).to_undirected()

    # arrange grp1 and grp2 nodes in different layers
    pos = {}
    grp1 = torch.where(nodetypes == 0)[0]
    for i, n in enumerate(grp1):
        pos[n.item()] = (i % 2, i)
    grp2 = torch.where(nodetypes == 1)[0]
    for i, n in enumerate(grp2):
        pos[n.item()] = (5, i)

    node_colors = homograph.x[:, -1].numpy()
    edge_colors = [
        "black" if n[0] in grp1 and n[1] in grp1 else "gray" for n in graph.edges
    ]
    print(homograph)
    # pos = {i: homograph.eigen[i] for i in range(len(pos))}

    if plot_graph:
        nx.draw(
            graph,
            with_labels=True,
            pos=pos,
            node_color=node_colors,
            edge_color=edge_colors,
        )
        plt.savefig(f"graph_sat={cnf.isSat}.png")

    back_to_cnf = SatProblem.fromGraph(heterograph)
    assert (
        cnf.isSat == back_to_cnf.isSat
    ), "isSat does not match after conversion and back"

    assert (
        cnf.n_vars == back_to_cnf.n_vars
    ), "n_vars does not match after conversion and back"

    assert (
        cnf.n_clauses == back_to_cnf.n_clauses
    ), "n_clauses does not match after conversion and back"

    assert str(cnf.clauses) == str(back_to_cnf.clauses), "clauses do not match"


def graph_color_sat_to_graph(clauses, colors, return_var_to_node=False):
    n_nodes = max([max(c) for c in clauses])
    var_to_node = {}
    node_counts = {}
    for i, c in enumerate(clauses):
        if len(c) == colors:
            for v in c:
                if v not in var_to_node:
                    var_to_node[v] = i+1
            node_counts[i+1] = 1
    n_added = 0

    edges = []
    edge_counts = {}

    for c in clauses:
        if len(c) == 2:
            try:
                if var_to_node[abs(c[0])] == var_to_node[abs(c[1])]:
                    node_counts[var_to_node[abs(c[0])]] = node_counts.get(
                        var_to_node[abs(c[0])], 0) + 1
                    continue
                edges.append([var_to_node[abs(c[0])], var_to_node[abs(c[1])]])
                val = edge_counts.get(
                    (var_to_node[abs(c[0])], var_to_node[abs(c[1])]), 0)
                edge_counts[(var_to_node[abs(c[0])],
                             var_to_node[abs(c[1])])] = val + 1
            except KeyError as e:
                continue

    g = nx.Graph(edges)
    if return_var_to_node:
        return g, var_to_node, edge_counts, node_counts
    return g


def general_prob_str_format(prob_str, n_vars):
    var_names = {}
    for i in range(n_vars):
        var_names[i+1] = f"x_{i}"

    i = 0
    clauses = []
    for c in prob_str.split("&"):
        c = c.strip(" ()").split("|")
        clause = []

        for lit in c:
            lit = lit.strip("() ")

            if lit[0] == "-":
                var = int(lit[1:])
                clause.append(f"\lnot {var_names[var]}")

            else:
                var = int(lit)
                clause.append(f"{var_names[var]}")

        clauses.append(" \lor ".join(clause))

    prob_str = ") \land (".join(clauses)
    return "("+prob_str+")"


def graph_3clr_prob_str_format(prob_str, n_nodes, n_colors):
    var_names = {}
    colors = ['r', 'g', 'b']
    count = 1
    for i in range(n_nodes):
        for j in range(n_colors):
            var_names[count] = f"{colors[j]}_{i}"
            count += 1

    i = 0
    clauses = []
    for c in prob_str.split("&"):
        c = c.strip(" ()").split("|")
        clause = []

        for lit in c:
            lit = lit.strip("() ")

            if lit[0] == "-":
                var = int(lit[1:])
                clause.append(f"\lnot {var_names[var]}")

            else:
                var = int(lit)
                clause.append(f"{var_names[var]}")

        clauses.append(" \lor ".join(clause))

    prob_str = ") \land (".join(clauses)

    return "("+prob_str+")"


def check_sat(problem):
    return problem.solve()


def parallel_sat(problems):
    with multiprocessing.Pool() as pool:
        result = pool.map_async(check_sat, problems)

    print(result.get())


def sequential_sat(problems):
    for p in problems:
        p.solve()


def test_runtime():
    problems = [SatProblem.random_grid_planning_problem(
        (5, 5), 10, False) for _ in range(32)]
    start = datetime.datetime.now()
    sequential_sat(problems)
    end = datetime.datetime.now()
    print((end-start).total_seconds())


if __name__ == "__main__":

    problem = SatProblem.random_graph_color_problem(10,3,3,sat=False)
    print(problem,'isSat: ', problem.isSat)
    problem_stats = {}
    
    problem_stats['clause_length_dist'] = {3: 0.9, 4: 0.1}
    problem_stats['ratios'] = [5]
    problem = SatProblem.random_formula_from_stats_sequential(stats=problem_stats, sat=False, num_vars_range=(10,100))
    
    print(problem, 'isSat: ',problem.isSat)
    