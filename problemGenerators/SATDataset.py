from typing import Union
import os
import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset, Data
from torch.utils.data import Dataset as TorchDataset
from problemGenerators.sat_problem import SatProblem
import pickle as pkl
import tqdm
import logging


class SATDataset(Dataset):
    def __init__(self, graphs: list[Data], sat_problems: list[SatProblem]):
        super().__init__()
        if type(graphs) is not list:
            raise TypeError(
                "graphs must be a list of Data objects. Did you mean to use the load method?")
        self.graphs = graphs

        self.sat_problems = {sat_problems[i].__hash__(
        ): sat_problems[i] for i in range(len(sat_problems))}

        self.max_var = max([g.n_vars+1 for g in self.sat_problems.values()])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: Union[int, np.integer]) -> Union['Dataset', Data]:
        return self.graphs[idx]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    def recalculate_spectral_emb(self, spectral_dim):
        for i in range(len(self.graphs)):
            self.graphs[i] = self.sat_problems[self.graphs[i].sat_problem_hash].toHeteroGraph(
                spectral_dim=spectral_dim)

    @classmethod
    def from_dimacs_dir(cls, dir, data_type="graph", spectral_dim=10):
        """Read a directory of CNF files and create SATProblem instances for each one."""
        graphs = []
        sat_problems = []
        for filename in tqdm.tqdm(os.listdir(dir)):
            if filename.endswith(".dimacs") or filename.endswith(".cnf"):
                sat_problem = SatProblem.from_file(os.path.join(dir, filename))
                graphs.append(sat_problem.toHeteroGraph(
                    spectral_dim=spectral_dim))
                sat_problems.append(sat_problem)
        return cls(graphs, sat_problems)

    @classmethod
    def from_file(cls, filename, spectral_dim=None):
        saved_dataset = None
        with open(filename, 'rb') as f:
            saved_dataset = pkl.load(f)
        return saved_dataset

    def describe(self):
        n_formulas = len(self.graphs)
        n_sat_formulas = sum([1 for g in self.graphs if g.isSat])

        print(f"Number of formulas: {n_formulas}")
        print(f"Number of SAT formulas: {n_sat_formulas}")


class SATBenchmarkDataset(Dataset):
    def __init__(self, path, node_dim, args):
        '''Recursively walk in path and load all compressed cnf files'''
        self.problems = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".cnf.bz2"):
                    self.problems.append(
                        SatProblem.from_bz2(os.path.join(root, file)))
                    self.problems[-1].fname = file

        self.args = args
        self.node_dim = node_dim

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: Union[int, np.integer]) -> Union['Dataset', Data]:

        return self.problems[idx].toHeteroGraph(spectral_dim=self.node_dim, use_spectral_emb=self.args.use_spectral_emb)


class SATBenchmarkDatasetOnDisk(Dataset):
    def __init__(self, path, node_dim, args):
        '''Recursively walk in path and load all compressed cnf files'''
        self.fnames = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if 'reduced' in file:
                    continue
                if file.endswith(".cnf.bz2"):
                    self.fnames.append(os.path.join(root, file))
        self.args = args
        self.node_dim = node_dim

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: Union[int, np.integer]) -> Union['Dataset', Data]:

        problem = SatProblem.from_bz2(self.fnames[idx])
        problem.fname = self.fnames[idx].split("/")[-1]

        return problem.toHeteroGraph(spectral_dim=self.node_dim, use_spectral_emb=self.args.use_spectral_emb)


class SATDatasetOnline(Dataset):
    def __init__(self, min_n: int, max_n: int, spectral_dim: int = 10, formulas_per_epoch=2000, formula_size_increment=0, generate_only: str = None, args=None):
        super().__init__()

        assert max_n >= min_n, "max_n must be greater than min_n"
       # assert spectral_dim<=2*max_n, f"{spectral_dim=} must be greater than 2*max_n={2*max_n} to ensure enough dimensions for the spectral embedding"
        assert generate_only in [
            None, "sat", "unsat"], "generate_only must be either None, 'sat' or 'unsat'"
        logger = logging.getLogger("experiment_logger.dataset")

        if generate_only in ['sat', 'unsat']:
            logger.warn(f"Generating only {generate_only} formulas")

        self.min_n = min_n
        self.max_n = max_n
        self.current_max_n = min_n
        self.spectral_dim = spectral_dim
        self.sat_problems = {}
        self.formulas_per_epoch = formulas_per_epoch
        self.formula_size_increment = formula_size_increment
        self.generate_only = generate_only
        self.gc_edge_prob = args.gc_edge_prob

        self.leftover = None

        self.args = args
        if args.problem_type == "from_stats":
            import json
            self.stats = json.load(open(args.stats_path, "r"))

    def __len__(self):
        # self.batch_per_epoch*self.batch_size #dynamic dataset
        return self.formulas_per_epoch

    def increment_max_n(self):

        self.current_max_n = min(
            self.current_max_n + self.formula_size_increment, self.max_n)

    def __getitem__(self, idx: Union[int, np.integer]) -> Union['Dataset', Data]:

        if self.formula_size_increment == 0:
            n = int(self.current_max_n)
        else:
            n = self.max_n

        generate_only = None if self.generate_only is None else self.generate_only == "sat"
        if self.args.problem_type == "sr":
            problem = SatProblem.random_problem(
                self.min_n, n, sat=generate_only, return_both=False, p_geo=self.args.sr_p_geo)
        if self.args.problem_type == "graph_coloring":

            problem = SatProblem.random_graph_color_problem(
                n, self.args.min_n_colors, self.args.max_n_colors, sat=generate_only, edge_prob=self.gc_edge_prob,)

        if self.args.problem_type == "random_ksat":
            problem = SatProblem.random_ksat_problem(
                n, self.args.ksat_k, self.args.ksat_ratio, sat=generate_only)

        if self.args.problem_type == "grid_planning":
            problem = SatProblem.random_grid_planning_problem(
                self.args.grid_planning_size, traj_length=self.args.grid_planning_traj, sat=generate_only)

        if self.args.problem_type == "logistics_planning":

            problem = SatProblem.random_logistics_planning_problem(
                n_packages=self.args.logistics_n_packages, trajectory_length=5, sat=generate_only)

        if self.args.problem_type == "from_stats":

            problem = SatProblem.random_formula_from_stats_sequential(
                self.stats, sat=generate_only)

        return problem.toHeteroGraph(spectral_dim=self.spectral_dim, use_spectral_emb=self.args.use_spectral_emb)


class SATTextDataset(Dataset):
    def __init__(self, formulas: list[str], sat_problems: list[SatProblem]):
        super().__init__()
        if type(formulas) is not list:
            raise TypeError(
                "graphs must be a list of Data objects. Did you mean to use the load method?")
        self.formulas = formulas

        self.sat_problems = {sat_problems[i].__hash__(
        ): sat_problems[i] for i in range(len(sat_problems))}

        self.max_var = max([g.n_vars+1 for g in self.sat_problems.values()])

        self.isSat = [i.isSat for i in self.sat_problems.values()]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: Union[int, np.integer]) -> Union['TorchDataset', str]:
        return self.formulas[idx], self.isSat[idx]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    @classmethod
    def from_dimacs_dir(cls, dir: str):
        """Read a directory of CNF files and create SATProblem instances for each one."""
        formulas = []

        sat_problems = []
        for filename in os.listdir(dir):
            if filename.endswith(".dimacs") or filename.endswith(".cnf"):
                sat_problem = SatProblem.from_file(os.path.join(dir, filename))
                formulas.append(sat_problem.toSequence())
                sat_problems.append(sat_problem)
        return cls(formulas, sat_problems)

    @classmethod
    def from_file(cls, filename: str):
        saved_dataset = None
        with open(filename, 'rb') as f:
            saved_dataset = pkl.load(f)
        return saved_dataset

    def describe(self):
        n_formulas = len(self.graphs)
        n_sat_formulas = sum([1 for g in self.graphs if g.isSat])

        print(f"Number of formulas: {n_formulas}")
        print(f"Number of SAT formulas: {n_sat_formulas}")


if __name__ == "__main__":
    # test save of dataset
    # dataset = SATDataset.from_dimacs_dir("/home/plymper/unsat-detection/neurosat/dimacs/train/sr5/grp1")
    # dataset.save("data/sr5-grp1.pkl")

    # dataset.describe()
    from args import get_default_args
    import datetime
    args = get_default_args()
    args.gc_edge_prob = 0.8
    args.problem_type = "graph_coloring"
    args.min_n_colors = 7
    args.max_n_colors = 7
    args.use_spectral_emb = False

    dataset = SATDatasetOnline(30, 30, 10, 2000, 0, 'unsat', args)
    now = datetime.datetime.now()
    for i in range(64):
        data = dataset[i]
        print("got in", (datetime.datetime.now()-now).total_seconds(), "s")

    print("total batch time", (datetime.datetime.now()-now).total_seconds(), "s")
