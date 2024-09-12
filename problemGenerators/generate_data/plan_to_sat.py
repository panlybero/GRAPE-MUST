import sys
if True:
    sys.path.append("src")
from pyperplan.planner import _parse, _ground
from pyperplan.pddl.parser import Parser
from pyperplan.search.sat import get_plan_formula, sat_solve, _extract_plan
from pyperplan.search import minisat
from problemGenerators.generate_data.simplify import minisat_simplify
import subprocess
import time
import problemGenerators.generate_data.planning_domains.CnfWriter as CnfWriter
from pysat.formula import CNF
import numpy as np
import matplotlib.pyplot as plt
import problemGenerators.generate_data.planning_domains.renderer as renderer
import os
import problemGenerators.generate_data.planning_domains.constants as constants
from pysat.solvers import Glucose3
import copy
import random


def dist(p1, p2):
    return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])


def random_grid(size: tuple[int, int], n_walls: int):
    grid = np.ones((size[0], size[1], 3))

    sample_coordinates_no_replacement = np.random.choice(
        size[0]*size[1], n_walls, replace=False)
    sample_coordinates = np.unravel_index(
        sample_coordinates_no_replacement, size)
    grid[sample_coordinates] = 0

    goal_position = np.random.randint(size[0]*size[1])
    goal_position = np.unravel_index(goal_position, size)
    grid[goal_position] = [0, 1, 0]

    start_position = np.random.randint(size[0]*size[1])
    start_position = np.unravel_index(start_position, size)
    while dist(start_position, goal_position) < 5:
        start_position = np.random.randint(size[0]*size[1])
        start_position = np.unravel_index(start_position, size)

    grid[start_position] = [0, 0, 1]

    return grid


def get_connectivity_line(loc1, loc2):
    if abs(loc1[0]-loc2[0]) > 1 or abs(loc1[1]-loc2[1]) > 1:
        raise ValueError("Locations are not adjacent")

    if loc1[0] == loc2[0]:
        if loc1[1] > loc2[1]:
            return f"(move-dir-left loc-{loc1[0]}-{loc1[1]} loc-{loc2[0]}-{loc2[1]})\n"
        else:
            return f"(move-dir-right loc-{loc1[0]}-{loc1[1]} loc-{loc2[0]}-{loc2[1]})\n"
    else:
        if loc1[0] > loc2[0]:
            return f"(move-dir-up loc-{loc1[0]}-{loc1[1]} loc-{loc2[0]}-{loc2[1]})\n"
        else:
            return f"(move-dir-down loc-{loc1[0]}-{loc1[1]} loc-{loc2[0]}-{loc2[1]})\n"


def get_adjacent_points(loc, grid):
    adjacent_points = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if i != 0 and j != 0:
                continue
            if loc[0]+i < 0 or loc[0]+i >= grid.shape[0]:
                continue
            if loc[1]+j < 0 or loc[1]+j >= grid.shape[1]:
                continue
            if grid[loc[0]+i, loc[1]+j].sum() == 0:
                continue
            adjacent_points.append((loc[0]+i, loc[1]+j))
    return adjacent_points


def grid_to_pddl_problem(problem_template: str, grid: np.ndarray):
    """
    Convert a grid to a PDDL problem
    """
    problem = copy.deepcopy(problem_template)
    clear_coords = np.where(grid.sum(-1) == 3)
    clear_coords = list(zip(clear_coords[0], clear_coords[1]))
    clear_set = set(clear_coords)
    walkable_set = copy.deepcopy(clear_set)
    goal_coords = np.where(grid.sum(-1) == 1)

    goal_coords = np.where(np.bitwise_and(
        grid[:, :, 1] == 1, grid.sum(-1) == 1))
    goal_coords = list(zip(goal_coords[0], goal_coords[1]))[0]
    walkable_set.add(goal_coords)

    start_coords = np.where(np.bitwise_and(
        grid[:, :, 2] == 1, grid.sum(-1) == 1))
    start_coords = list(zip(start_coords[0], start_coords[1]))[0]
    walkable_set.add(start_coords)
    locations_str = f""
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            locations_str += f"loc-{i}-{j} - location\n"

    clear_str = f""
    for i, j in clear_coords:
        clear_str += f"(clear loc-{i}-{j})\n"
    clear_str += f"(clear loc-{goal_coords[0]}-{goal_coords[1]})\n"

    goal_str = f"(is-goal loc-{goal_coords[0]}-{goal_coords[1]})\n"

    start_str = f"(at player-1 loc-{start_coords[0]}-{start_coords[1]})\n"
    final_goal_str = f"(:goal (at player-1 loc-{goal_coords[0]}-{goal_coords[1]}))\n"

    connectivity_str = f""
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for loc in get_adjacent_points((i, j), grid):
                if loc in walkable_set and (i, j) in walkable_set:

                    connectivity_str += get_connectivity_line((i, j), loc)
                    connectivity_str += get_connectivity_line(loc, (i, j))

    problem = problem.replace("#add_locations", locations_str)
    problem = problem.replace("#add_clear_locations", clear_str)
    problem = problem.replace("#add_goal_location", goal_str)
    problem = problem.replace("#add_player_location", start_str)
    problem = problem.replace("#add_location_connectivity", connectivity_str)
    problem = problem.replace("#add_goal_str", final_goal_str)

    return problem


def random_logistics_pddl(n_planes: int = 1, n_airports: int = 2, n_locations: int = 2, n_cities: int = 2, n_trucks: int = 2, n_packages: int = 5):
    # print("generating")
    problem_template = constants.logistics_problem_template
    problem = copy.deepcopy(problem_template)
    objects_str = f""
    airplane_str = ""
    airport_str = ""
    locations_str = ""
    cities_str = ""
    trucks_str = ""
    packages_str = ""
    for i in range(n_planes):
        airplane_str += f"plane{i} "
    airplane_str += "- airplane\n"
    for i in range(n_airports):
        airport_str += f"airport{i} "
    airport_str += "- airport\n"

    for i in range(n_locations):
        locations_str += f"location{i} "
    locations_str += "- location\n"
    for i in range(n_cities):
        cities_str += f"city{i} "
    cities_str += "- city\n"
    for i in range(n_trucks):
        trucks_str += f"truck{i} "
    trucks_str += "- truck\n"
    for i in range(n_packages):
        packages_str += f"package{i} "
    packages_str += "- package\n"

    objects_str = airplane_str + airport_str + locations_str + \
        cities_str + trucks_str + packages_str

    problem = problem.replace("#add_objects", objects_str)
    #print("objects added")
    # Making initial state
    # randomly assign locations to cities
    location_city_map = {}
    for i in range(n_locations):
        location_city_map[f"location{i}"] = f"city{random.randint(0,n_cities-1)}"

    # randomly assign packages to locations
    package_location_map = {}
    for i in range(n_packages):
        package_location_map[f"package{i}"] = f"location{random.randint(0,n_locations-1)}"

    # randomly assign trucks to locations
    truck_location_map = {}
    for i in range(n_trucks):
        truck_location_map[f"truck{i}"] = f"location{random.randint(0,n_locations-1)}"
    # randomly assign planes to airports
    plane_airport_map = {}
    for i in range(n_planes):
        plane_airport_map[f"plane{i}"] = f"airport{random.randint(0,n_airports-1)}"

    # randomly assign airports to cities
    airport_city_map = {}
    for i in range(n_airports):
        airport_city_map[f"airport{i}"] = f"city{random.randint(0,n_cities-1)}"

    init_state_map = {}
    init_state_map.update(location_city_map)
    init_state_map.update(package_location_map)
    init_state_map.update(truck_location_map)
    init_state_map.update(plane_airport_map)
    init_state_map.update(airport_city_map)

    initial_state_str = ""
    for k, v in init_state_map.items():
        if k.startswith("location"):
            initial_state_str += f"(in-city {k} {v})\n"
        if k.startswith("airport"):
            initial_state_str += f"(in-city {k} {v})\n"
        if k.startswith("package"):
            initial_state_str += f"(at {k} {v})\n"
        if k.startswith("truck"):
            initial_state_str += f"(at {k} {v})\n"
        if k.startswith("plane"):
            initial_state_str += f"(at {k} {v})\n"

    problem = problem.replace("#add_init", initial_state_str)
    #print("initial state added")
    # Making goal state
    goal_state_str = "(and "

    # randomly assign packages to locations
    package_location_map = {}
    for i in range(n_packages):
        package_location_map[f"package{i}"] = f"location{random.randint(0,n_locations-1)}"
    for k, v in package_location_map.items():
        goal_state_str += f"(at {k} {v})\n"

    goal_state_str += ")"
    problem = problem.replace("#add_goal", goal_state_str)
    #print("goal state added")
    return problem


def format_cnfstr_to_dimacs(cnf_str: str):
    """
    Convert a CNF object to a DIMACS string
    """

    cnf = CNF(from_string=cnf_str)
    n_vars = cnf.nv
    result = f"p cnf {n_vars} {len(cnf.clauses)}\n"+cnf_str

    return result


def random_grid_problem(grid_size: tuple[int, int], n_walls: int, save_grid=False):
    '''
    Generates a random grid navigation planning problem. 

    '''

    n_walls = n_walls
    grid = random_grid(grid_size, n_walls)
    if save_grid:
        plt.imsave(f"grid.png", grid)
    #print("Making ", grid_size, " grid with ", n_walls, " walls")
    problem = grid_to_pddl_problem(constants.grid_problem_template, grid)
    return problem


def pddl_to_cnf(domain_str: str, problem_str: str, trajecotry_length: int):
    parser = Parser("", "")
    parser.domInput = domain_str
    parser.probInput = problem_str
    domain = parser.parse_domain(read_from_file=False)
    problem = parser.parse_problem(domain, read_from_file=False)

    task = _ground(problem, remove_irrelevant_operators=False)

    formula = get_plan_formula(task, trajecotry_length)
    f = copy.deepcopy(formula)
    # print(formula)
    #print("planner says", minisat.solve(f))

    writer = CnfWriter.CnfWriter()
    vars_to_numbers = writer.write(formula)
    # print(vars_to_numbers)
    cnf_str = writer.get_cnf_str()
    dimacs_str = format_cnfstr_to_dimacs(cnf_str)
    return dimacs_str  # ,vars_to_numbers


def _random_grid_planning_cnf(grid_size: tuple[int, int], n_walls: int, trajectory_length: int):
    """
    Generate a random grid navigation planning problem and convert it to a CNF object
    """
    # get current file path
    domain_str = constants.grid_domain_file
    problem = random_grid_problem(grid_size, n_walls)
    #open("problem.pddl", "w").write(problem)
    cnf, var_meanings = pddl_to_cnf(
        domain_str, problem, trajectory_length)

    return cnf, var_meanings


def M_pddl_to_CNF(domain_str: str, problem_str: str, trajecotry_length: int):

    indmnfilename = f'/tmp/tmp_cnf_dmn_in_{np.random.randint(10000)}-{np.random.randint(100000)}.cnf'
    inprbfilename = f'/tmp/tmp_cnf_prb_in_{np.random.randint(10000)}-{np.random.randint(100000)}.cnf'

    outfilename = f"/tmp/tmp_cnf_out_{np.random.randint(10000)}-{np.random.randint(100000)}"

    open(indmnfilename, "w").write(domain_str)
    open(inprbfilename, "w").write(problem_str)

    cmd = f"/home/plymper/unsat-detection/solvers/M {indmnfilename} {inprbfilename} -P 0 -1 -O -S 1 -F {trajecotry_length-1} -T {trajecotry_length} -b {outfilename}"

    try:
        out = subprocess.check_output(
            cmd.split(), timeout=1, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise (e)
        pass

    lines = out.splitlines()
    fnames = [l for l in lines if l.startswith(b"Writing")]
    fnames = [f.decode().split()[1] for f in fnames]
    try:

        cmd = f"rm {indmnfilename} {inprbfilename}"

        subprocess.run(cmd, shell=True)

        cnf = open(fnames[-1]).read()

        subprocess.run(f"rm {fnames[-1]}", shell=True)
    except Exception as e:
        raise (e)
    return cnf


def random_grid_planning_cnf(grid_size: tuple[int, int], trajectory_length: int, save_grid=False):
    domain_str = constants.grid_domain_file
    area = grid_size[0]*grid_size[1]
    done = False
    while not done:
        try:
            a = np.random.randint(area//1.5, area//1.3)
            problem = random_grid_problem(grid_size, a, save_grid=save_grid)
            #cnf = M_pddl_to_CNF(domain_str, problem, trajectory_length)
            cnf = pddl_to_cnf(domain_str, problem, trajectory_length)
            # print("made")
            cnf = CNF(from_string=cnf)
            clauses = minisat_simplify(cnf.clauses)
            # print("simplified")
            if cnf.nv + len(cnf.clauses) > 20000:
                #print(cnf.nv, len(cnf.clauses))
                raise Exception("too big")
            if check_trivial_unsat(clauses):
                raise Exception("trivial unsat")
            cnf = CNF(from_clauses=clauses)
            done = True
            # print("done")

        except Exception as e:
            # print(e)

            pass

    return cnf.nv, cnf.clauses


def check_trivial_unsat(clauses):
    forced_vars = {}
    for clause in clauses:
        if len(clause) == 1:
            var = abs(clause[0])
            val = clause[0] > 0
            if var in forced_vars:
                if forced_vars[var] != val:
                    return True
            else:
                forced_vars[var] = val
    return False


def random_logistics_cnf(n_planes: int = 1, n_airports: int = 2, n_locations: int = 2, n_cities: int = 2, n_trucks: int = 2, n_packages: int = 5, trajectory_length: int = 5):
    domain_str = constants.logistics_domain_file

    done = False
    while not done:
        try:
            problem = random_logistics_pddl(
                n_planes, n_airports, n_locations, n_cities, n_trucks, n_packages)
            cnf = M_pddl_to_CNF(domain_str, problem, trajectory_length)

            cnf = CNF(from_string=cnf)

            if cnf.nv + len(cnf.clauses) > 15000:
                #print(cnf.nv, len(cnf.clauses))
                raise Exception("too big")
            if check_trivial_unsat(cnf.clauses):
                raise Exception("trivial unsat")

            done = True
            # print("done")

        except Exception as e:
            # print(e)
            #raise (e)
            # fail silently and just retry
            pass

    return cnf.nv, cnf.clauses


if __name__ == '__main__':
    import copy
    import matplotlib.pyplot as plt
    sys.path.append("/home/plymper/unsat-detection/solvers/PyMiniSolvers/")
    sys.path.append("/home/plymper/unsat-detection/src/")
    from minisolvers import MinisatSolver
    from simplify import simplify_clauses, minisat_simplify
    from solver import marco_external_solver
    from sat_problem import SatProblem

    nv, clauses = random_logistics_cnf(n_packages=10, trajectory_length=5)
    #nv, clauses = random_grid_planning_cnf((8, 8), 3, save_grid=True)
    print(nv, len(clauses))
    prob = SatProblem(nv, clauses)

    if prob.isSat:
        print("sat")
        exit()

    m = marco_external_solver(prob, 3, {})
    min_size = 10000
    print(m['core'])
    if m['core'] == 2:
        open("prob.cnf", 'w').write(prob.to_dimacs())

    print(len(m['all']))
    print(m['all'][0]['time'])

    print(m['all'][-1]['time'])
    min_k = None  # m['all'][0]
    for k in m['all']:
        if len(k['result']) > 2 and len(k['result']) < min_size:
            min_size = len(k['result'])
            min_k = k

    print(min_k['size'], min_k['time'])

    pass
