"""
Modified from
https://github.com/ibipul/coloring_SAT/blob/master/graph2cnf.py
"""

from io import StringIO
import numpy as np
import networkx as nx


def graph2cnf(n_nodes, edge_list, n_colors):

    cnf_strio = StringIO()
    colors = n_colors
    vertices = n_nodes
    edges = len(edge_list)

    print(
        "p cnf "
        + str(colors * vertices)
        + " "
        + str(vertices + vertices * colors *
              (colors - 1) // 2 + colors * edges),
        file=cnf_strio,
    )
    X = [[0] * (colors + 1) for _ in range(vertices + 1)]
    counter = 1
    i = 1
    while i < vertices + 1:
        j = 1
        while j < colors + 1:
            X[i][j] = counter
            # print counter
            counter = counter + 1
            j = j + 1
        i = i + 1

    # print X
    #print(f"No of Vertices:: {vertices}, No of edges:: {edges}")
    i = 1
    while i <= vertices:
        c = 1
        while c <= colors:
            print(X[i][c], end=" ", file=cnf_strio)
            if c == colors:
                print("0", file=cnf_strio)
            c = c + 1
        i = i + 1

    i = 1
    while i <= vertices:
        c = 1
        while c <= colors - 1:
            d = c + 1
            while d <= colors:
                print("-" + str(X[i][c]) + " -" +
                      str(X[i][d]) + " 0", file=cnf_strio)
                d = d + 1
            c = c + 1
        i = i + 1
    print('c edges', file=cnf_strio)
    for edge in edge_list:
        u = edge[0]
        v = edge[1]
        c = 1
        while c <= colors:
            print(
                "-" + str(X[int(u)][c]) + " -" + str(X[int(v)][c]) + " 0",
                file=cnf_strio,
            )
            c = c + 1

    return cnf_strio.getvalue()


def random_graph_color(n_nodes, n_colors, n_edges):
    """Generate a random graph coloring problem."""
    edge_list = []
    edge_hash = set()
    for i in range(n_edges):
        vs = np.random.choice(n_nodes, 2, replace=False) + 1
        edge = tuple(sorted(vs))
        if edge not in edge_hash:
            edge_hash.add(edge)
            edge_list.append([vs[0], vs[1]])

    cnf = graph2cnf(n_nodes, edge_list, n_colors).splitlines()

    return cnf


def random_erdos_graph_color(n_nodes, n_colors, p):
    """Generate a random graph coloring problem."""

    G = nx.erdos_renyi_graph(n_nodes, p)
    edgelist = [(int(line.split(" ")[0])+1, int(line.split(" ")[1])+1)
                for line in nx.generate_edgelist(G)]

    cnf = graph2cnf(n_nodes, edgelist, n_colors).splitlines()

    return cnf


def check_sat_ratio():
    from tqdm import tqdm
    from sat_problem import SatProblem
    is_sat = []
    for i in tqdm(range(100)):
        n_nodes = 20  # np.random.randint(50, 50)
        n_colors = 8
        edge_prob = 0.8
        print("making + solving")
        sat_problem = SatProblem.from_lines(
            random_erdos_graph_color(n_nodes, n_colors, edge_prob))
        #print("is sat", sat_problem.isSat)

        is_sat.append(sat_problem.isSat)
    print(np.mean(is_sat))



if __name__ == "__main__":
    import sys
    if True:
        sys.path.append("src")
    from sat_problem import SatProblem
    n_nodes = 4
    edge_list = [[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4]]
    n_colors = 3
    cnf = graph2cnf(n_nodes, edge_list, n_colors).splitlines()
    print(cnf)
    sat_problem = SatProblem.from_lines(cnf)
    print(sat_problem.toSequence())
    print(sat_problem.isSat)

    check_sat_ratio()

    random_erdos_graph_color(10, 3, 0.5)

