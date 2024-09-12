from problemGenerators.sat_problem import SatProblem
from problemGenerators.SATDataset import SATDatasetOnline
from types import SimpleNamespace

if __name__=='__main__':

    problem = SatProblem.random_graph_color_problem(10,3,3,sat=False)
    print(problem,'isSat: ', problem.isSat)
    problem_stats = {}
    
    problem_stats['clause_length_dist'] = {3: 0.9, 4: 0.1}
    problem_stats['ratios'] = [5]
    problem = SatProblem.random_formula_from_stats_sequential(stats=problem_stats, sat=False, num_vars_range=(10,100))
    
    print(problem, 'isSat: ',problem.isSat)
    
    problem = SatProblem.random_problem(10,20,sat=False)
    print(problem, 'isSat: ',problem.isSat)

    args = SimpleNamespace()
    args.problem_type = 'graph_coloring'
    args.min_n_colors=3
    args.max_n_colors=3 
    args.gc_edge_prob = 0.8
    args.use_spectral_emb = False

    dataset = SATDatasetOnline(10,20,generate_only='unsat',args=args)

    for i in range(10):
        print(dataset[i])