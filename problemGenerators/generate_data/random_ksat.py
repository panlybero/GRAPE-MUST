import random
import numpy as np


def random_ksat(n_vars: int, ratio: float = 5, k: int = 3):
    '''
    Generate random KSAT formulas in the dimacs format. 
    The number of clauses is n_vars*ratio.
    The number of variables is n_vars.
    The number of literals per clause is k.

    We can generate mostly UNSAT problems by setting the ratio above the phase transition for the corresponding k.

    '''

    n_clauses = int(n_vars*ratio)
    clauses = ["p cnf {} {}".format(n_vars, n_clauses)]
    for i in range(n_clauses):
        clause = []
        literals = np.random.choice(n_vars, k, replace=False)+1
        # random flip
        literals = np.random.choice([-1, 1], k, replace=True)*literals
        clause.extend(literals.tolist())

        # for j in range(k):
        #     literal = random.randint(1, n_vars)
        #     if random.random() < 0.5:
        #         literal = -literal
        #     clause.append(literal)
        clause.append(0)
        clause = ' '.join(map(str, clause))
        clauses.append(clause)

    return clauses
