import matplotlib.pyplot as plt
import numpy as np


def load_pddl_problem(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line != '']
    return lines


def loc_str_to_coords(loc_str):
    splt = loc_str.split('-')
    x = int(splt[1].strip(" ()"))
    y = int(splt[2].strip(" ()"))
    return x, y


def make_grid(lines):
    locations = []
    for line in lines:
        if line.endswith('- location'):
            locations.append(loc_str_to_coords(line))

            print(line)

    max_x = max([x for x, y in locations])
    max_y = max([y for x, y in locations])

    grid = np.zeros((max_x+1, max_y+1, 3))
    print(grid.shape)

    for line in lines:
        if line.startswith('(clear'):
            loc = loc_str_to_coords(line.split(' ')[1])
            grid[loc] = [1, 1, 1]
        if line.startswith("(at player-1"):
            loc = loc_str_to_coords(line.split(' ')[2])
            grid[loc] = [0, 0, 1]
        if line.startswith("(is-goal"):
            loc = loc_str_to_coords(line.split(' ')[1])
            grid[loc] = [0, 1, 0]

    return grid


if __name__ == "__main__":
    base_path = '/home/plymper/unsat-detection/src/generate_data/planning_domains'
    lines = load_pddl_problem(base_path+'/problem0.pddl')
    grid = make_grid(lines)

    plt.imshow(grid)
    plt.savefig(base_path+'/grid.png')

    pass
