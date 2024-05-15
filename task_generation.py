import copy
from pathlib import Path
from collections import deque, defaultdict
from typing import List, Callable


import numpy as np
import argparse

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import math
from heapq import heappop, heappush
from typing import Tuple, List, Iterable, Callable, Type, Dict, Union, Optional
import numpy.typing as npt


from collections import deque
from matplotlib import pyplot as plt

from pathlib import Path
from tqdm import tqdm
from scipy.spatial import distance

import cppimport.import_hook
from grid_planner import grid_planner

np.random.seed(42)

class Map:
    """
    Square grid map class represents the environment for our moving agent.

    Attributes
    ----------
    _width : int
        The number of columns in the grid
        
    _height : int
        The number of rows in the grid
        
    _cells : ndarray[int, ndim=2]
        The binary matrix, that represents the grid. 0 - cell is traversable, 1 - cell is blocked
    """

    def __init__(self, cells: npt.NDArray):
        """
        Initialization of map by 2d array of cells.
        
        Parameters
        ----------
        cells : ndarray[int, ndim=2]
            The binary matrix, that represents the grid. 0 - cell is traversable, 1 - cell is blocked.
        """
        self._width = cells.shape[1]
        self._height = cells.shape[0]
        self._cells = cells


    def in_bounds(self, i: int, j: int) -> bool:
        """
        Check if the cell (i, j) is on a grid.
        
        Parameters
        ----------
            i : int
                Number of the cell row in grid
            j : int
                Number of the cell column in grid
        Returns
        ----------
             bool
                Is the cell inside grid.
        """
        return (0 <= j < self._width) and (0 <= i < self._height)
    

    def traversable(self, i: int, j: int) -> bool:
        """
        Check if the cell (i, j) is not an obstacle.
        
        Parameters
        ----------
            i : int
                Number of the cell row in grid
            j : int
                Number of the cell column in grid
        Returns
        ----------
             bool
                Is the cell traversable.
        """
        return not self._cells[i, j]


    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """
        Get a list of neighbouring cells as (i,j) tuples.
        It's assumed that grid is 4-connected (i.e. only moves into cardinal directions are allowed)
                
        Parameters
        ----------
            i : int
                Number of the cell row in grid
            j : int
                Number of the cell column in grid
        Returns
        ----------
            neighbors : List[Tuple[int, int]]
                List of neighbouring cells.
        """ 
        neighbors = []
        # delta = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        delta = [[0, 1], [1, 0], [0, -1], [-1, 0], 
                 [1, 1], [-1, 1], [1, -1], [-1, -1]]
        for d in delta:
            if self.in_bounds(i + d[0], j + d[1]) and self.traversable(i + d[0], j + d[1]):
                neighbors.append((i + d[0], j + d[1]))
        return neighbors

    
    def get_size(self) -> Tuple[int, int]:
        """
        Returns size of grid in cells.
        
        Returns
        ----------
            (height, widht) : Tuple[int, int]
                Number of rows and columns in grid
        """
        return (self._height, self._width)


class Node:

    def __init__(self, 
                 i: int, j: int, 
                 level: Union[float, int] = 0, 
                 parent: 'Node' = None):
        """
        Initialization of search node.
        
        Parameters
        ----------
        i, j : int, int
            Coordinates of corresponding grid element.
        g : float | int
            g-value of the node.
        h : float | int 
            h-value of the node // always 0 for Dijkstra.
        f : float | int 
            f-value of the node // always equal to g-value for Dijkstra.
        parent : Node 
            Pointer to the parent-node.
        """
        self.i = i
        self.j = j
        self.level = level    
        self.parent = parent

    
    def __eq__(self, other):
        """
        Estimating where the two search nodes are the same,
        which is needed to detect dublicates in the search tree.
        """
        
        return (self.i == other.i) and (self.j == other.j)

    def __hash__(self):
        """
        To implement CLOSED as set/dict of nodes we need Node to be hashable.
        """
        return hash((self.i, self.j))

    
    def __str__(self) -> str:
        return f"({self.i}, {self.j}) -> level={self.level}"
    

def compute_cost(x1, y1, x2, y2):
    return distance.euclidean([x1, y1], [x2, y2])

def fill_heuristic_values(finish_node: Node, task_map: Map, heuristic_func: Callable):
    height, width = task_map.get_size()
    x_arr = np.tile(np.arange(height), (width, 1)).T
    y_arr = np.tile(np.expand_dims(np.arange(width), axis=1), (1, height)).T

    h_arr = heuristic_func(x_arr, y_arr, finish_node.i, finish_node.j)
    h_arr[task_map._cells != 0] = 0
    return h_arr


def fill_true_dists_4_way(finish_node: Node, task_map: Map):
    layer = deque()
    layer.append(finish_node)

    node_levels = np.full(task_map.get_size(), np.inf)
    node_levels[finish_node.i, finish_node.j] = 0

    while len(layer) > 0:
        cur_layer_node = layer.popleft()

        for i, j in task_map.get_neighbors(cur_layer_node.i, cur_layer_node.j):
            child_node = Node(i=i, j=j, level=cur_layer_node.level+1)

            if node_levels[i][j] == np.inf and child_node != finish_node:
                layer.append(child_node)
                node_levels[i][j] = child_node.level
    return node_levels

def fill_true_dists_8_way(finish_node: Node, task_map: Map):
    layer = deque()
    layer.append(finish_node)

    # node_levels = np.zeros(task_map.get_size())
    node_levels = np.full(task_map.get_size(), np.inf)
    node_levels[finish_node.i, finish_node.j] = 0

    while len(layer) > 0:
        cur_layer_node = layer.popleft()

        for i, j in task_map.get_neighbors(cur_layer_node.i, cur_layer_node.j):
            step_length = compute_cost(cur_layer_node.i, cur_layer_node.j, i, j)
            child_node_level = cur_layer_node.level + step_length

            child_node = Node(i=i, j=j, level=child_node_level)

            if child_node != finish_node:
                if node_levels[i][j] == np.inf or child_node_level < node_levels[i][j]:
                    layer.append(child_node)
                    node_levels[i][j] = child_node.level

    return node_levels

def invert_cells(map: Map):
    # ones_ids = map._cells == 1
    # zeros_ids = map._cells == 0
    new_cells = map._cells.copy()
    new_cells[map._cells == 0] = 1
    new_cells[map._cells == 1] = 0

    return Map(new_cells)

def fill_true_dists_8_way_cpp(finish_node: Node, task_map: Map):
    task_map = invert_cells(task_map)
    planner = grid_planner(task_map._cells.tolist())
    goal = (finish_node.i, finish_node.j)

    true_dists = planner.fill_true_dists_8_way(goal)
    true_dists = np.array(true_dists)
    return true_dists


def calc_cf_values(true_dists, h_values, goal_node):
    true_dists[true_dists == 0.0] = np.inf
    cf_values = h_values / true_dists
    cf_values[goal_node.i][goal_node.j] = 1
    return cf_values

def fill_cf_values(finish_node: Node, task_map: Map, heuristic_func: Callable):
    true_dists = fill_true_dists_8_way_cpp(finish_node, task_map)
    true_dists[true_dists == 0.0] = np.inf # to ignore zero division
    
    h_values = fill_heuristic_values(finish_node, task_map, heuristic_func)

    cf_values = h_values / true_dists
    cf_values[finish_node.i][finish_node.j] = 1

    return cf_values

euclidean_distance = lambda x_arr, y_arr, goal_x, goal_y: np.sqrt((x_arr - goal_x) ** 2 + (y_arr - goal_y) ** 2)
manhattan_distance = lambda x_arr, y_arr, goal_x, goal_y: np.abs(x_arr - goal_x) + np.abs(y_arr - goal_y)

def diagonal_distance(x_arr, y_arr, goal_x, goal_y): 
    dx = np.abs(x_arr - goal_x)
    dy = np.abs(y_arr - goal_y)
    return np.abs(dx - dy) + np.sqrt(2) * np.minimum(dx, dy)

def extract_node_pos(map):
    pos = np.where(map == 1)
    return (pos[0][0], pos[1][0])

def suggest_goals(map, n=10):
    empty_cells = np.array(list(zip(*np.where(map == 1))))
    goal_ids = np.random.choice(np.arange(len(empty_cells)), size=n, replace=False)
    goal_cells = empty_cells[goal_ids]

    # goal_maps = np.full((n, *map.shape), 0)
    # for i, cell in enumerate(goal_cells):
    #     goal_maps[i][cell[0], cell[1]] = 1

    return goal_cells

def suggest_goal(map):
    empty_cells = np.array(list(zip(*np.where(map == 1))))
    goal_id = np.random.choice(np.arange(len(empty_cells)))
    goal_cell = empty_cells[goal_id]

    return goal_cell

def suggest_start(map, hardness_map, hardness_threshold=1.05):
    hard_cells = np.array(list(zip(*np.where((hardness_map > hardness_threshold) & (map == 1)))))

    if len(hard_cells) == 0: # NO HARD ROUTES DETECTED or NO PATHS EXISTS
        return None

    cell_id = np.random.choice(np.arange(len(hard_cells)))
    return hard_cells[cell_id]

def create_tasks(map):
    map_example = Map(map)
    map_example = invert_cells(map_example)

    goals = []
    starts = []
    cfs = []

    n_tasks = 0
    while n_tasks < 10:
        goal = suggest_goal(map)
        goal_node = Node(goal[0], goal[1])
        
        heuristic_values = fill_heuristic_values(goal_node, map_example, diagonal_distance)
        true_dists = fill_true_dists_8_way_cpp(goal_node, map_example)
        cf_values = calc_cf_values(true_dists, heuristic_values, goal_node)
        
        # Replace unreachable nodes with zeros
        map_size = map_example.get_size()
        max_path_length = map_size[0] * map_size[1]
        unreachable_nodes_mask = (true_dists > max_path_length) | (map == 0)
        # Creating start node
        route_hardness_map = 1 / cf_values
        # Filling walls and unreachable nodes with zeros
        true_dists[unreachable_nodes_mask] = 0
        heuristic_values[unreachable_nodes_mask] = 0
        cf_values[unreachable_nodes_mask] = 0
        route_hardness_map[unreachable_nodes_mask] = 0
        
        
        start = suggest_start(map, route_hardness_map)
        if start is not None:
            n_tasks += 1

            goals.append(goal)
            starts.append(start)
            cfs.append(cf_values)
    
    return starts, goals, cfs

def generated_tasks(maps):
    tasks = {
        "cfs": [],
        "starts": [],
        "goals": [],
        "maps": [],
    }

    for map in tqdm(maps):
        starts, goals, cfs = create_tasks(map)
        # if starts is None: # map contains impossible goal
        #     continue
        # print("->", len(cfs))
        
        tasks["maps"].append(map)
        tasks["cfs"].append(cfs)
        tasks["starts"].append(starts)
        tasks["goals"].append(goals)
    
    return tasks

def merge_predictions(dir):
    merged_tasks = {
        "cfs": defaultdict(list),
        "starts": defaultdict(list),
        "goals": defaultdict(list),
        "maps": defaultdict(list),
    }
    for t in ["train", "test", "valid"]:
        load_dir = dir / t
        for path in tqdm(load_dir.glob('*')):
            batch = np.load(path)
            for k in merged_tasks.keys():
                for task in batch[k]:
                    merged_tasks[k][t].append(task)
    return merged_tasks
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps_path', type=str)
    args = parser.parse_args()

    load_path = Path(args.maps_path)
    
    
    # Generating tasks

    maps = np.load(load_path)
    batch_size = 100
    for data_type in maps.keys():
        print(f"Processing {data_type}")

        n_batches = (len(maps[data_type]) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(n_batches)):
            
            batch = maps[data_type][batch_size*batch_idx: batch_size*(batch_idx+1)]
            
            tasks_batch = generated_tasks(batch)
        
            save_dir = load_path.parent / data_type 
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir/ f"{load_path.stem}_batch_{batch_idx}.npz"
            np.savez(save_path, 
                     starts=tasks_batch["starts"],
                     goals=tasks_batch["goals"],
                     maps=tasks_batch["maps"],
                     cfs=tasks_batch["cfs"],
                     )
            

    # Merging predictions
    data = merge_predictions(load_path.parent)

    # Saving merged
    for k in data.keys():
        np.savez(load_path.parent / f"{load_path.stem}_{k}.npz", 
                 train=data[k]["train"], 
                 valid=data[k]["valid"], 
                 test=data[k]["test"])