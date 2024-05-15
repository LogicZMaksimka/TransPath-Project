import argparse
import numpy as np
from tqdm import tqdm
import cppimport.import_hook
from grid_planner import grid_planner

def generate_tasks(num_tasks, grid_size):
    tasks = []
    while len(tasks) < num_tasks:
        coords = np.random.randint(0, grid_size, 4)
        if abs(coords[0] - coords[2]) + abs(coords[1] - coords[3]) > grid_size:
            tasks.append({'start': (coords[0], coords[1]), 'goal': (coords[2], coords[3])})
    return tasks

def get_focal_values(map: np.array, start, goal):
    results = []
    planner = grid_planner(map.tolist())
    for task_start, task_goal in zip(start, goal):
        results.append(planner.find_heatmap(task_start, task_goal))
    return np.stack(results)[:, None, :, :]

def proc_file(filenames: list[str], coef=1_000_000):
    
    new_filename = filenames[0][:-4] + '_focal.npz'
    
    all_focals = {}
    all_maps = np.load(filenames[0])
    all_starts = np.load(filenames[1])
    all_goals = np.load(filenames[2])
    
    for data_split in ["train", "valid", "test"]:
        focals = []
        maps, starts, goals = all_maps[data_split], all_starts[data_split], all_goals[data_split]
        
        maps = maps * coef
        
        for map, start, goal in tqdm(zip(maps, starts, goals)):
            focal = get_focal_values(map, start, goal)
            focals.append(focal)
        
        all_focals[data_split] = np.stack(focals)[:, None, :, :]
    
    
    np.savez(new_filename, train=all_focals["train"], valid=all_focals["valid"], test=all_focals["test"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', nargs='+', type=str, default=['./val.npz', './train.npz', './test.npz'])
    args = parser.parse_args()
    args.filename.append(args.filename[0][:-4] + "_starts.npz")
    args.filename.append(args.filename[0][:-4] + "_goals.npz")
    proc_file(args.filename)
        
    
if __name__ == '__main__':
    main()
