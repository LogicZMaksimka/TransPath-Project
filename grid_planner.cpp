// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>
#include <queue>
#include <cmath>
#include <set>
#include <map>
#include <list>
#include <iostream>
#define INF 1000000000
namespace py = pybind11;

class grid_planner
{
    // std::pair<int, int> start;
    std::vector<std::vector<float>> grid;

    float euclidian_distance(std::pair<int, int> a, std::pair<int, int> b)
    {
        float dx = a.first - b.first;
        float dy = a.second - b.second;
        return std::sqrt(dx * dx + dy * dy);
    }

    std::vector<std::pair<int,int>> get_neighbors(std::pair<int, int> node)
    {
        std::vector<std::pair<int,int>> neighbors;
        std::vector<std::pair<int,int>> deltas = {{0,1},{1,0},{-1,0},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
        for(auto d:deltas)
        {
            std::pair<int,int> n(node.first + d.first, node.second + d.second);
            bool in_bounds = n.first >= 0 and n.first < int(grid.size()) and n.second >= 0 and n.second < int(grid.front().size());
            bool traversable = grid[node.first][node.second] == 1;
            if (in_bounds and traversable)
                neighbors.push_back(n);
        }
        return neighbors;
    }

public:

    grid_planner(std::vector<std::vector<float>> _grid):grid(_grid){}

    std::vector<std::vector<float>> fill_true_dists_8_way(std::pair<int, int> goal)
    {
        std::deque<std::pair<int, int>> layer;
        layer.push_back(goal);

        std::vector<std::vector<float>> node_levels(grid.size(), std::vector<float>(grid[0].size(), INF));
        node_levels[goal.first][goal.second] = 0;

        while (!layer.empty())
        {
            std::pair<int, int> cur_layer_node = layer.front();
            layer.pop_front();

            for(auto child_node: get_neighbors(cur_layer_node))
            {
                float step_length = euclidian_distance(cur_layer_node, child_node);
                float child_node_level = node_levels[cur_layer_node.first][cur_layer_node.second] + step_length;

                if (child_node != goal)
                {
                    if (node_levels[child_node.first][child_node.second] == INF or child_node_level < node_levels[child_node.first][child_node.second])
                    {
                        layer.push_back(child_node);
                        node_levels[child_node.first][child_node.second] = child_node_level;
                    }
                }
            }
        }
        
        return node_levels;
    }
};


PYBIND11_MODULE(grid_planner, m) {
    py::class_<grid_planner>(m, "grid_planner")
            .def(py::init<std::vector<std::vector<float>>>())
            .def("fill_true_dists_8_way", &grid_planner::fill_true_dists_8_way);
}

/*
<%
setup_pybind11(cfg)
%>
*/