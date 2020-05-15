# -*- coding: utf-8 -*-

import heapq
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

SL = 20  #square lenth
LW = 1  #line width


class State(object):
    def __init__(self,
                 s,
                 t,
                 terrain_cost=None,
                 pos=None,
                 pre_state=None,
                 direction=None):
        if pre_state:
            self.coordinate = pos
            self.g = pre_state.get_g() + np.linalg.norm(
                direction) + terrain_cost
            path = pre_state.get_path()
            path.append(direction)
            self.path = path
        else:
            self.coordinate = tuple(s)
            self.g = 0
            self.path = []
        self.h = get_line_distance(self.coordinate, t)
        self.f = self.g + self.h

    def get_g(self):
        return self.g

    def get_coordinate(self):
        return self.coordinate

    def get_path(self):
        return self.path.copy()

    def __gt__(self, other):
        return self.f > other.f

    def __lt__(self, other):
        return self.f < other.f

    def __ge__(self, other):
        return self.f >= self.f

    def __le__(self, other):
        return self.f <= self.f

    def __eq__(self, other):
        return self.f == self.f


def get_map_terrain_cost(terrain_file):
    map_terrain_cost = []
    with open(terrain_file, "r", encoding="utf-8") as f:
        start = tuple(map(int, f.readline().strip().split(',')))
        end = tuple(map(int, f.readline().strip().split(',')))
        for line in f:
            line = list(map(int, line.strip().split(',')))
            map_terrain_cost.append(line)
    map_terrain_cost = np.array(map_terrain_cost)
    return start, end, map_terrain_cost


def get_line_distance(a, b):
    return np.linalg.norm((a[0] - b[0], a[1] - b[1]))
    # return math.sqrt(pow((a[0] - b[0]), 2) + pow((a[1] - b[1]), 2))


def is_same_position(pos1, pos2):
    return pos1[0] == pos2[0] and pos1[1] == pos2[1]


def is_extensible(pos, closed_list, terrain_cost):
    return terrain_cost != -1 and (pos not in closed_list)


def is_in_range(pos, map_size):
    return pos[0] >= 0 and pos[0] < map_size[0] and pos[1] >= 0 and pos[
        1] < map_size[1]


def reverse_path(path):
    new_path = [(-d[0], -d[1]) for d in path]
    new_path.reverse()
    return new_path


def a_star_bidirectional(start, end, map_terrain_cost, directions_dict):
    map_size = map_terrain_cost.shape
    open_list1 = []
    closed_list1 = set()
    initial_state1 = State(start, end)
    heapq.heappush(open_list1, initial_state1)
    open_list2 = []
    closed_list2 = set()
    initial_state2 = State(end, start)
    heapq.heappush(open_list2, initial_state2)
    while True:
        if len(open_list1) == 0 or len(open_list2) == 0:
            return "no meet"
        node1 = heapq.heappop(open_list1)
        cur_pos1 = node1.get_coordinate()
        node2 = heapq.heappop(open_list2)
        cur_pos2 = node2.get_coordinate()
        if get_line_distance(cur_pos1, cur_pos2) < 2:
            path1 = node1.get_path()
            path2 = node2.get_path()
            path2 = reverse_path(path2)
            path1_cost = node1.get_g()
            path2_cost = node2.get_g()
            temp_direction = (cur_pos2[0] - cur_pos1[0],
                              cur_pos2[1] - cur_pos1[1])
            path1.append(temp_direction)
            if not is_same_position(cur_pos1, cur_pos2):
                all_path_cost = path1_cost + path2_cost + np.linalg.norm(
                    temp_direction) + map_terrain_cost[end[0], end[1]]
            else:
                all_path_cost = path1_cost + path2_cost - map_terrain_cost[
                    cur_pos2[0], cur_pos2[1]] + map_terrain_cost[end[0], end[1]]
            return path1, path1_cost, path2, path2_cost, all_path_cost
        for direction in directions_dict:
            new_pos1 = (cur_pos1[0] + direction[0], cur_pos1[1] + direction[1])
            if is_in_range(new_pos1, map_size):
                terrain_cost1 = map_terrain_cost[new_pos1[0], new_pos1[1]]
                if is_same_position(new_pos1, end):
                    return "no meet"
                if is_extensible(new_pos1, closed_list1, terrain_cost1):
                    new_state1 = State(start, end, terrain_cost1, new_pos1,
                                       node1, direction)
                    heapq.heappush(open_list1, new_state1)
            new_pos2 = (cur_pos2[0] + direction[0], cur_pos2[1] + direction[1])
            if is_in_range(new_pos2, map_size):
                terrain_cost2 = map_terrain_cost[new_pos2[0], new_pos2[1]]
                if is_same_position(new_pos2, start):
                    return "no meet"
                if is_extensible(new_pos2, closed_list2, terrain_cost2):
                    new_state2 = State(end, start, terrain_cost2, new_pos2,
                                       node2, direction)
                    heapq.heappush(open_list2, new_state2)
        closed_list1.add(cur_pos1)
        closed_list2.add(cur_pos2)


def a_star_unidirectional(start, end, map_terrain_cost, directions_dict):
    print(start)
    print(end)
    print(map_terrain_cost)
    map_size = map_terrain_cost.shape
    print(map_size)
    open_list = []
    closed_list = set()
    initial_state = State(start, end)
    heapq.heappush(open_list, initial_state)
    while True:
        if len(open_list) == 0:
            return "no arrived"
        node = heapq.heappop(open_list)
        cur_pos = node.get_coordinate()
        for direction in directions_dict:
            new_pos = (cur_pos[0] + direction[0], cur_pos[1] + direction[1])
            if is_in_range(new_pos, map_size):
                terrain_cost = map_terrain_cost[new_pos[0], new_pos[1]]
                if is_same_position(new_pos, end):
                    new_state = State(start, end, terrain_cost, end, node,
                                      direction)
                    path_cost = new_state.get_g()
                    path = new_state.get_path()
                    return path, path_cost
                if is_extensible(new_pos, closed_list, terrain_cost):
                    new_state = State(start, end, terrain_cost, new_pos, node,
                                      direction)
                    heapq.heappush(open_list, new_state)
        closed_list.add(cur_pos)


def get_path_position(start, path):
    path_position = []
    cur_pos = start
    for d in path:
        cur_pos = (cur_pos[0] + d[0], cur_pos[1] + d[1])
        path_position.append(cur_pos)
    return path_position  #[:-1]


def get_path_text(path, directions_dict):
    return [directions_dict[d] for d in path]


def draw_terrain(start, end, map_terrain_cost):
    h, w = map_terrain_cost.shape
    graph = np.ones(((h + 1) * SL, (w + 1) * SL, 3))
    for i in range(0, (h + 1) * SL, SL):
        graph[range(i, i + LW), :-SL + LW] = [0, 0, 0]
    for i in range(0, (w + 1) * SL, SL):
        graph[:-SL + LW, range(i, i + LW)] = [0, 0, 0]
    colors = {-1: [127, 127, 127], 2: [0, 176, 240], 4: [255, 192, 0]}
    for i in range(h):
        for j in range(w):
            if map_terrain_cost[i, j] != 0:
                for k in range(i * SL + LW, (i + 1) * SL):
                    graph[k, range(j * SL + LW, j * SL + SL)] = np.array(
                        colors[map_terrain_cost[i, j]]) / 255
    colors_st = {start: [0, 100, 0], end: [255, 0, 255]}
    for pos in colors_st:
        i, j = pos
        for k in range(i * SL + LW + SL // 4, (i + 1) * SL - SL // 4):
            graph[k, range(j * SL + LW + SL // 4, j * SL + SL -
                           SL // 4)] = np.array(colors_st[pos]) / 255
    return graph


def draw_path(graph, paths, colors, path_cost=None):
    if path_cost != None:
        plt.title("path_cost:" + str(path_cost))
    for path, color in zip(paths, colors):
        for pos in path:
            i, j = pos
            for k in range(i * SL + LW + SL // 4, (i + 1) * SL - SL // 4):
                graph[k,
                      range(j * SL + LW + SL // 4, j * SL + SL -
                            SL // 4)] = np.array(color) / 255
    return graph


def show_graph(graph):

    plt.imshow(graph)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":

    start, end, map_terrain_cost = get_map_terrain_cost("terrain1.csv")
    directions_dict = {
        (-1, 0): "up",
        (1, 0): "down",
        (0, -1): "left",
        (0, 1): "right",
        (-1, 1): "right up",
        (1, 1): "right down",
        (-1, -1): "left up",
        (1, -1): "left down"
    }

    s = time.time()
    path, path_cost = a_star_unidirectional(start, end, map_terrain_cost,
                                            directions_dict)
    print("a star unidirectional time:", time.time() - s)
    s1 = time.time()
    path1, path1_cost, path2, path2_cost, all_path_cost = a_star_bidirectional(
        start, end, map_terrain_cost, directions_dict)
    print("a star bidirectional time:", time.time() - s1)
    print(path)
    print(get_path_text(path, directions_dict))
    print(path_cost)

    print(path1)
    print(get_path_text(path1, directions_dict))
    print(path1_cost)
    print(path2)
    print(get_path_text(path2, directions_dict))
    print(path2_cost)
    print(all_path_cost)
    path_position = get_path_position(start, path)
    print(path_position)
    # plt.subplot(1, 2, 1)
    graph = draw_terrain(start, end, map_terrain_cost)
    graph = draw_path(graph, [path_position[:-1]], [[0, 255, 0]], [path_cost])
    show_graph(graph)

    path1_position = get_path_position(start, path1)
    path2_position = path1_position[-1:] + get_path_position(
        path1_position[-1], path2)[:-1]
    path1_position = path1_position[:-1]
    print(path1_position)
    print(path2_position)
    paths = [path1_position, path2_position]
    # plt.subplot(1, 2, 2)
    graph = draw_terrain(start, end, map_terrain_cost)
    graph = draw_path(graph, paths, [[0, 255, 0], [238, 44, 44]],
                      [path1_cost, path2_cost, all_path_cost])
    if path1[-1] == (0, 0):
        graph = draw_path(graph, [path1_position[-1:]], [[255, 255, 0]])
    show_graph(graph)
    # plt.show()
