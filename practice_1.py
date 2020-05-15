# -*- coding: utf-8 -*-

import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import Stack

X_RANGE = (0, 100)
Y_RANGE = (0, 100)
EPSILON = 1e-6


def generate_dots(number):
    dot_set = set()
    idx = 0
    while idx < number:
        x = random.random() * (X_RANGE[1] - X_RANGE[0]) + X_RANGE[0]
        y = random.random() * (Y_RANGE[1] - Y_RANGE[0]) + Y_RANGE[0]
        if (x, y) not in dot_set:
            dot_set.add((x, y))
            idx += 1
    return dot_set



def g(dot_A, dot_B, dot_P):
    a = dot_B[1] - dot_A[1]  # A = Y2 - Y1
    b = dot_A[0] - dot_B[0]  # B = X1 - X2
    c = dot_B[0] * dot_A[1] - dot_A[0] * dot_B[1]  # C = X2*Y1 - X1*Y2
    return a * dot_P[0] + b * dot_P[1] + c  # A*X+B*Y+C


def atan2_p0(p, p0):
    x, y = p[0] - p0[0], p[1] - p0[1]
    return math.atan2(y, x)


def sort_by_angle(dot_set):
    p0 = min(dot_set, key=lambda dot: (dot[1], dot[0]))
    angle_sored_list = sorted(dot_set, key=lambda p: (atan2_p0(p, p0), p[0]))
    return angle_sored_list


def get_convex_hull_by_brute_force(dot_set):
    convex_hull_set = set()
    for A in dot_set:
        for B in dot_set:
            if A != B:
                pos_flag, neg_flag = False, False
                for P in dot_set:
                    if abs(g(A, B, P)) > EPSILON:
                        if g(A, B, P) > EPSILON:
                            pos_flag = True
                        else:
                            neg_flag = True
                        if pos_flag and neg_flag:
                            break
                if (pos_flag or neg_flag) and not (pos_flag and neg_flag):
                    convex_hull_set.add(A)
                    convex_hull_set.add(B)
    # neg_convex_hull_set = dot_set.difference(convex_hull_set)
    left_dot = min(convex_hull_set)
    right_dot = max(convex_hull_set)
    up_dot_set = set(
        filter(lambda p: g(right_dot, left_dot, p) > EPSILON, convex_hull_set))
    down_dot_set = convex_hull_set.difference(up_dot_set)

    up_dot_list = sorted(up_dot_set, reverse=True)
    down_dot_list = sorted(down_dot_set)
    convex_hull_list = down_dot_list + up_dot_list
    return convex_hull_list


def get_convex_hull_by_graham_scan(angle_sorted_list):
    temp_stack = Stack()
    temp_stack.push(angle_sorted_list[0])
    temp_stack.push(angle_sorted_list[1])
    for p in angle_sorted_list[2:] + angle_sorted_list[:1]:
        while True:
            top = temp_stack.peek()
            second_top = temp_stack.peek_peek()
            b = (p[0] - top[0], p[1] - top[1])
            a = (top[0] - second_top[0], top[1] - second_top[1])
            if a[0] * b[1] - a[1] * b[0] < 0:
                temp_stack.pop()
            else:
                break
        temp_stack.push(p)
    temp_stack.pop()
    return temp_stack.items


def get_convex_hull_by_divide_and_conquer(dot_set):
    if len(dot_set) <= 3:
        return sort_by_angle(dot_set)
    x_median = np.median([dot[0] for dot in dot_set])
    # print("x_median：", x_median)
    left_dot_set = set(filter(lambda p: p[0] <= x_median, dot_set))
    right_dot_set = dot_set.difference(left_dot_set)
    left_convex_hull = get_convex_hull_by_divide_and_conquer(left_dot_set)
    # plot_result(left_convex_hull,left_convex_hull)
    right_convex_hull = get_convex_hull_by_divide_and_conquer(right_dot_set)
    # plot_result(right_convex_hull,right_convex_hull)
    left_y_min, left_y_max = min(
        left_convex_hull, key=lambda dot: dot[1]), max(left_convex_hull,
                                                       key=lambda dot: dot[1])
    p0 = ((left_y_min[0] + left_y_max[0]) / 2,
          (left_y_min[1] + left_y_max[1]) / 2)

    temp = [atan2_p0(p, p0) for p in left_convex_hull]
    left_angle_min_idx = np.argmin(temp)
    l1 = left_convex_hull[
        left_angle_min_idx:] + left_convex_hull[:left_angle_min_idx]

    temp = [atan2_p0(p, p0) for p in right_convex_hull]
    right_angle_min_idx, right_angle_max_idx = np.argmin(temp), np.argmax(temp)

    if right_angle_min_idx < right_angle_max_idx:
        l2 = right_convex_hull[right_angle_min_idx:right_angle_max_idx + 1]
        l3 = right_convex_hull[right_angle_max_idx +
                               1:] + right_convex_hull[:right_angle_min_idx]
    else:
        l2 = right_convex_hull[right_angle_min_idx +
                               1:] + right_convex_hull[:right_angle_max_idx]

        l3 = right_convex_hull[right_angle_max_idx:right_angle_min_idx + 1]
    l3.reverse()

    ordered_list = merge([l1, l2, l3], p0)
    # print("merge：",ordered_list)
    y_min_idx = np.argmin([p[1] for p in ordered_list])
    ordered_list = ordered_list[y_min_idx:] + ordered_list[:y_min_idx]

    merged_convex_hull = get_convex_hull_by_graham_scan(ordered_list)
    # print("sort_from_y_min：",ordered_list)
    # print(merged_convex_hull)
    # plot_result(merge([l1, l2, l3], p0), merged_convex_hull)
    return merged_convex_hull


def merge(origin_ordered_lists, p0):
    result = []
    ordered_lists = []
    for i in range(len(origin_ordered_lists)):
        if len(origin_ordered_lists[i]) >= 1:
            ordered_lists.append(origin_ordered_lists[i])
    idx = [0] * len(ordered_lists)
    lenth = [len(l) for l in ordered_lists]
    while True:
        if len(ordered_lists) == 1:
            return result + ordered_lists[0][idx[0]:]
        candidate_min_list = [
            atan2_p0(ordered_lists[i][idx[i]], p0)
            for i in range(len(ordered_lists))
        ]
        min_idx = np.argmin(candidate_min_list)
        cur_min = ordered_lists[min_idx][idx[min_idx]]
        result.append(cur_min)
        if idx[min_idx] < lenth[min_idx] - 1:
            idx[min_idx] += 1
        else:
            ordered_lists.pop(min_idx)
            idx.pop(min_idx)
            lenth.pop(min_idx)


def plot_result(dot_set, convex_hull_list, fmt='r-'):
    x_dot_set = [dot[0] for dot in dot_set]
    y_dot_set = [dot[1] for dot in dot_set]
    x_convex_hull_list = [dot[0] for dot in convex_hull_list]
    y_convex_hull_list = [dot[1] for dot in convex_hull_list]
    plt.scatter(x_dot_set, y_dot_set)
    plt.plot(x_convex_hull_list, y_convex_hull_list, fmt)


if __name__ == "__main__":

    dot_set = generate_dots(1000)
    start = time.time()
    convex_hull_list_bf = get_convex_hull_by_brute_force(dot_set)
    convex_hull_list_bf += convex_hull_list_bf[:1]
    time_cost_bf = time.time() - start

    start = time.time()
    convex_hull_list_gs = get_convex_hull_by_graham_scan(
        sort_by_angle(dot_set))
    convex_hull_list_gs += convex_hull_list_gs[:1]
    time_cost_gs = time.time() - start

    start = time.time()
    convex_hull_list_dc = get_convex_hull_by_divide_and_conquer(dot_set)
    convex_hull_list_dc += convex_hull_list_dc[:1]
    time_cost_dc = time.time() - start

    plt.subplot(2, 2, 1)
    plt.title('Brute Force\n' + '%.3fms' % (time_cost_bf * 1000),
              fontsize='large')
    plot_result(dot_set, convex_hull_list_bf)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.subplot(2, 2, 2)
    plt.title('Graham Scan\n' + '%.3fms' % (time_cost_gs * 1000),
              fontsize='large')
    plot_result(dot_set, convex_hull_list_gs)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.subplot(2, 2, 3)
    plt.title('Divide and Conquer\n' + '%.3fms' % (time_cost_dc * 1000),
              fontsize='large')
    plot_result(dot_set, convex_hull_list_dc)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

    # dot_number_range = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    # time_cost_list_bf = []
    # time_cost_list_gs = []
    # time_cost_list_dc = []
    # for number in tqdm(dot_number_range):
    #     dot_set = generate_dots(number)

    #     start = time.time()
    #     convex_hull_list_bf = get_convex_hull_by_brute_force(dot_set)
    #     convex_hull_list_bf += convex_hull_list_bf[:1]
    #     time_cost = time.time() - start
    #     time_cost_list_bf.append(time_cost)

    #     start = time.time()
    #     convex_hull_list_gs = get_convex_hull_by_graham_scan(
    #         sort_by_angle(dot_set))
    #     convex_hull_list_gs += convex_hull_list_gs[:1]
    #     time_cost = time.time() - start
    #     time_cost_list_gs.append(time_cost)

    #     start = time.time()
    #     convex_hull_list_dc = get_convex_hull_by_divide_and_conquer(dot_set)
    #     convex_hull_list_dc += convex_hull_list_dc[:1]
    #     time_cost = time.time() - start
    #     time_cost_list_dc.append(time_cost)

    # plt.subplot(2,2,1)
    # plt.title('Brute Force',fontsize='large')
    # plt.plot(dot_number_range, time_cost_list_bf, 'ro-', label='Brute Force')
    # plt.xlabel('Data Size')
    # plt.ylabel('Time/s')
    # plt.subplot(2,2,2)
    # plt.title('Graham Scan',fontsize='large')
    # plt.plot(dot_number_range, time_cost_list_gs, 'go-', label='Graham Scan')
    # plt.xlabel('Data Size')
    # plt.ylabel('Time/s')
    # plt.subplot(2,2,3)
    # plt.title('Divide and Conquer',fontsize='large')
    # plt.plot(dot_number_range,
    #          time_cost_list_dc,
    #          'bo-',
    #          label='Divide and Conquer')
    # plt.xlabel('Data Size')
    # plt.ylabel('Time/s')
    # plt.subplot(2,2,4)
    # plt.title('Three Algorithms',fontsize='large')
    # plt.plot(dot_number_range, time_cost_list_bf, 'ro-', label='Brute Force')
    # plt.plot(dot_number_range, time_cost_list_gs, 'go-', label='Graham Scan')
    # plt.plot(dot_number_range,
    #          time_cost_list_dc,
    #          'bo-',
    #          label='Divide and Conquer')
    # plt.legend(loc='upper left')
    # plt.xlabel('Data Size')
    # plt.ylabel('Time/s')
    # plt.tight_layout()
    # plt.show()
