# -*- coding: utf-8 -*-

import math
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import Stack


def generate_data_sets(repeat_rate):
    i = 0
    data_set = set()
    while i < 1e6 * (1 - repeat_rate):
        num = random.randint(0, 1e7)
        if num not in data_set:
            data_set.add(num)
            i += 1
    while True:
        num = random.randint(0, 1e7)
        if num not in data_set:
            data_set = list(data_set)
            for _ in range(int(1e6 * repeat_rate)):
                data_set.append(num)
            break
    data_set = list(data_set)
    random.shuffle(data_set)
    return data_set


def random_partition(A, p, r):
    i = random.randint(p, r)
    A[r], A[i] = A[i], A[r]
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] < x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1


def random_partition3(A, p, r):
    i = random.randint(p, r)
    x = A[i]
    i = p - 1
    k = r + 1
    j = p
    while j < k:
        if A[j] < x:
            i += 1
            A[i], A[j] = A[j], A[i]
            j += 1
        elif A[j] > x:
            k -= 1
            A[k], A[j] = A[j], A[k]
        else:
            j += 1
    return i + 1, k - 1


def quick_sort_by_recursion(A, p, r):
    if p < r:
        q = random_partition(A, p, r)
        quick_sort_by_recursion(A, p, q - 1)
        quick_sort_by_recursion(A, q + 1, r)


def quick_sort_by_stack(A, p, r):
    stack = Stack()
    q = random_partition(A, p, r)
    if q > p + 1:
        stack.push(p)
        stack.push(q - 1)
    if q < r - 1:
        stack.push(q + 1)
        stack.push(r)
    while not stack.is_empty():
        r = stack.pop()
        p = stack.pop()
        q = random_partition(A, p, r)
        if q > p + 1:
            stack.push(p)
            stack.push(q - 1)
        if q < r - 1:
            stack.push(q + 1)
            stack.push(r)


def quick_sort_by_stack3(A, p, r):
    stack = Stack()
    q, t = random_partition3(A, p, r)
    if q > p + 1:
        stack.push(p)
        stack.push(q - 1)
    if t < r - 1:
        stack.push(t + 1)
        stack.push(r)
    while not stack.is_empty():
        r = stack.pop()
        p = stack.pop()
        q, t = random_partition3(A, p, r)
        if q > p + 1:
            stack.push(p)
            stack.push(q - 1)
        if t < r - 1:
            stack.push(t + 1)
            stack.push(r)


if __name__ == "__main__":
    # data_set = generate_data_sets(0.1)
    data_set = np.random.randint(20, size=20)
    print("origin data:", data_set)
    start = time.time()
    # quick_sort_by_recursion(data_set, 0, len(data_set) - 1)
    quick_sort_by_stack(data_set, 0, len(data_set) - 1)
    time_cost = time.time() - start
    print("Two-way quick sort time:", time_cost)
    print("Two-way quick sort result:", data_set)

    start = time.time()
    quick_sort_by_stack3(data_set, 0, len(data_set) - 1)
    time_cost = time.time() - start
    print("Three-way quick sort time:", time_cost)
    print("Three-way quick sort result:", data_set)

    # s = time.time()
    # time_cost_list_two_way = []
    # time_cost_list_three_way = []
    # time_cost_list_sorted = []
    # for i in tqdm(range(11)):

    #     repeat_rate = i / 10.0
    #     print("\nrepeat rate:",repeat_rate)
    #     data_set = generate_data_sets(repeat_rate)
    #     start = time.time()
    #     if time.time()-s<7200:
    #         # quick_sort_by_stack(data_set, 0, len(data_set) - 1)
    #         time_cost = time.time() - start
    #         print("Two-way quick sort time:", time_cost)
    #         time_cost_list_two_way.append(time_cost)
    #     else:
    #         time_cost_list_two_way.append(time_cost_list_two_way[-1])

    #     start = time.time()
    #     quick_sort_by_stack3(data_set, 0, len(data_set) - 1)
    #     time_cost = time.time() - start
    #     print("Three-way quick sort time:", time_cost)
    #     time_cost_list_three_way.append(time_cost)

    #     start = time.time()
    #     data_set = sorted(data_set)
    #     time_cost = time.time() - start
    #     print("Python Sorted time:", time_cost)
    #     time_cost_list_sorted.append(time_cost)

    # plt.subplot(2, 2, 1)
    # plt.title('Two-way quick sort', fontsize='large')
    # plt.plot([i / 10 for i in range(11)],
    #          time_cost_list_two_way,
    #          'ro-',
    #          label='Two-way quick sort')
    # plt.xlabel('Repeat Rate')
    # plt.ylabel('Time/s')
    # plt.subplot(2, 2, 2)
    # plt.title('Three-way quick sort', fontsize='large')
    # plt.plot([i / 10 for i in range(11)],
    #          time_cost_list_three_way,
    #          'go-',
    #          label='Three-way quick sort')
    # plt.xlabel('Repeat Rate')
    # plt.ylabel('Time/s')

    # plt.subplot(2, 2, 3)
    # plt.title('Python Sorted', fontsize='large')
    # plt.plot([i / 10 for i in range(11)],
    #          time_cost_list_sorted,
    #          'bo-',
    #          label='Python Sorted')
    # plt.xlabel('Repeat Rate')
    # plt.ylabel('Time/s')

    # plt.subplot(2, 2, 4)
    # plt.title('Three Algorithms', fontsize='large')
    # plt.plot([i / 10 for i in range(11)],
    #          time_cost_list_two_way,
    #          'ro-',
    #          label='Two-way quick sort')
    # plt.plot([i / 10 for i in range(11)],
    #          time_cost_list_three_way,
    #          'go-',
    #          label='Three-way quick sort')
    # plt.plot([i / 10 for i in range(11)],
    #          time_cost_list_sorted,
    #          'bo-',
    #          label='Python Sorted')
    # plt.legend(loc='upper left')
    # plt.xlabel('Repeat Rate')
    # plt.ylabel('Time/s')
    # plt.tight_layout()
    # plt.show()
