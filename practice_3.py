# -*- coding: utf-8 -*-

import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from pulp import LpMinimize, LpProblem, LpVariable, value
from tqdm import tqdm


def generate_X(number):
    X = set()
    i = 0
    while i < number:
        x = random.randint(1, 100)
        y = random.randint(1, 100)
        if (x, y) not in X:
            X.add((x, y))
            i += 1
    return X


def generate_F(X, number):
    S = []
    S_union = set()
    other = X - S_union
    while len(other) >= 20:
        if len(S) == 0:
            n, x = 20, 20
        else:
            n = random.randint(1, 20)
            x = random.randint(1, n)
        S_new = set()
        i = 0
        other_list = list(other)
        while i < n:
            j = random.randint(0, len(other) - 2)
            if other_list[j] not in S_new:
                S_new.add(other_list[j])
                i += 1
        i = 0
        S_union_list = list(S_union)
        while i < n - x:
            j = random.randint(0, len(S_union) - 2)
            if S_union_list[j] not in S_new:
                S_new.add(S_union_list[j])
                i += 1
        S_union = S_union | S_new
        S.append(S_new)
        other = X - S_union
    S.append(other)

    X_list = list(X)
    for _ in range(number - len(S)):
        S_new = set()
        i = 0
        n = random.randint(1, 20)
        while i < n:
            j = random.randint(0, len(X) - 2)
            if X_list[j] not in S_new:
                S_new.add(X_list[j])
                i += 1
        S.append(S_new)
    return S


def greedy_set_cover(X, F):

    U = X.copy()
    C = []
    other_F = F.copy()
    while len(U) != 0:
        max_S = set()
        for S in other_F:
            if len(S & U) > len(max_S & U):
                max_S = S
        other_F.remove(max_S)
        U = U - max_S
        C.append(max_S)
    return C


def lp_set_cover(X, F):
    prob = LpProblem('LP_Set_Cover', LpMinimize)
    x = []
    z = 0
    for i in range(len(F)):
        var = LpVariable("x" + str(i), lowBound=0)
        x.append(var)
        z += var
    prob += z
    f_max = 0
    for e in X:
        con = 0
        f_e = 0
        for i in range(len(F)):
            if e in F[i]:
                con += x[i]
                f_e += 1
        if f_max < f_e:
            f_max = f_e
        prob += con >= 1
    # status = prob.solve()
    prob.solve()
    # print(status)
    # print(LpStatus[status])
    res = {}
    for i in prob.variables():
        res[int(i.name[1:])] = i.varValue

    print(f_max, 1 / f_max)
    # print(prob)
    C = []
    for i in range(len(F)):
        if res[i] >= 1.0 / f_max:
            C.append(F[i])
    return value(prob.objective), res, C


if __name__ == "__main__":
    X = generate_X(100)
    F = generate_F(X, 100)
    start = time.time()
    C = greedy_set_cover(X, F)
    time_cost = time.time() - start
    print("Greedy:")
    print("greedy time cost:", time_cost)
    s = set()
    for i in C:
        s = s | i
    print("Result Set:")
    print(s == X)
    print(len(C))
    print(C)
    print("\nLP:")
    start = time.time()
    optimal_cost, optimal_solution, C = lp_set_cover(X, F)
    time_cost = time.time() - start
    print("lp time cost:", time_cost)
    print("Optimal Cost:", optimal_cost)
    print("Optimal Solution:\n", optimal_solution)
    s = set()
    for i in C:
        s = s | i
    print("Result Set:")
    print(s == X)
    print(len(C))
    print(C)

    # number_range = [100, 500, 1000, 2000, 3000, 4000, 5000]
    # time_cost_list_greedy = []
    # time_cost_list_lp = []
    # for number in tqdm(number_range):
    #     X = generate_X(number)
    #     F = generate_F(X, number)

    #     start = time.time()
    #     C = greedy_set_cover(X,F)
    #     time_cost = time.time() - start
    #     time_cost_list_greedy.append(time_cost)

    #     start = time.time()
    #     C = lp_set_cover(X,F)
    #     time_cost = time.time() - start
    #     time_cost_list_lp.append(time_cost)

    # plt.subplot(2,2,1)
    # plt.title('Greedy',fontsize='large')
    # plt.plot(number_range, time_cost_list_greedy, 'ro-', label='Greedy')
    # plt.xlabel('Set Size')
    # plt.ylabel('Time/s')
    # plt.subplot(2,2,2)
    # plt.title('LP',fontsize='large')
    # plt.plot(number_range, time_cost_list_lp, 'go-', label='LP')
    # plt.xlabel('Set Size')
    # plt.ylabel('Time/s')

    # plt.subplot(2,2,3)
    # plt.title('Two Algorithms',fontsize='large')
    # plt.plot(number_range, time_cost_list_greedy, 'ro-', label='Greedy')
    # plt.plot(number_range, time_cost_list_lp, 'go-', label='LP')
    # plt.legend(loc='upper left')
    # plt.xlabel('Set Size')
    # plt.ylabel('Time/s')
    # plt.tight_layout()
    # plt.show()
