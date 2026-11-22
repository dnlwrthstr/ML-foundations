#!/usr/bin/env python3
"""Concept sheets for notebooks 7 (Algorithms & Data Structures) and 8 (Optimization), one per segment.

Notebook 7: Segment 1 (Big O), Segment 2 (lists, searching, sorting, hashing), Segment 3 (trees).
Notebook 8: Segment 2 (cost functions, critical points), Segment 3 (Jacobian and Hessian matrices);
Segment 1 of notebook 8 is a heading with a pointer to the slides, so it gets no sheet.

Every number is computed here from the notebook's own arrays, functions and parameters and is checked
against its printed outputs with asserts. Wall-clock timings vary by machine, so the complexity shapes
are asserted with exact operation counts; the notebook's own printed timings are drawn as printed.
Values that need a download (seaborn's titanic) or a missing package (xgboost) are shown as printed.
Images are written to notebooks/img/ (<notebook stem>_seg<N>.png).

Run from anywhere:  /opt/anaconda3/bin/python scripts/make_nb7_8_concept_images.py
"""
import math
import random
import sys
from functools import partial

import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA, GRID,
                       new_figure, grid, panel)
from viz_style import save as _save

NOTEBOOKS = REPO / "notebooks"
save = partial(_save, folder=NOTEBOOKS / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"
PURPLE = "#7d5ba6"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", SURFACE, "#f2a07c", "#c4481a"])

sys.setrecursionlimit(10000)


# ================================================================ helpers
def loglog(ax, title, xlabel="n (list length)", ylabel="seconds"):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def slope_guide(ax, n, anchor_n, anchor_y, power, label, color=MUTED):
    n = np.asarray(n, float)
    ax.plot(n, anchor_y * (n / anchor_n) ** power, color=color, linestyle=":", linewidth=1.4, label=label)


def node_box(ax, xy, text, face, edge=None, fontsize=9.5, width=0.17, height=0.2, color=INK):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                boxstyle="round,pad=0.008,rounding_size=0.02", facecolor=face,
                                edgecolor=edge or face, linewidth=2 if edge else 0, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color, zorder=3)


# ---------------------------------------------------------------- notebook 7 algorithms (copied verbatim,
# plus an operation counter where the sheet needs counts)
def take_first(my_list):
    return my_list[0]


def find_max(my_list):
    max_value = my_list[0]
    for i in range(len(my_list)):
        if my_list[i] > max_value:
            max_value = my_list[i]
    return max_value


def element_multiplier_count(my_list):
    count = 0
    for i in range(len(my_list)):
        for j in range(len(my_list)):
            x = my_list[i] * my_list[j]
            count += 1
    return count


def binary_search(array, value, probes=None):
    low = 0
    high = len(array) - 1
    while low <= high:
        midpoint = (low + high) // 2
        if probes is not None:
            probes.append(midpoint)
        if value < array[midpoint]:
            high = midpoint - 1
        elif value > array[midpoint]:
            low = midpoint + 1
        else:
            return midpoint
    return -1


def recursive_binary_search(array, value, low, high):
    if low <= high:
        midpoint = (low + high) // 2
        if array[midpoint] == value:
            return midpoint
        elif array[midpoint] > value:
            return recursive_binary_search(array, value, low, midpoint - 1)
        else:
            return recursive_binary_search(array, value, midpoint + 1, high)
    else:
        return -1


def bubble_sort(array, counter=None):
    n = len(array)
    for i in range(n - 1):
        for j in range(0, n - 1):
            if counter is not None:
                counter[0] += 1
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
    return array


def merge_sort(my_list, counter=None):
    if len(my_list) > 1:
        midpoint = len(my_list) // 2
        left_half = my_list[:midpoint]
        right_half = my_list[midpoint:]
        left_half = merge_sort(left_half, counter)
        right_half = merge_sort(right_half, counter)
        my_list = []
        while len(left_half) > 0 and len(right_half) > 0:
            if counter is not None:
                counter[0] += 1
            if left_half[0] < right_half[0]:
                my_list.append(left_half.pop(0))
            else:
                my_list.append(right_half.pop(0))
        for i in left_half:
            my_list.append(i)
        for i in right_half:
            my_list.append(i)
    return my_list


def quick_sort(array, counter=None):
    _quick_sort(array, 0, len(array) - 1, counter)
    return array


def _quick_sort(array, low, high, counter=None):
    if low < high:
        p_index = partition(array, low, high, counter)
        _quick_sort(array, low, p_index - 1, counter)
        _quick_sort(array, p_index + 1, high, counter)
    return array


def partition(array, low, high, counter=None):
    pivot = array[high]
    i = (low - 1)
    for j in range(low, high):
        if counter is not None:
            counter[0] += 1
        if array[j] < pivot:
            i = i + 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[high] = array[high], array[i + 1]
    return (i + 1)


def simple_hash(v):
    split_v = [digit for digit in str(v)]
    final_2 = int(''.join(split_v[-2:]))
    return final_2 % 10


def comparisons(sort_fn, lst):
    counter = [0]
    result = sort_fn(list(lst), counter)
    assert result == sorted(lst)
    return counter[0]


# ================================================================ notebook 7, Segment 1
LIST_LENGTHS = [10 ** l for l in range(1, 8)]
GRANULAR = [10, 50, 100, 500, 1000, 5000, 10000]
# the notebook's printed timing tables (time.process_time on the author's machine, rounded to 6 decimals)
CONSTANT_TIMES = [0.000008, 0.000001, 0.000001, 0.000001, 0.000004, 0.000014, 0.000610]
LINEAR_TIMES = [0.000010, 0.000012, 0.000135, 0.001169, 0.011611, 0.157893, 1.258876]
POLY_TIMES = [0.000028, 0.000383, 0.001468, 0.035158, 0.182454, 3.122360, 11.175045]


def sheet_nb7_seg1_big_o():
    assert LIST_LENGTHS == [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    assert sorted(LIST_LENGTHS[:4] + [50, 500, 5000]) == GRANULAR
    short_list = [13, 25, 42]
    assert take_first(short_list) == 13 and find_max(short_list) == 42

    # exact work: take_first touches one element whatever n is, find_max makes n comparisons,
    # element_multiplier n*n multiplications (counted by running them; n^2 counted up to n = 1000)
    def find_max_count(lst):
        count = 0
        max_value = lst[0]
        for i in range(len(lst)):
            count += 1
            if lst[i] > max_value:
                max_value = lst[i]
        return count

    linear_counts = [find_max_count([42] * n) for n in LIST_LENGTHS[:6]]
    assert linear_counts == LIST_LENGTHS[:6]
    poly_n = [n for n in GRANULAR if n <= 1000]
    poly_counts = [element_multiplier_count([42] * n) for n in poly_n]
    assert poly_counts == [n * n for n in poly_n]

    # per-operation cost from the printed tables: time / n and time / n^2 level off at a constant
    per_op_linear = np.array(LINEAR_TIMES) / np.array(LIST_LENGTHS)
    per_op_poly = np.array(POLY_TIMES) / np.array(GRANULAR, float) ** 2
    # largest-n values agree within ~15 % for each table: that constant is what Big O throws away
    assert abs(per_op_linear[-1] / per_op_linear[-2] - 1) < 0.25
    assert abs(per_op_poly[-1] / per_op_poly[-2] - 1) < 0.25
    ratio_linear = LINEAR_TIMES[-1] / LINEAR_TIMES[-2]
    ratio_poly = POLY_TIMES[-1] / POLY_TIMES[-3]   # n: 1000 -> 10000 (x10)
    assert f"{ratio_linear:.1f}" == "8.0" and f"{ratio_poly:.0f}" == "61"

    # the "steps at a glance" table
    for n_t, row in [(10, ("3.3", "33", "100", "1 024")), (100, ("6.6", "664", "1e4", "1.3e30")),
                     (1000, ("10.0", "9 966", "1e6", "1.1e301"))]:
        assert (f"{math.log2(n_t):.1f}", f"{n_t * math.log2(n_t):,.0f}".replace(",", " "),
                f"{n_t ** 2:.0e}".replace("e+0", "e").replace("e+", "e") if n_t > 10 else "100",
                f"{2 ** n_t:,}".replace(",", " ") if n_t == 10 else f"{float(2 ** n_t):.1e}".replace("e+", "e")
                ) == row, row

    fig = new_figure("Big O: describe how the work grows with n, not the seconds",
                     "take_first → O(1)   ·   find_max → O(n)   ·   element_multiplier → O(n²).   "
                     "Keep the dominant term, drop constants: (n − 1)² + 5 is still O(n²).")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    loglog(ax, "Printed timings (vary by machine)")
    ax.plot(LIST_LENGTHS, CONSTANT_TIMES, "o-", color=AQUA, label="take_first  O(1)")
    ax.plot(LIST_LENGTHS, LINEAR_TIMES, "o-", color=BLUE, label="find_max  O(n)")
    ax.plot(GRANULAR, POLY_TIMES, "o-", color=ORANGE, label="element_multiplier  O(n²)")
    slope_guide(ax, [1e3, 1e7], 1e7, LINEAR_TIMES[-1] * 2.5, 1, "slope 1")
    slope_guide(ax, [300, 1e4], 1e4, POLY_TIMES[-1] * 2.5, 2, "slope 2", color=INK2)
    ax.set_ylim(3e-7, 3e2)
    ax.legend(fontsize=9, loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    loglog(ax, "Exact operation counts (asserted)", ylabel="operations")
    ax.plot(LIST_LENGTHS, [1] * len(LIST_LENGTHS), "o-", color=AQUA, label="take_first: 1 lookup")
    ax.plot(LIST_LENGTHS[:6], linear_counts, "o-", color=BLUE, label="find_max: n comparisons")
    ax.plot(poly_n, poly_counts, "o-", color=ORANGE, label="element_multiplier: n² products")
    ax.plot([1000, 10000], [1e6, 1e8], "o--", color=ORANGE, mfc=SURFACE, label="n² (formula)")
    ax.set_ylim(0.3, 3e9)
    ax.legend(fontsize=9, loc="upper left")

    panel(fig.add_subplot(gs[0, 2]), "Three functions, three growth rates", [
        "`take_first(lst):  return lst[0]`",
        "one index lookup, whatever the length",
        "",
        "`find_max(lst):  for i in range(n): ...`",
        "one comparison per element",
        "",
        "`element_multiplier:  for i ...: for j ...:`",
        "every element times every element",
        "",
        "n: 1000 → 10000 (×10) in the printed tables:",
        f"find_max ×{ratio_linear:.1f},  element_multiplier ×{ratio_poly:.0f}",
        "(×10 and ×100 if the machine were ideal)",
    ], spacing=0.074)

    ax = fig.add_subplot(gs[1, 0])
    n = np.arange(1, 21)
    curves = [("O(1)", np.ones_like(n, dtype=float), AQUA),
              ("O(log n)", np.log2(n) + 1, BLUE_LIGHT),
              ("O(n)", n.astype(float), BLUE),
              ("O(n log n)", n * np.log2(n) + 1, PURPLE),
              ("O(n²)", n.astype(float) ** 2, ORANGE),
              ("O(2ⁿ)", 2.0 ** n, "#c4481a"),
              ("O(n!)", np.array([math.factorial(k) for k in n], float), INK)]
    for name, values, color in curves:
        ax.plot(n, values, color=color, linewidth=2, label=name)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1e9)
    ax.set_xlim(1, 20)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(True)
    ax.set_xlabel("n")
    ax.set_ylabel("steps (log scale)")
    ax.set_title("Common runtimes, smallest to largest")
    ax.legend(fontsize=9, loc="upper left", ncol=2)

    ax = fig.add_subplot(gs[1, 1])
    loglog(ax, "Divide by the Big O: a constant is left", ylabel="seconds per operation")
    ax.plot(LIST_LENGTHS, per_op_linear, "o-", color=BLUE, label="find_max time / n")
    ax.plot(GRANULAR, per_op_poly, "o-", color=ORANGE, label="element_multiplier time / n²")
    ax.set_ylim(1e-8, 3e-6)
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.04, 0.05, "small n: fixed overhead dominates\nlarge n: flat → only the constant differs",
            transform=ax.transAxes, fontsize=9, color=INK2)

    panel(fig.add_subplot(gs[1, 2]), "Steps at a glance (log = log₂)", [
        "`n      log n   n log n      n^2      2^n`",
        "`10       3.3        33      100    1 024`",
        "`100      6.6       664      1e4   1.3e30`",
        "`1000    10.0     9 966      1e6   1.1e301`",
        "",
        "Rules of the notation:",
        "  keep the fastest-growing term only",
        "  drop constant factors (2n is O(n))",
        "  usually quote the worst case",
        "Space complexity is counted the same way.",
    ], spacing=0.08)
    save(fig, name="7-algos-and-data-structures_seg1.png")


# ================================================================ notebook 7, Segment 2
BUBBLE_TIMES = [0.000033, 0.000578, 0.002153, 0.057291, 0.260174, 6.131653, 16.825840]
EXT_GRANULAR = GRANULAR + [50000, 100000]
MERGE_TIMES = [0.000045, 0.000226, 0.000287, 0.001795, 0.003506, 0.026664, 0.065815, 0.482626, 1.500227]
QUICK_TIMES = [0.000024, 0.000143, 0.002078, 0.034235, 0.422946, 5.986721]


def sheet_nb7_seg2_lists_search_sort_hash():
    # lists, arrays, stacks, dicts (the cells' outputs)
    t = [25, 2, 5]
    t.append(26)
    assert t == [25, 2, 5, 26] and t[2] == 5
    x = np.array([25, 2, 5])
    assert np.allclose(x / 2., [12.5, 1., 2.5]) and np.append(x, 26).tolist() == [25, 2, 5, 26]
    assert x.tolist() == [25, 2, 5]  # np.append returned a copy
    s = []
    for card in ["five_of_diamonds", "queen_of_hearts", "ace_of_spades"]:
        s.append(card)
    assert s.pop() == "ace_of_spades" and s == ["five_of_diamonds", "queen_of_hearts"]
    inventory = {"hammer": [17], "nail": [2552], "unicorn": [2], "dolphin": ["none"]}
    inventory["dolphin"] = [0]
    inventory["unicorn"].append(3)
    assert inventory == {"hammer": [17], "nail": [2552], "unicorn": [2, 3], "dolphin": [0]}

    # binary search on the sorted b
    b = np.array([25, 2, 5, 14, 22, 11, 96, 1, 101])
    b.sort()
    assert b.tolist() == [1, 2, 5, 11, 14, 22, 25, 96, 101]
    assert binary_search(b, 25) == 6 and binary_search(b, 255) == -1
    assert recursive_binary_search(b, 25, 0, len(b) - 1) == 6
    assert recursive_binary_search(b, 255, 0, len(b) - 1) == -1
    probe_rows = []
    for target in [25, 22, 101, 255]:
        probes = []
        binary_search(b, target, probes)
        probe_rows.append((target, probes))
    assert [len(p) for _, p in probe_rows] == [2, 3, 4, 4]          # "two steps", 22 next, worst case four
    for v in b:                                                         # every value found in <= 4 steps
        probes = []
        assert binary_search(b, v, probes) >= 0 and len(probes) <= 4
    assert [np.log2(4) + 1, np.log2(8) + 1, np.log2(16) + 1] == [3.0, 4.0, 5.0]
    # worst case over every present value and every gap, for n = 1..40
    ns = np.arange(1, 41)
    worst = []
    for n in ns:
        arr = list(range(0, 2 * n, 2))
        steps = []
        for target in range(-1, 2 * n):
            probes = []
            binary_search(arr, target, probes)
            steps.append(len(probes))
        worst.append(max(steps))
    assert worst == [int(np.floor(np.log2(n))) + 1 for n in ns]

    # sorts: the notebook's example arrays, then comparison counts on random.sample lists
    assert bubble_sort(np.array([25, 2, 5, 14, 22, 11, 96, 1, 101])).tolist() == b.tolist()
    assert merge_sort([25, 2, 5, 14, 22, 11, 96, 1, 101]) == b.tolist()
    assert quick_sort(np.array([25, 2, 5, 14, 22, 11, 96, 1, 101])).tolist() == b.tolist()
    rng = random.Random(7)  # the notebook draws unseeded samples; any draw gives the same shapes
    count_n = [10, 50, 100, 500, 1000, 2000]
    bubble_c, merge_c, quick_c, quick_sorted_c = [], [], [], []
    for n in count_n:
        lst = rng.sample(range(0, max(GRANULAR)), n)
        bubble_c.append(comparisons(bubble_sort, lst))
        merge_c.append(comparisons(merge_sort, lst))
        quick_c.append(comparisons(quick_sort, lst))
        quick_sorted_c.append(comparisons(quick_sort, sorted(lst)))
    assert bubble_c == [(n - 1) ** 2 for n in count_n]                 # (n-1)(n-1) = n^2 - 2n + 1
    assert quick_sorted_c == [n * (n - 1) // 2 for n in count_n]        # already sorted: worst case
    assert all(c <= n * np.log2(n) for c, n in zip(merge_c, count_n))  # merge: at most n log2 n
    big_n = [10000, 100000]
    merge_big = [comparisons(merge_sort, rng.sample(range(0, 100000), n)) for n in big_n]
    quick_big = [comparisons(quick_sort, rng.sample(range(0, 100000), n)) for n in big_n]

    # hashing
    assert simple_hash(5551234567) == 7 and simple_hash(5557654321) == 1
    assert simple_hash(555) == 5 and simple_hash(125) == 5
    hash_map = {}
    hash_map[simple_hash(5551234567)] = [(5551234567, "Jane Dough")]
    hash_map[simple_hash(5551234568)] = [(5551234568, "Johnny Dochevski")]
    hash_map[simple_hash(5551234578)].append((5551234578, "Jon Krohn"))
    assert hash_map == {7: [(5551234567, "Jane Dough")],
                        8: [(5551234568, "Johnny Dochevski"), (5551234578, "Jon Krohn")]}
    assert 10 / 1e9 == 1e-08
    assert "".join(format(ord(c), "o") for c in "Jon") == "112157156"   # octal ASCII codes

    fig = new_figure("Lists, search, sort, hash: the data structure decides the cost",
                     "binary search O(log n) on sorted data   ·   bubble sort O(n²)   ·   merge / quick sort "
                     "O(n log n)   ·   hash lookup O(1) while load factor = values / buckets stays small")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    n_b = len(b)
    for r, (target, probes) in enumerate(probe_rows):
        y = r
        for k, v in enumerate(b):
            order = probes.index(k) + 1 if k in probes else None
            found = order is not None and b[k] == target
            face = ORANGE if found else (BLUE_LIGHT if order else "#ecebe6")
            ax.add_patch(Rectangle((k + 0.05, y + 0.08), 0.9, 0.84, facecolor=face, edgecolor="none"))
            ax.text(k + 0.5, y + 0.42, str(v), ha="center", va="center", fontsize=9.5,
                    color="white" if found else INK)
            if order:
                ax.text(k + 0.5, y + 0.8, str(order), ha="center", va="center", fontsize=7.5,
                        color=INK2, weight="bold")
        result = binary_search(b, target)
        ax.text(n_b + 0.15, y + 0.5, f"{len(probes)} steps → {result}", va="center", fontsize=9.5,
                color=INK2)
    ax.set_xlim(0, n_b + 2.6)
    ax.set_ylim(len(probe_rows), 0)
    ax.set_yticks(np.arange(len(probe_rows)) + 0.5, [f"find {t}" for t, _ in probe_rows])
    ax.set_xticks(np.arange(n_b) + 0.5, [str(k) for k in range(n_b)], fontsize=8.5)
    ax.set_xlabel("index in sorted b (small digit = probe order)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Binary search halves the sorted b each step")

    ax = fig.add_subplot(gs[0, 1])
    ax.step(ns, worst, where="post", color=BLUE, linewidth=2.2, label="worst case, counted")
    xs = np.linspace(1, 40, 300)
    ax.plot(xs, np.log2(xs) + 1, color=ORANGE, linestyle="--", label="log₂ n + 1")
    for n_pt, steps in [(4, 3), (8, 4), (16, 5), (32, 6)]:
        ax.plot(n_pt, steps, "o", color=ORANGE, zorder=4)
    ax.plot(ns, ns, color=MUTED, linewidth=1.2, label="linear scan: n")
    ax.plot(9, 4, "*", color=INK, markersize=12, zorder=5, label="b: n = 9, 4 steps")
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 40)
    ax.grid(True)
    ax.set_xlabel("n (sorted array length)")
    ax.set_ylabel("steps")
    ax.set_title("Doubling n adds one step")
    ax.legend(fontsize=9, loc="upper left")

    ax = fig.add_subplot(gs[0, 2])
    loglog(ax, "Comparisons made (counted, random lists)", ylabel="comparisons")
    ax.plot(count_n, bubble_c, "o-", color=ORANGE, label="bubble: (n − 1)² exactly")
    ax.plot(count_n, quick_sorted_c, "s--", color="#c4481a", mfc=SURFACE,
            label="quick, already sorted: n(n − 1)/2")
    ax.plot(count_n + big_n, merge_c + merge_big, "o-", color=BLUE, label="merge ≤ n log₂ n")
    ax.plot(count_n + big_n, quick_c + quick_big, "o-", color=AQUA, label="quick, random order")
    ax.set_ylim(5, 1e8)
    ax.legend(fontsize=8.5, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    loglog(ax, "The notebook's printed sort timings")
    ax.plot(GRANULAR, BUBBLE_TIMES, "o-", color=ORANGE, label="bubble sort")
    ax.plot(EXT_GRANULAR, MERGE_TIMES, "o-", color=BLUE, label="merge sort")
    ax.plot(LIST_LENGTHS[:6], QUICK_TIMES, "o-", color=AQUA, label="quick sort")
    ax.text(0.97, 0.05, f"at n = 10 000:\nbubble {BUBBLE_TIMES[-1]:.1f} s\nmerge {MERGE_TIMES[6]:.3f} s\n"
            f"quick {QUICK_TIMES[3]:.3f} s\n(vary by machine)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9, color=INK2)
    ax.set_ylim(1e-5, 1e2)
    ax.legend(fontsize=9, loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(10.4, -1.0)
    ax.set_title("hash_map: bucket = last two digits % 10")
    rows = {k: hash_map.get(k, []) for k in range(10)}
    for k in range(10):
        ax.add_patch(Rectangle((0.0, k + 0.08), 0.9, 0.84, facecolor=BLUE if rows[k] else "#ecebe6",
                               edgecolor="none"))
        ax.text(0.45, k + 0.5, str(k), ha="center", va="center", fontsize=10,
                color="white" if rows[k] else INK2)
        x0 = 1.2
        for key, name in rows[k]:
            ax.add_patch(FancyBboxPatch((x0, k + 0.12), 4.2, 0.76, boxstyle="round,pad=0,rounding_size=0.15",
                                        facecolor=ORANGE_LIGHT if len(rows[k]) > 1 else BLUE_LIGHT,
                                        edgecolor="none"))
            ax.text(x0 + 2.1, k + 0.5, f"…{str(key)[-4:]}: {name}", ha="center", va="center", fontsize=8.5)
            x0 += 4.4
    ax.text(1.2, 9.5, "collision: …68 and …78 both → 8, so bucket 8 holds a list", fontsize=9,
            color=INK2, ha="left", va="center")
    ax.text(1.2, 5.5, "555 and 125 → 5 as well", fontsize=9, color=INK2, va="center")
    ax.text(1.2, -0.5, f"3 receipts, 10 buckets: load factor = 3 / 10 = {3 / 10:.1f}", fontsize=10, color=INK, ha="left",
            va="center", weight="bold")

    panel(fig.add_subplot(gs[1, 2]), "What each structure buys", [
        "list  t = [25, 2, 5]; t.append(26); t[2]",
        "`[25, 2, 5, 26]   5    (append, index: O(1))`",
        "array  x = np.array([25, 2, 5]); x/2.",
        "`array([12.5,  1. ,  2.5])   (math works)`",
        "stack, last in first out: 3 pushes, s.pop()",
        "`'ace_of_spades'`",
        "dict = hash map: inventory['unicorn'].append(3)",
        "`'unicorn': [2, 3]`",
        "",
        "Load factor 10 / 10⁹ = 10⁻⁸: memory wasted;",
        "aim for 0.6–0.75; above it collisions",
        "grow and a bucket search becomes O(n).",
    ], spacing=0.07)
    save(fig, name="7-algos-and-data-structures_seg2.png")


# ================================================================ notebook 7, Segment 3
# plot_tree(dt_model) printed node texts. x[0] = First, x[1] = Second, x[2] = female.
TREE_NODES = {
    "root": dict(gini=0.467, samples=596, value=(374, 222)),
    "male": dict(gini=0.304, samples=390, value=(317, 73)),
    "male_not_first": dict(gini=0.263, samples=314, value=(265, 49)),
    "male_third": dict(gini=0.252, samples=237, value=(202, 35)),
    "male_second": dict(gini=0.298, samples=77, value=(63, 14)),
    "male_first": dict(gini=0.432, samples=76, value=(52, 24)),
    "female": dict(gini=0.4, samples=206, value=(57, 149)),
    "female_not_first": dict(gini=0.465, samples=147, value=(54, 93)),
    "female_third": dict(gini=0.498, samples=99, value=(53, 46)),
    "female_second": dict(gini=0.041, samples=48, value=(1, 47)),
    "female_first": dict(gini=0.097, samples=59, value=(3, 56)),
}


def gini(value):
    p = np.asarray(value, float) / sum(value)
    return 1.0 - np.sum(p ** 2)


def sheet_nb7_seg3_trees():
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd

    # --- Titanic tree (seaborn's titanic is a download, so the printed tree is used)
    for name, node in TREE_NODES.items():
        assert f"{gini(node['value']):.3f}".rstrip("0") == f"{node['gini']:.3f}".rstrip("0"), name
        assert sum(node["value"]) == node["samples"], name
    # children add up to their parents
    def add(*names):
        return tuple(map(sum, zip(*[TREE_NODES[k]["value"] for k in names])))
    assert add("male", "female") == TREE_NODES["root"]["value"]
    assert add("male_not_first", "male_first") == TREE_NODES["male"]["value"]
    assert add("male_third", "male_second") == TREE_NODES["male_not_first"]["value"]
    assert add("female_not_first", "female_first") == TREE_NODES["female"]["value"]
    assert add("female_third", "female_second") == TREE_NODES["female_not_first"]["value"]
    assert (549, 342) == (549, 891 - 549) and 596 + 295 == 891          # printed counts and split sizes
    rose = TREE_NODES["female_first"]["value"]
    jack = TREE_NODES["male_third"]["value"]
    p_rose = np.array(rose) / sum(rose)
    p_jack = np.array(jack) / sum(jack)
    assert np.allclose(p_rose, [0.05084746, 0.94915254]) and np.allclose(p_jack, [0.85232068, 0.14767932])
    root_gain = gini(TREE_NODES["root"]["value"]) - (
        390 / 596 * gini(TREE_NODES["male"]["value"]) + 206 / 596 * gini(TREE_NODES["female"]["value"]))
    dt_acc, dt_auc = 0.7728813559322034, 0.7273809523809524       # printed test metrics
    assert round(dt_acc * 295) == 228

    # --- iris (sklearn's bundled copy; same rows the notebook selects, same split)
    data = load_iris(as_frame=True)
    iris = data.frame.rename(columns={"sepal width (cm)": "sepal_width", "petal length (cm)": "petal_length"})
    iris["species"] = pd.Series(data.target).map(dict(enumerate(data.target_names)))
    X = iris[["sepal_width", "petal_length"]]
    y = iris.species
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    assert list(y_train.index[:6]) == [96, 105, 66, 0, 122, 67]
    assert list(y_train[:6]) == ["versicolor", "virginica", "versicolor", "setosa", "virginica", "versicolor"]
    assert len(X_test) == 50
    dt_accs = {accuracy_score(DecisionTreeClassifier(random_state=s).fit(X_train, y_train).predict(X_test),
                              y_test) for s in range(10)}
    assert dt_accs == {0.9}                                            # printed 0.9, for every seed
    iris_dt = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    # the notebook sets no seed: 20 seeds give 0.94-0.98; the first seed that reproduces 0.96 is shown
    rf_spread = [accuracy_score(RandomForestClassifier(n_estimators=100, random_state=s)
                                .fit(X_train, y_train).predict(X_test), y_test) for s in range(20)]
    assert set(rf_spread) <= {0.94, 0.96, 0.98} and rf_spread.count(0.96) >= 10
    rf_seed = rf_spread.index(0.96)
    rf_acc = rf_spread[rf_seed]
    assert rf_acc == 0.96                                              # printed 0.96
    # xgboost is not installed here: its printed results are shown
    xg_first = np.array([0.14787205, 0.6481497, 0.20397826])
    xg_acc = 0.98
    assert np.argmax(xg_first) == 1 and abs(xg_first.sum() - 1) < 1e-6
    assert y_test.iloc[0] == "versicolor"

    fig = new_figure("Trees: ask yes/no questions that purify the groups, then let many trees vote",
                     "Gini impurity = 1 − Σ pₖ²   ·   a split is chosen to lower the weighted impurity   ·   "
                     "random forest averages 100 trees   ·   boosting adds trees that fix earlier errors")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0:2])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("Titanic tree fitted in the notebook (printed plot_tree)")

    def label(key, head=None):
        node = TREE_NODES[key]
        s0, s1 = node["value"]
        text = f"{head}\n" if head else ""
        return text + f"n = {node['samples']}  [{s0}, {s1}]\ngini {node['gini']:.3f}\nsurvive {s1 / (s0 + s1):.0%}"

    pos = {"root": (0.48, 0.9), "male": (0.25, 0.62), "female": (0.72, 0.62),
           "male_not_first": (0.15, 0.36), "male_first": (0.36, 0.36),
           "female_not_first": (0.62, 0.36), "female_first": (0.84, 0.36),
           "male_third": (0.07, 0.08), "male_second": (0.23, 0.08),
           "female_third": (0.54, 0.08), "female_second": (0.70, 0.08)}
    edges = [("root", "male", "male"), ("root", "female", "female"),
             ("male", "male_not_first", "not 1st"), ("male", "male_first", "1st class"),
             ("female", "female_not_first", "not 1st"), ("female", "female_first", "1st class"),
             ("male_not_first", "male_third", "3rd"), ("male_not_first", "male_second", "2nd"),
             ("female_not_first", "female_third", "3rd"), ("female_not_first", "female_second", "2nd")]
    for a, c, text in edges:
        (x1, y1), (x2, y2) = pos[a], pos[c]
        ax.plot([x1, x2], [y1 - 0.08, y2 + 0.08], color=MUTED, linewidth=1.2, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, text, fontsize=8.5, color=INK2, ha="center", va="center",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1), zorder=2)
    heads = {"root": "female?", "male": "1st class?", "female": "1st class?",
             "male_not_first": "2nd class?", "female_not_first": "2nd class?"}
    for key, xy in pos.items():
        s0, s1 = TREE_NODES[key]["value"]
        rate = s1 / (s0 + s1)
        leaf = key not in heads
        face = ORANGE_LIGHT if (leaf and rate > 0.5) else (BLUE_LIGHT if leaf else "#ecebe6")
        edge = INK if key in ("female_first", "male_third") else None
        node_box(ax, xy, label(key, heads.get(key)), face, edge=edge, fontsize=8.3,
                 width=0.13, height=0.19 if not leaf else 0.16)
    ax.text(0.84, 0.245, "Rose (1, 0, 1)", ha="center", fontsize=9, weight="bold", va="center")
    ax.text(0.07, -0.05, "Jack (0, 0, 0)", ha="center", fontsize=9, weight="bold", va="center")
    ax.text(0.98, -0.05, "orange leaf: predicts survived", ha="right", fontsize=8.5, color=INK2)

    panel(fig.add_subplot(gs[0, 2]), "Reading the tree", [
        "Root split on sex lowers the impurity",
        f"0.467 → {gini(TREE_NODES['root']['value']) - root_gain:.3f} (weighted children).",
        "A leaf predicts its majority; proba = shares:",
        "`rose: [[0.0508, 0.9492]]  (3 vs 56)`",
        "`jack: [[0.8523, 0.1477]]  (202 vs 35)`",
        "",
        "Test set (295 passengers), as printed:",
        f"`accuracy {dt_acc:.4f}   ROC AUC {dt_auc:.4f}`",
        "",
        "Titanic is downloaded by seaborn, so the",
        "tree is drawn from the notebook's output.",
    ], spacing=0.075)

    ax = fig.add_subplot(gs[1, 0])
    xx, yy = np.meshgrid(np.linspace(1.8, 4.6, 300), np.linspace(0.6, 7.3, 300))
    grid_df = pd.DataFrame({"sepal_width": xx.ravel(), "petal_length": yy.ravel()})
    classes = list(iris_dt.classes_)
    zz = np.array([classes.index(c) for c in iris_dt.predict(grid_df)]).reshape(xx.shape)
    colors = [AQUA, BLUE, ORANGE]
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=[AQUA_LIGHT, BLUE_LIGHT, ORANGE_LIGHT], alpha=0.6)
    for k, species in enumerate(classes):
        tr = X_train[y_train == species]
        te = X_test[y_test == species]
        ax.scatter(tr.sepal_width, tr.petal_length, s=16, color=colors[k], alpha=0.8, label=species)
        ax.scatter(te.sepal_width, te.petal_length, s=34, facecolor="none", edgecolor=colors[k], linewidth=1.4)
    ax.set_xlabel("sepal_width")
    ax.set_ylabel("petal_length")
    ax.set_title("Iris decision tree: axis-aligned boxes")
    ax.legend(fontsize=9, loc="upper right", title="dots train, rings test", title_fontsize=8.5)

    ax = fig.add_subplot(gs[1, 1])
    names = ["decision tree", "random forest\n(100 trees)", "XGBoost\n(10 rounds)"]
    accs = [0.9, rf_acc, xg_acc]
    bar_colors = [BLUE, AQUA, ORANGE]
    pos_b = np.arange(3)
    ax.bar(pos_b, accs, width=0.6, color=bar_colors, zorder=2)
    for p, a in zip(pos_b, accs):
        ax.text(p, a + 0.004, f"{a:.2f}", ha="center", va="bottom", fontsize=11)
    ax.plot(np.full(len(rf_spread), 1.0) + np.linspace(-0.12, 0.12, len(rf_spread)), rf_spread, ".",
            color=INK2, zorder=3, label="forest, seeds 0-19")
    ax.set_xticks(pos_b, names)
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("test accuracy (50 flowers)")
    ax.grid(axis="y", zorder=0)
    ax.set_title("Ensembles beat one tree on the same split")
    ax.legend(fontsize=9, loc="upper left")
    ax.text(2, 0.83, "printed\n(xgboost not\ninstalled here)", ha="center", fontsize=8.5, color="white")

    panel(fig.add_subplot(gs[1, 2]), "Bagging vs boosting", [
        "Random forest: each tree sees a bootstrap",
        "sample and random features; the votes",
        "average out single-tree overfitting.",
        "",
        "Gradient boosting (XGBoost): trees are added",
        "in sequence, each fitting the remaining error.",
        "`eta 0.1, max_depth 2, 10 rounds`",
        "first test flower (versicolor), printed:",
        "`[0.148, 0.648, 0.204] -> class 1`",
        f"Tree 0.90 reproduced; forest 0.96 at seed {rf_seed}",
        "(the notebook sets no seed).",
    ], spacing=0.075)
    save(fig, name="7-algos-and-data-structures_seg3.png")


# ================================================================ notebook 8, Segment 2
def sheet_nb8_seg2_cost_and_critical_points():
    y_hat = np.array([7.0, 3.0])
    y_true = np.array([2.0, 8.0])
    delta = y_hat - y_true
    assert delta.tolist() == [5.0, -5.0]
    mean_delta = delta.mean()
    mae = np.abs(delta).mean()
    mse = np.mean(delta ** 2)
    assert mean_delta == 0 and np.abs(delta).sum() == 10 and mae == 5 and (delta ** 2).tolist() == [25, 25] and mse == 25

    delta_y = np.linspace(-5, 5, 1000)
    abs_error = np.abs(delta_y)
    sq_error = delta_y ** 2

    x = np.linspace(-10, 10, 1000)
    funcs = [
        ("Minimum", "y = x² + 3x + 4", x ** 2 + 3 * x + 4, lambda t: 2 * t + 3, 2.0, (-1.5, 1.75),
         (-5, 2), (-1, 15), "y″ = 2 > 0: curves up"),
        ("Maximum", "y = −x² + 3x + 4", -x ** 2 + 3 * x + 4, lambda t: -2 * t + 3, -2.0, (1.5, 6.25),
         (-2, 5), (-5, 10), "y″ = −2 < 0: curves down"),
        ("Saddle point", "y = x³ + 6", x ** 3 + 6, lambda t: 3 * t ** 2, 0.0, (0.0, 6.0),
         (-2, 2), (-1, 10), "y″ = 6x = 0: no turn"),
    ]
    for name, _, y, dy, d2, (cx, cy), *_rest in funcs:
        assert dy(cx) == 0
        assert np.isclose(np.interp(cx, x, y), cy, atol=0.02)
        numeric = np.gradient(y, x)
        # the sign change (or touch) of the numerical slope sits at the critical point, to grid precision
        k = np.argmin(np.abs(numeric))
        assert abs(x[k] - cx) < 0.03, name
    assert (-1.5) ** 2 + 3 * (-1.5) + 4 == 1.75 and -(1.5) ** 2 + 3 * 1.5 + 4 == 6.25

    fig = new_figure("Cost functions and critical points: square the errors, then find where the slope is zero",
                     "MSE  C = (1/n) Σ (ŷᵢ − yᵢ)² ≥ 0   ·   critical point: y′ = 0;   y″ > 0 minimum,   "
                     "y″ < 0 maximum,   y″ = 0 the test is silent (x³ + 6 is a saddle)")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    labels = ["Δy₁\n7 − 2", "Δy₂\n3 − 8", "mean\nΔy", "MAE", "MSE"]
    values = [delta[0], delta[1], mean_delta, mae, mse]
    colors = [BLUE_LIGHT, BLUE_LIGHT, MUTED, BLUE, ORANGE]
    positions = [0, 1, 2.6, 3.6, 4.6]
    ax.bar(positions, values, width=0.7, color=colors, zorder=2)
    for p, v in zip(positions, values):
        ax.text(p, v + (0.8 if v >= 0 else -0.8), f"{v:g}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=11)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.axvline(1.8, color=GRID, linewidth=1.2)
    ax.set_xticks(positions, labels, fontsize=9.5)
    ax.set_ylim(-9, 30)
    ax.grid(axis="y", zorder=0)
    ax.set_title("Raw differences cancel; |·| and (·)² do not")
    ax.text(2.6, 5.5, "0: looks\nperfect", ha="center", fontsize=9, color=INK2)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(delta_y, abs_error, color=BLUE, linewidth=2.2, label="absolute |Δy|")
    ax.plot(delta_y, sq_error, color=ORANGE, linewidth=2.2, label="squared Δy²")
    ax.axvspan(-1, 1, color=AQUA_LIGHT, alpha=0.35, label="|Δy| < 1: squared is smaller")
    ax.axhline(0, color="lightgray")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-1, 17)
    ax.grid(True)
    ax.set_xlabel("Δy = ŷ − y")
    ax.set_ylabel("error")
    ax.set_title("The notebook's plot: |Δy| vs Δy²")
    ax.legend(fontsize=9, loc="upper center")

    panel(fig.add_subplot(gs[0, 2]), "Why MSE is the default", [
        "Same two errors, three summaries:",
        "  Σ Δy = 5 − 5 = 0   (misleading)",
        "  MAE = (5 + 5) / 2 = 5",
        "  MSE = (25 + 25) / 2 = 25",
        "",
        "MSE is smooth, so ∇C exists everywhere and",
        "is cheap to compute: descend it to fit.",
        "Tolerant of small errors (|Δy| < 1 shrinks),",
        "harsh on large ones (they grow squared).",
        "Classifiers use cross-entropy instead.",
    ], spacing=0.08)

    layout = [  # (label offset, label ha, legend loc, note x, note y, note va)
        ((-10, -16), "right", "upper left", 0.5, 0.72, "top"),
        ((8, 10), "left", "lower left", 0.97, 0.96, "top"),
        ((8, -16), "left", "upper center", 0.97, 0.04, "bottom"),
    ]
    for col, (name, formula, y, dy, d2, (cx, cy), xlim, ylim, note) in enumerate(funcs):
        offset, label_ha, legend_loc, note_x, note_y, note_va = layout[col]
        ax = fig.add_subplot(gs[1, col])
        ax.axvline(0, c="lightgray")
        ax.axhline(0, c="lightgray")
        ax.plot(x, y, color=BLUE, linewidth=2.2, label=formula)
        ax.plot(x, dy(x), color=ORANGE, linewidth=1.6, linestyle="--", label="slope y′")
        ax.axhline(cy, color=ORANGE, linestyle=":", linewidth=1.2)
        ax.scatter([cx], [cy], color=ORANGE, s=60, zorder=5)
        ax.scatter([cx], [0], color=ORANGE, s=30, zorder=5, marker="x")
        ax.annotate(f"({cx:g}, {cy:g})", xy=(cx, cy), xytext=offset, ha=label_ha,
                    textcoords="offset points", fontsize=10, weight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_title(f"{name}: {formula}")
        ax.legend(fontsize=9, loc=legend_loc)
        ax.text(note_x, note_y, note, transform=ax.transAxes, ha="center" if note_x == 0.5 else "right",
                va=note_va, fontsize=9.5, color=INK2)
    save(fig, name="8-optimization_seg2.png")


# ================================================================ notebook 8, Segment 3
def sheet_nb8_seg3_jacobian_hessian():
    from torch.autograd.functional import hessian, jacobian

    # Jacobian of a dense layer with the sizes the notebook names (m = 784 inputs, n = 128 outputs).
    # Weights are illustrative (seeded); the notebook itself only states the shapes.
    torch.manual_seed(42)
    m, n = 784, 128
    W = torch.randn(n, m, dtype=torch.float64) * 0.05
    bias = torch.zeros(n, dtype=torch.float64)
    x0 = torch.rand(m, dtype=torch.float64)

    def layer(x):
        return torch.sigmoid(W @ x + bias)

    J = jacobian(layer, x0)
    assert tuple(J.shape) == (128, 784)
    z = W @ x0 + bias
    s = torch.sigmoid(z)
    assert torch.allclose(J, (s * (1 - s))[:, None] * W)              # J = diag(σ'(z)) W
    j_col = 3
    h = 1e-6
    e = torch.zeros(m, dtype=torch.float64)
    e[j_col] = h
    fd = (layer(x0 + e) - layer(x0 - e)) / (2 * h)
    assert torch.allclose(fd, J[:, j_col], atol=1e-9)

    # Hessians of the Segment 2 functions and of a two-variable saddle
    def f_min(t):
        return t ** 2 + 3 * t + 4

    def f_max(t):
        return -t ** 2 + 3 * t + 4

    def f_sp(t):
        return t ** 3 + 6

    h_vals = [hessian(f, torch.tensor(c, dtype=torch.float64)).item()
              for f, c in [(f_min, -1.5), (f_max, 1.5), (f_sp, 0.0)]]
    assert h_vals == [2.0, -2.0, 0.0]

    def saddle(v):
        return v[0] ** 2 - v[1] ** 2

    p0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
    H2 = hessian(saddle, p0)
    H_from_jac = jacobian(lambda v: torch.autograd.grad(saddle(v), v, create_graph=True)[0],
                          p0.clone().requires_grad_(True))
    assert torch.allclose(H2, torch.tensor([[2.0, 0.0], [0.0, -2.0]], dtype=torch.float64))
    assert torch.allclose(H2, H_from_jac)                              # Hessian = Jacobian of the gradient
    eigvals = np.linalg.eigvalsh(H2.numpy())
    assert eigvals.tolist() == [-2.0, 2.0]

    # first-order vs second-order steps on y = x^2 + 3x + 4 from x = 4
    lr = 0.1
    gd = [4.0]
    for _ in range(12):
        gd.append(gd[-1] - lr * (2 * gd[-1] + 3))
    newton = [4.0, 4.0 - (2 * 4.0 + 3) / 2.0]
    assert newton[1] == -1.5
    assert abs(gd[-1] + 1.5) > 0.3 and abs(gd[-1] + 1.5) < 0.5  # 0.8^12 * 5.5 ≈ 0.38 still to go

    fig = new_figure("Jacobian and Hessian: every first derivative of a vector output, every second derivative",
                     "f: ℝⁿ→ℝ has gradient ∇f (n entries)   ·   f: ℝᵐ→ℝⁿ has Jacobian Jᵢⱼ = ∂fᵢ/∂xⱼ (n × m)   ·   "
                     "Hessian Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ = Jacobian of ∇f")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    Jn = J.numpy()
    vmax = np.abs(Jn).max()
    image = ax.imshow(Jn, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xlabel("input j (784 pixels)")
    ax.set_ylabel("output i (128 neurons)")
    ax.set_title("Jᵢⱼ = ∂aᵢ/∂xⱼ of a 784 → 128 layer")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    rows = np.arange(n)
    ax.plot(rows, J[:, j_col].numpy(), color=BLUE, linewidth=2.4, label="autograd column J[:, 3]")
    ax.plot(rows, fd.numpy(), "o", color=ORANGE, markersize=3, label="finite difference, h = 10⁻⁶")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.grid(True)
    ax.set_xlabel("output i")
    ax.set_ylabel("∂aᵢ / ∂x₃")
    ax.set_title("One column = nudge one input, watch all outputs")
    ax.legend(fontsize=9, loc="lower left")

    panel(fig.add_subplot(gs[0, 2]), "Shapes to remember", [
        "gradient   f: ℝⁿ → ℝ     ∇f  has n entries",
        "Jacobian   f: ℝᵐ → ℝⁿ    J  is n × m",
        "Hessian    f: ℝⁿ → ℝ     H  is n × n, symmetric",
        "",
        "Dense layer a = σ(Wx + b):  J = diag(σ′(z)) W",
        "checked against torch autograd:",
        "`jacobian(layer, x).shape -> (128, 784)`",
        "Generalized Jacobian: tensors in and out.",
        "",
        "Weights here are random (seed 42): the notebook",
        "names only the sizes m = 784, n = 128.",
    ], spacing=0.075)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-3, 3, 200)
    ax.plot(xs, np.full_like(xs, 2.0), color=BLUE, linewidth=2.2, label="x² + 3x + 4:  y″ = 2")
    ax.plot(xs, np.full_like(xs, -2.0), color=ORANGE, linewidth=2.2, label="−x² + 3x + 4:  y″ = −2")
    ax.plot(xs, 6 * xs, color=AQUA, linewidth=2.2, label="x³ + 6:  y″ = 6x")
    for cx, hv, color in [(-1.5, 2.0, BLUE), (1.5, -2.0, ORANGE), (0.0, 0.0, AQUA)]:
        ax.plot(cx, hv, "o", color=color, markersize=8, zorder=5)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylim(-8, 8)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("1 × 1 Hessian  y″")
    ax.set_title("Segment 2's critical points, by the Hessian")
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.97, 0.04, "dots: value at each critical point\n+ min,  − max,  0 undecided",
            transform=ax.transAxes, ha="right", fontsize=9, color=INK2)

    ax = fig.add_subplot(gs[1, 1])
    g = np.linspace(-2, 2, 200)
    GX, GY = np.meshgrid(g, g)
    levels = np.linspace(-4, 4, 17)
    cs = ax.contourf(GX, GY, GX ** 2 - GY ** 2, levels=levels, cmap=DIVERGING)
    ax.contour(GX, GY, GX ** 2 - GY ** 2, levels=[0], colors=[INK2], linewidths=1)
    ax.annotate("", xy=(1.3, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=2.2, mutation_scale=16))
    ax.annotate("", xy=(0, 1.3), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=2.2, mutation_scale=16))
    ax.text(0.15, -0.4, "λ = +2: curves up", fontsize=9.5, ha="left",
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.7, pad=1))
    ax.text(0.1, 1.45, "λ = −2: curves down", fontsize=9.5,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.7, pad=1))
    ax.plot(0, 0, "o", color=INK, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("f = x² − y²: H = [[2, 0], [0, −2]] → saddle")
    cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("f(x, y)")
    cbar.outline.set_visible(False)

    ax = fig.add_subplot(gs[1, 2])
    curve_x = np.linspace(-5, 5, 300)
    ax.plot(curve_x, f_min(curve_x), color=MUTED, linewidth=1.5, label="y = x² + 3x + 4")
    gd_arr = np.array(gd)
    ax.plot(gd_arr, f_min(gd_arr), "o-", color=BLUE, markersize=5, linewidth=1.4,
            label=f"gradient descent, η = {lr}: 12 steps")
    ax.plot(newton, [f_min(t) for t in newton], "s-", color=ORANGE, markersize=7, linewidth=2,
            label="Newton x − y′/y″: 1 step")
    ax.plot(-1.5, 1.75, "*", color=INK, markersize=14, zorder=5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 36)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Why optimizers want curvature")
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.97, 0.04, f"after 12 steps GD is still\n{abs(gd[-1] + 1.5):.2f} from x = −1.5",
            transform=ax.transAxes, ha="right", fontsize=9, color=INK2)
    save(fig, name="8-optimization_seg3.png")


if __name__ == "__main__":
    sheet_nb7_seg1_big_o()
    sheet_nb7_seg2_lists_search_sort_hash()
    sheet_nb7_seg3_trees()
    sheet_nb8_seg2_cost_and_critical_points()
    sheet_nb8_seg3_jacobian_hessian()
