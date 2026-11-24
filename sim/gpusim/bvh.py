"""Flat BVH over a triangle soup (built on CPU with numpy, median split), for the CUDA ray kernels.

Layout (all int32/float32, uploaded once per map):
  node_lo[N,3], node_hi[N,3]   AABB per node
  node_left[N]                 child index (right child = left+1) for internal nodes, -1 for leaves
  node_start[N], node_count[N] triangle range in `order` for leaves
  tri[T,3,3] reordered triangles (world coords), tri_id[T] original triangle id (materials, tie-breaking)
"""
import numpy as np


def build_bvh(tri, leaf=4):
    tri = np.asarray(tri, np.float64)
    T = len(tri)
    c = tri.mean(1); lo_t = tri.min(1); hi_t = tri.max(1)
    order = np.arange(T)
    nodes = []                                     # (lo, hi, left, start, count)
    stack = [(0, T, -1, None)]                     # (start, end, parent, side)

    def make(start, end):
        idx = order[start:end]
        return lo_t[idx].min(0), hi_t[idx].max(0)

    # iterative build with explicit child placement (left child at k, right at k+1)
    nodes.append(None)
    work = [(0, 0, T)]
    while work:
        ni, s, e = work.pop()
        lo, hi = make(s, e)
        if e - s <= leaf:
            nodes[ni] = (lo, hi, -1, s, e - s); continue
        idx = order[s:e]
        ext = c[idx].max(0) - c[idx].min(0)
        ax = int(np.argmax(ext))
        if ext[ax] <= 0:
            nodes[ni] = (lo, hi, -1, s, e - s); continue
        k = np.argsort(c[idx, ax], kind='stable')
        order[s:e] = idx[k]
        m = (s + e) // 2
        li = len(nodes); nodes.append(None); nodes.append(None)
        nodes[ni] = (lo, hi, li, -1, 0)
        work.append((li, s, m)); work.append((li + 1, m, e))
    N = len(nodes)
    out = dict(
        node_lo=np.array([n[0] for n in nodes], np.float32),
        node_hi=np.array([n[1] for n in nodes], np.float32),
        node_left=np.array([n[2] for n in nodes], np.int32),
        node_start=np.array([n[3] for n in nodes], np.int32),
        node_count=np.array([n[4] for n in nodes], np.int32),
        tri=tri[order].astype(np.float32),
        tri_id=order.astype(np.int32))
    return out
