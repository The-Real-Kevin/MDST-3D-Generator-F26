"""Check that an STL is watertight and printable.  usage: python scripts/check_stl.py data/gallery/city_001_x.stl"""
import sys
from collections import Counter
import numpy as np

def check(path):
    raw = open(path, "rb").read()
    n = int.from_bytes(raw[80:84], "little")
    rec = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    tri = np.frombuffer(raw, dtype=rec, count=n, offset=84)["v"].astype(np.float64)
    assert len(raw) == 84 + 50 * n, "file length doesn't match triangle count"
    pts, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    T = inv.reshape(-1, 3)
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    directed = Counter()
    for a, b, c in T:
        for e in ((a, b), (b, c), (c, a)):
            directed[e] += 1
    bad_dup = sum(1 for k, v in directed.items() if v != 1)
    unpaired = sum(1 for (a, b) in directed if (b, a) not in directed)
    # shells via union-find on vertices
    parent = list(range(len(pts)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b, c in T:
        ra = find(a)
        for o in (b, c):
            ro = find(o)
            if ro != ra: parent[ro] = ra
    shell = np.array([find(t[0]) for t in T])
    vols = Counter(); faces = Counter(); verts = {}
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6
    for s, v in zip(shell, signed): vols[s] += v; faces[s] += 1
    for vi in range(len(pts)): verts.setdefault(find(vi), set()).add(vi)
    euler_bad = 0
    for s in faces:
        F = faces[s]; E = F * 3 // 2; V = len(verts[s])
        if V - E + F != 2: euler_bad += 1
    neg = sum(1 for v in vols.values() if v <= 0)
    print(f"{path}: {n} triangles, {len(faces)} shells")
    print(f"  edges used !=1 time in same direction: {bad_dup}")
    print(f"  edges without an opposite partner:      {unpaired}")
    print(f"  shells failing Euler V-E+F=2:          {euler_bad}")
    print(f"  shells with non-positive volume:       {neg}")
    print(f"  zero-area triangles:                   {(area < 1e-9).sum()}")
    ok = bad_dup == unpaired == euler_bad == neg == 0 and (area < 1e-9).sum() == 0
    print("  WATERTIGHT, outward-facing" if ok else "  PROBLEMS FOUND")
    return ok

if __name__ == "__main__":
    sys.exit(0 if all(check(p) for p in sys.argv[1:]) else 1)
