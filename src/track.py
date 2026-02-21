import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import random

"""
Modul za generisanje i obradu geometrije staze.
Staza je definisana kao centerline (lista 2D tacaka)
i koristi se za racunanje bocne grecke (CTE)
i lookahead tacke za Pure Pursuit kontroler.
"""

Point = Tuple[float, float]

@dataclass
class Track:
    centerline: List[Point]
    width: float

def make_s_track(n: int = 350, scale: float = 1.0) -> Track:
    """
    Generise S-oblik staze koriscenjem sinusnih funkcija.
    Sluzi za testiranje ponasanja vozila u krivinama.
    """
    xs = np.linspace(0, 120*scale, n)
    ys = 8*scale*np.sin(xs/(12*scale)) + 3*scale*np.sin(xs/(4*scale))
    center = list(zip(xs.tolist(), ys.tolist()))
    return Track(centerline=center, width=6.0*scale)

def make_circle(r: float = 35.0, n: int = 420) -> Track:
    """
    Generise kruznu stazu sa poluprecnikom r.
    Staza se koristi za testiranje stabilnosti kretanja vozila
    pri konstantnom zakrivljenju i zatvorenoj putanji.
    """
    ang = np.linspace(0, 2*math.pi, n, endpoint=False)
    xs = r*np.cos(ang)
    ys = r*np.sin(ang)
    center = list(zip(xs.tolist(), ys.tolist()))
    return Track(centerline=center, width=6.0)

# Ako hocemo krivudavu stazu, treba da povecamo control_points i jitter
# Ako hocemo blagu stazu, smanji jitter
def make_random_track(n: int = 420,
                      scale: float = 1.0,
                      seed: int | None = None,
                      control_points: int = 8,
                      radius: float = 35.0,
                      jitter: float = 10.0,
                      closed: bool = True) -> Track:

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if closed:
        angles = np.linspace(0, 2*math.pi, control_points, endpoint=False)
        ctrl = []
        for a in angles:
            r = radius*scale + random.uniform(-jitter, jitter)*scale
            x = r * math.cos(a)
            y = r * math.sin(a)
            ctrl.append((x, y))
    else:
        xs = np.linspace(0, radius*2*scale, control_points)
        ctrl = []
        for x in xs:
            y = random.uniform(-jitter, jitter) * scale
            ctrl.append((x, y))

    center = catmull_rom_chain(ctrl, n_points=n, closed=closed)
    return Track(centerline=center, width=6.0*scale)

def polyline_segments(points: List[Point]):
    """
    Razbija poliliniju (centerline staze) na uzastopne segmente.
    Svaki segment je par susednih tacaka i koristi se za
    racunanje udaljenosti vozila od staze.
    """
    for i in range(len(points)-1):
        yield points[i], points[i+1]

def closest_point_on_segment(p: Point, a: Point, b: Point) -> Tuple[Point, float]:
    """
    Racuna najbližu tacku sa dužine AB na tacku P.
    Koristi se za projekciju vozila na centerline staze.
    """
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = (bx-ax), (by-ay)
    wx, wy = (px-ax), (py-ay)
    vv = vx*vx + vy*vy
    if vv == 0:
        return (a, 0.0)
    t = (wx*vx + wy*vy) / vv
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t*vx, ay + t*vy
    return ( (cx, cy), t )

def distance_to_centerline(p: Point, centerline: List[Point]) -> Tuple[float, int, Point]:
    """
    Racuna minimalno rastojanje vozila od centerline staze (CTE).
    Iterira kroz sve segmente staze i traži najblizu tacku.
    """
    best_d2 = float("inf")
    best_i = 0
    best_cp = centerline[0]
    for i, (a,b) in enumerate(polyline_segments(centerline)):
        cp, _ = closest_point_on_segment(p, a, b)
        dx = p[0]-cp[0]
        dy = p[1]-cp[1]
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
            best_cp = cp
    return math.sqrt(best_d2), best_i, best_cp

def find_lookahead_goal(p: Point, centerline: List[Point], start_seg_idx: int, Ld: float) -> Point:
    """
    Pronalazi lookahead tacku za Pure Pursuit kontroler.
    Tacka se nalazi unapred duz centerline na udaljenosti Ld.
    """
    idx = max(0, min(len(centerline)-2, start_seg_idx))
    dist = 0.0
    cur = centerline[idx]
    i = idx
    while i < len(centerline)-1 and dist < Ld:
        nxt = centerline[i+1]
        seg = math.hypot(nxt[0]-cur[0], nxt[1]-cur[1])
        dist += seg
        cur = nxt
        i += 1
    return cur

def finish_line_point(centerline: List[Point]) -> Point:
    return centerline[-1]

# Catmull-Rom spline
def _catmull_rom_one_segment(p0, p1, p2, p3, n_points):
    t = np.linspace(0, 1, n_points, endpoint=False)
    t2 = t * t
    t3 = t2 * t

    a0 = -0.5*p0[0] + 1.5*p1[0] - 1.5*p2[0] + 0.5*p3[0]
    a1 = p0[0] - 2.5*p1[0] + 2*p2[0] - 0.5*p3[0]
    a2 = -0.5*p0[0] + 0.5*p2[0]
    a3 = p1[0]

    b0 = -0.5*p0[1] + 1.5*p1[1] - 1.5*p2[1] + 0.5*p3[1]
    b1 = p0[1] - 2.5*p1[1] + 2*p2[1] - 0.5*p3[1]
    b2 = -0.5*p0[1] + 0.5*p2[1]
    b3 = p1[1]

    xs = a0*t3 + a1*t2 + a2*t + a3
    ys = b0*t3 + b1*t2 + b2*t + b3
    return list(zip(xs.tolist(), ys.tolist()))


def catmull_rom_chain(points, n_points=400, closed=True):
    if len(points) < 2:
        return points[:]

    pts = points[:]
    if closed:
        pts = [points[-2], points[-1]] + points + [points[0], points[1]]
    else:
        pts = [points[0]] + points + [points[-1]]

    segments = len(pts) - 3
    per_seg = max(1, n_points // segments)

    curve = []
    for i in range(segments):
        p0, p1, p2, p3 = pts[i], pts[i+1], pts[i+2], pts[i+3]
        curve.extend(_catmull_rom_one_segment(p0, p1, p2, p3, per_seg))

    return curve[:n_points]
