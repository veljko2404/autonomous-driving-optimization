import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

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
    ang = np.linspace(0, 2*math.pi, n, endpoint=False)
    xs = r*np.cos(ang)
    ys = r*np.sin(ang)
    center = list(zip(xs.tolist(), ys.tolist()))
    return Track(centerline=center, width=6.0)

def polyline_segments(points: List[Point]):
    for i in range(len(points)-1):
        yield points[i], points[i+1]

def closest_point_on_segment(p: Point, a: Point, b: Point) -> Tuple[Point, float]:
    """
    Računa najbližu tačku sa dužine AB na tačku P.
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
