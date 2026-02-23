import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import random

"""
Modul za generisanje i obradu geometrije staze.

Konvencije:
- Point = (x, y) — tuple od dva float-a
- centerline: lista Point tacaka koje predstavljaju srednju liniju staze redom
- width: širina staze (float), koristi se samo kao meta-podatak ovde
- Većina funkcija radi u 2D euclid prostoru (metrički sistem proizvoljan)
- Centerline se tretira kao polilinija (niz segmenata između susednih tacaka).
"""

Point = Tuple[float, float]

@dataclass
class Track:
    """
    Jednostavan tip podatka koji enkapsulira srednju liniju i širinu.
    Korisno za prosleđivanje staze u druge module (simulacija, vizuelizacija, kontroleri).
    """
    centerline: List[Point]
    width: float


def make_s_track(n: int = 350, scale: float = 1.0) -> Track:
    """
    Generiše S-oblik staze koristeći sinusne funkcije.
    - n: broj uzoraka duž centerline (veći n -> glatkija linija)
    - scale: skalirni faktor koji skalira sve dimenzije (dužinu i amplitude)
    Povratna vrednost: Track sa `centerline` kao listom Point i width = 6.0*scale.

    Napomene:
    - xs su ravnomerno raspoređeni u opsegu [0, 120*scale]
    - ys kombinacija dve sinusoide različitih frekvencija -> S-oblik sa varijacijama
    - Koristi se za testiranje ponašanja vozila u krivinama različitih skala.
    """
    xs = np.linspace(0, 120*scale, n)
    ys = 8*scale*np.sin(xs/(12*scale)) + 3*scale*np.sin(xs/(4*scale))
    center = list(zip(xs.tolist(), ys.tolist()))
    return Track(centerline=center, width=6.0*scale)


def make_circle(r: float = 35.0, n: int = 420) -> Track:
    """
    Generiše kružnu stazu:
    - r: poluprečnik kruga
    - n: broj tačaka (što veći n, to bolje aproksimira krug polilinijom)
    Povratna vrednost: Track(centerline, width=6.0)

    Napomene:
    - Tačke su raspoređene uniformno po uglu [0, 2π)
    - Rezultat je zatvorena staza (poslednja tačka ne ponavlja prvu; kod koji koristi stazu
      treba znati da je centerline cikličan ako je potrebno).
    """
    ang = np.linspace(0, 2*math.pi, n, endpoint=False)
    xs = r*np.cos(ang)
    ys = r*np.sin(ang)
    center = list(zip(xs.tolist(), ys.tolist()))
    return Track(centerline=center, width=6.0)


def make_random_track(n: int = 420,
                      scale: float = 1.0,
                      seed: int | None = None,
                      control_points: int = 8,
                      radius: float = 35.0,
                      jitter: float = 10.0,
                      closed: bool = True) -> Track:
    """
    Generiše "nasumičnu" stazu koristeći kontrolne tačke i Catmull-Rom interpolaciju.
    Parametri:
      - n: ukupno uzoraka/tačaka na izlaznoj centerline
      - scale: skaliranje svih veličina
      - seed: (opciono) seed za reproducibilnost
      - control_points: broj kontrolnih tačaka koje određuju oblik staze
      - radius: tipična udaljenost kontrolnih tačaka od centra (za zatvorenu stazu)
      - jitter: maksimalna devijacija od radius-a (ili y-odstupanje za otvorene staze)
      - closed: ako True, pravi zatvorenu stazu (petlja); inače pravi open-track

    Logika:
      - Ako closed: rasporedi `control_points` ravnomerno po uglu i daj im heterogen radijus
        (radius +/- jitter). To daje blago nasumične kružnice.
      - Ako not closed: rasporedi control_points duž x ose i daj nasumični y jitter.
      - Zatim upotrebiti catmull_rom_chain da dobijemo glatku centerline od kontrolnih tačaka.

    Napomene:
      - Više control_points -> kompleksniji oblik; veći jitter -> krivudavija staza.
      - Seed postavlja i `random` i `numpy` generator radi determinističnosti.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if closed:
        # generiši kontrolne tačke raspoređene po uglu, sa variranjem radijusa
        angles = np.linspace(0, 2*math.pi, control_points, endpoint=False)
        ctrl = []
        for a in angles:
            r = radius*scale + random.uniform(-jitter, jitter)*scale
            x = r * math.cos(a)
            y = r * math.sin(a)
            ctrl.append((x, y))
    else:
        # otvorena staza: kontrolne tačke duž x sa nasumičnim y odstupanjem
        xs = np.linspace(0, radius*2*scale, control_points)
        ctrl = []
        for x in xs:
            y = random.uniform(-jitter, jitter) * scale
            ctrl.append((x, y))

    # Interpoliraj kontrolne tačke Catmull-Rom spline-om u n_points tačaka
    center = catmull_rom_chain(ctrl, n_points=n, closed=closed)
    return Track(centerline=center, width=6.0*scale)


def polyline_segments(points: List[Point]):
    """
    Generator koji iterira kroz susedne tačke i vraća segmente (a, b).
    - Ulaz: lista tačaka [p0, p1, p2, ...]
    - Yield: (p0, p1), (p1, p2), ...
    Koristi se za pretragu najbližih segmenata i za izračunavanje CTE.
    """
    for i in range(len(points)-1):
        yield points[i], points[i+1]


def closest_point_on_segment(p: Point, a: Point, b: Point) -> Tuple[Point, float]:
    """
    Projekcija tačke p na segment AB (najbliža tačka na segmentu).
    Vraća tuple (closest_point, t) gde je t parametar projekcije unutar [0,1]:
      - t=0 => najbliže u A
      - t=1 => najbliže u B
    Algoritam:
      - računamo projekciju vektora AP na vektor AB,
      - normalizujemo t u interval [0,1], i vraćamo koordinatu.

    Edge-case:
      - Ako su A i B isti (segment dužine 0), vraća A i t=0.
    Kompleksnost: O(1) po segmentu.
    """
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = (bx-ax), (by-ay)       # vektor AB
    wx, wy = (px-ax), (py-ay)       # vektor AP
    vv = vx*vx + vy*vy
    if vv == 0:
        # degenerate segment: vrati A
        return (a, 0.0)
    # parametar t = proj(AP) na AB / |AB|^2
    t = (wx*vx + wy*vy) / vv
    # klipujemo na segment
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t*vx, ay + t*vy
    return ( (cx, cy), t )


def distance_to_centerline(p: Point, centerline: List[Point]) -> Tuple[float, int, Point]:
    """
    Pronalaženje minimalne udaljenosti tačke p do polilinije centerline.
    Vraća:
      - distance: minimalna euklidska udaljenost (float)
      - best_i: indeks segmenta (a,b) na kojem je najbliža tačka (vraća i takav da segment=(centerline[i], centerline[i+1]))
      - best_cp: koordinate najbliže tačke na segmentu (Point)

    Implementacija:
      - iterira kroz sve segmente (O(m) gde je m = len(centerline)-1)
      - za svaki segment koristi closest_point_on_segment i računa kvadratnu udaljenost
      - pamti najmanju i vraća sqrt na kraju

    Napomene:
      - Ako centerline ima < 2 tačke, kod van funkcije bi trebao obratiti pažnju; ovde pretpostavljamo validan centerline.
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
    Pronalazi lookahead tačku koja je na distance Ld **uzduž** centerline počevši
    od segmenta sa indeksom start_seg_idx (ili najbližeg segmenta).
    Povratna vrednost: Point (x,y) na centerline.

    Algoritam:
      - kreće od tačke centerline[start_seg_idx] i sledi segmente sve dok
        akumulirana dužina < Ld ili dok ne dođe do kraja centerline.
      - vraća poslednju posećenu centerline tačku (aproksimacija: ne interpolira tačno
        na segmentu; za veću preciznost se može dodati linearna interpolacija po preostalom delu segmenta).

    Napomene:
      - Ako Ld je veće od preostale dužine staze, funkcija vraća poslednju tačku.
      - start_seg_idx je klipovan u validni indeks [0, len(centerline)-2].
    """
    idx = max(0, min(len(centerline)-2, start_seg_idx))
    dist = 0.0
    cur = centerline[idx]
    i = idx
    # prelazimo po tačkama centerline i sumiramo dužine segmenata
    while i < len(centerline)-1 and dist < Ld:
        nxt = centerline[i+1]
        seg = math.hypot(nxt[0]-cur[0], nxt[1]-cur[1])
        dist += seg
        cur = nxt
        i += 1
    return cur


def finish_line_point(centerline: List[Point]) -> Point:
    """
    Jednostavan helper koji vraća "cilj" staze: poslednju tačku centerline.
    Može se koristiti kao referentna tačka za završetak kruga ili za merenje pređenog puta.
    """
    return centerline[-1]


# ---------------------------------------------------------------------
# Catmull-Rom spline (lokalna hermitska interpolacija) implementacija
# ---------------------------------------------------------------------
def _catmull_rom_one_segment(p0, p1, p2, p3, n_points):
    """
    Izračunava n_points izlomaka Catmull–Rom kurbe između p1 i p2 koristeći p0 i p3
    kao susedne kontrolne tačke. Vraća listu Point tačaka.
    - p0,p1,p2,p3: kontrolne tačke (x,y)
    - n_points: koliko uzoraka vratiti za ovaj segment (obično >=1)
    Napomena:
    - Koristimo standardne koeficijente Catmull-Rom sa parametrizacijom tension=0.5.
    - endpoint=False u linspace-u znači da ne uključujemo tačku p2 (to se rešava u glavnoj petlji gde se segmente spajaju).
    """
    t = np.linspace(0, 1, n_points, endpoint=False)
    t2 = t * t
    t3 = t2 * t

    # koeficijenti polinoma za x koordinatu
    a0 = -0.5*p0[0] + 1.5*p1[0] - 1.5*p2[0] + 0.5*p3[0]
    a1 = p0[0] - 2.5*p1[0] + 2*p2[0] - 0.5*p3[0]
    a2 = -0.5*p0[0] + 0.5*p2[0]
    a3 = p1[0]

    # isti koeficijenti za y
    b0 = -0.5*p0[1] + 1.5*p1[1] - 1.5*p2[1] + 0.5*p3[1]
    b1 = p0[1] - 2.5*p1[1] + 2*p2[1] - 0.5*p3[1]
    b2 = -0.5*p0[1] + 0.5*p2[1]
    b3 = p1[1]

    xs = a0*t3 + a1*t2 + a2*t + a3
    ys = b0*t3 + b1*t2 + b2*t + b3
    return list(zip(xs.tolist(), ys.tolist()))


def catmull_rom_chain(points, n_points=400, closed=True):
    """
    Izračunava Catmull-Rom interpoliranu krivulju kroz zadate kontrolne tačke.
    Parametri:
      - points: lista kontrolnih tačaka (barem 2)
      - n_points: željeni broj izlaznih tačaka (aproksimacija ukupne dužine)
      - closed: ako True, tretira listu kao zatvorenu petlju

    Algoritam:
      - Pripremi niz pts koji eksplicitno uključuje dodatne tačke na početku i kraju
        tako da svaki segment ima definisan p0,p1,p2,p3.
      - Broj segmenata = len(pts) - 3.
      - Podeli totalni n_points ravnomerno po segmentima (per_seg).
      - Za svaki segment pozovi _catmull_rom_one_segment i priključi uzorke.
      - Na kraju isečeš listu na tačno n_points (ako je višak).

    Edge-cases i napomene:
      - Ako ima manje od 2 kontrolne tačke, vraća originalnu listu (nema smisla interpolirati).
      - Za closed=True: umetnemo poslednje dve i prve dve tačke da bi spline "wrap-ovao".
      - Per-seg podela koristi celo deljenje; zbog toga neka mala nesrazmera može nastati,
        pa se krajnji rezultat sekvencijalno trimuje do n_points.
      - Kompliciranije varijante bi koristile adaptivno uzorkovanje po dužini segmenta, ali
        za naše svrhe jednolika podela po segmentima je jednostavna i dovoljna.
    """
    if len(points) < 2:
        return points[:]

    pts = points[:]
    if closed:
        # za zatvorenu petlju: dodaj zadnje dve na početak i prve dve na kraj
        pts = [points[-2], points[-1]] + points + [points[0], points[1]]
    else:
        # za otvorenu krivu: dupliciraj krajnje kontrolne tačke radi "natural" ponašanja
        pts = [points[0]] + points + [points[-1]]

    segments = len(pts) - 3
    per_seg = max(1, n_points // segments)  # broj uzoraka po segmentu (osigurava >=1)

    curve = []
    for i in range(segments):
        p0, p1, p2, p3 = pts[i], pts[i+1], pts[i+2], pts[i+3]
        curve.extend(_catmull_rom_one_segment(p0, p1, p2, p3, per_seg))

    # trimuj na tačno n_points (ako je rezultat duži)
    return curve[:n_points]