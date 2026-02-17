import math
from typing import List, Tuple, Optional, Dict, Any
import pygame

Point = Tuple[float,float]

def world_to_screen(x: float, y: float, origin: Point, scale: float) -> Tuple[int,int]:
    ox, oy = origin
    sx = int(ox + x*scale)
    sy = int(oy - y*scale)
    return sx, sy

def compute_view(track_pts: List[Point], w: int, h: int, padding: int = 40):
    xs = [p[0] for p in track_pts]
    ys = [p[1] for p in track_pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = max(1e-6, maxx - minx)
    spany = max(1e-6, maxy - miny)
    scalex = (w - 2*padding) / spanx
    scaley = (h - 2*padding) / spany
    scale = min(scalex, scaley)

    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    origin = (w/2 - cx*scale, h/2 + cy*scale)
    return origin, scale

def draw_track(screen, centerline: List[Point], width_m: float, origin: Point, scale: float):
    pts = [world_to_screen(x,y,origin,scale) for x,y in centerline]
    if len(pts) >= 2:
        road_px = max(3, int(width_m * scale))
        pygame.draw.lines(screen, (80,80,80), False, pts, road_px)
        pygame.draw.lines(screen, (240,240,240), False, pts, 2)

def draw_path(screen, path: List[Point], origin: Point, scale: float, color=(30,200,255), width=2):
    if not path or len(path) < 2:
        return
    pts = [world_to_screen(x,y,origin,scale) for x,y in path]
    pygame.draw.lines(screen, color, False, pts, width)

def draw_car(screen, x: float, y: float, yaw: float, origin: Point, scale: float):
    px, py = world_to_screen(x,y,origin,scale)
    L = 3.6 * scale
    W = 1.8 * scale

    fx = math.cos(yaw)
    fy = -math.sin(yaw)  # y is inverted on screen
    rx, ry = -fy, fx

    p1 = (px + fx*L*0.6, py + fy*L*0.6)
    p2 = (px - fx*L*0.4 + rx*W*0.4, py - fy*L*0.4 + ry*W*0.4)
    p3 = (px - fx*L*0.4 - rx*W*0.4, py - fy*L*0.4 - ry*W*0.4)

    pygame.draw.polygon(screen, (255,180,60), [p1,p2,p3])

def draw_text(screen, lines: List[str], x: int = 10, y: int = 10):
    font = pygame.font.SysFont("consolas", 18)
    yy = y
    for line in lines:
        surf = font.render(line, True, (240,240,240))
        screen.blit(surf, (x, yy))
        yy += 20

def run_replay(track, replay_path: List[Point], replay_goals: Optional[List[Point]] = None, meta: Optional[Dict[str,Any]] = None,
               window=(1100, 700), fps=60):
    pygame.init()
    screen = pygame.display.set_mode(window)
    pygame.display.set_caption("Autonomno vozilo - replay")
    clock = pygame.time.Clock()

    origin, scale = compute_view(track.centerline, window[0], window[1], padding=60)

    i = 0
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    i = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not paused and replay_path:
            i = min(len(replay_path)-1, i+1)

        screen.fill((18,18,22))
        draw_track(screen, track.centerline, track.width, origin, scale)

        draw_path(screen, replay_path[:max(2,i)], origin, scale, color=(30,200,255), width=3)

        if replay_goals and i < len(replay_goals):
            gx, gy = replay_goals[i]
            gsx, gsy = world_to_screen(gx, gy, origin, scale)
            pygame.draw.circle(screen, (120,255,120), (gsx,gsy), 6)

        if replay_path:
            x, y = replay_path[i]
            if i < len(replay_path)-1:
                x2, y2 = replay_path[i+1]
            else:
                x2, y2 = replay_path[i]
            yaw = math.atan2(y2-y, x2-x) if (x2!=x or y2!=y) else 0.0
            draw_car(screen, x, y, yaw, origin, scale)

        lines = [
            "SPACE: pause | R: restart | ESC: exit",
            f"frame: {i}/{max(1,len(replay_path)-1)}"
        ]
        if meta:
            for k in ("best_J","t","mean_cte","offroad_time","steer_jerk","reached"):
                if k in meta:
                    lines.append(f"{k}: {meta[k]}")
        draw_text(screen, lines, 10, 10)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
