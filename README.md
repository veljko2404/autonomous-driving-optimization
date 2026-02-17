# Autonomno vozilo (2D) + numerička optimizacija (bez RL)

**Šta dobijaš:**
- 2D simulator (kinematički bicycle model)
- Kontroler: Pure Pursuit (volan) + PID (brzina)
- Optimizacija parametara:
  - Random Search (baseline)
  - Coordinate Descent + Golden Section (1D line-search po koordinatama)
- Pygame vizuelizacija + replay najbolje putanje

## Instalacija
```bash
pip install -r requirements.txt
```

## Pokretanje (primeri)
1) Samo simulacija (bez UI):
```bash
python main.py --mode sim --track s
```

2) Random Search (sačuva best.json):
```bash
python main.py --mode random --iters 200 --track s
```

3) Coordinate Descent + Golden Section:
```bash
python main.py --mode cd --cycles 4 --gs_iters 20 --track s
```

4) Pusti UI + replay best:
```bash
python main.py --mode play --load best.json --track s
```

## Parametri (theta)
- Ld: lookahead distance (Pure Pursuit)
- Kp,Ki,Kd: PID za brzinu
- v_ref: ciljna brzina
- delta_max_deg: limit volana

Sve u `src/config.py`.
