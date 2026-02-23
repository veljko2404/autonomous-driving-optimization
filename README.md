# Autonomno vozilo (2D) + numerička optimizacija

- 2D simulator (kinematički bicycle model)
- Kontroler: Pure Pursuit (volan) + PID (brzina)
- Optimizacija parametara:
  - Random Search (baseline)
  - Coordinate Descent + Golden Section (1D line-search po koordinatama)
  - Nelder-Mead
  - CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- Pygame vizuelizacija + replay najbolje putanje

## Instalacija
```bash
pip install -r requirements.txt
```

## Pokretanje
### 1. Samo simulacija (bez UI):
```bash
python main.py --mode sim --track s
```

### 2. Optimizacija

#### 2.1 Random Search (sacuva best.json):
```bash
python main.py --mode random --iters 200 --track s
```

#### 2.2 Coordinate Descent + Golden Section:
```bash
python main.py --mode cd --cycles 4 --gs_iters 20 --track s
```

#### 2.3 Nelder-Mead:
#### CMA-ES:

### 3. Pusti UI + replay best:
```bash
python main.py --mode play --load best.json --track s
```

## Parametri (theta)
- Ld: lookahead distance (Pure Pursuit)
- Kp,Ki,Kd: PID za brzinu
- v_ref: ciljna brzina
- delta_max_deg: limit volana

Sve je u `src/config.py`.
