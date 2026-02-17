import math

# Parametri vozila (bicycle model)
WHEELBASE = 2.6 # rastojanje izmedju tockova (L), u metrima

# Parametri simulacije
DT = 0.05 # vremenski korak simulacije (diskretizacija), u sekundama
MAX_TIME = 40.0 # maksimalno trajanje jednog rollout-a u sekundama

# Ogranicenja upravljanja (aktuatori)
A_MIN, A_MAX = -6.0, 3.0 # granice ubrzanja/kocenja, m/s^2
V_MIN, V_MAX = 0.0, 30.0  # granice brzine, m/s
DELTA_MIN, DELTA_MAX = -math.radians(35), math.radians(35) # granice ugla volana, rad

# Geometrija staze
TRACK_WIDTH = 6.0 # sirina puta u metrima (koristi se i za detekciju offroad)

# Opsezi parametara koje optimizujemo (theta)
# Svaki parametar ima donju i gornju granicu (box constraints)
BOUNDS = {
    "Ld": (2.0, 20.0), # koliko unapred auto “gleda” po stazi dok vozi (lookahead distance)
    "Kp": (0.0, 6.0), # proporcionalni koeficijent PID-a
    "Ki": (0.0, 2.0), # integralni koeficijent PID-a
    "Kd": (0.0, 2.0), # diferencijalni koeficijent PID-a
    "v_ref": (4.0, 24.0), # ciljna brzina koju PID prati, m/s
    "delta_max_deg": (10.0, 35.0) # limit ugla volana u stepenima (dodatna stabilizacija)
}

# Pocetne vrednosti parametara kontrolera
# Koriste se kao start za simulaciju i kao inicijalna tacka za coordinate descent
DEFAULT_THETA = {
    "Ld": 8.0,
    "Kp": 1.8,
    "Ki": 0.2,
    "Kd": 0.4,
    "v_ref": 14.0,
    "delta_max_deg": 28.0,
}