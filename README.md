# Parcial 2 — Carrera de robots DARwIn-OP en Webots con aprendizaje por refuerzo seguro

Cuatro agentes compiten en una pista con curvas, chicana, obstáculos, un peatón que cruza
y zonas de piso resbaladizo/rugoso:

| Agente | Algoritmo | Implementación |
|---|---|---|
| **DAC** (propio) | Discovery Actor-Critic del Taller 2 (ensamble de críticos + módulo de descubrimiento) | propia (PyTorch) |
| PPO | Proximal Policy Optimization | propia (PyTorch) |
| DQN | Deep Q-Network | propia (PyTorch) |
| A2C | Advantage Actor-Critic | propia (PyTorch) |

## Estructura

```
race_core/        núcleo compartido (sólo numpy) — lo usan el entrenamiento y Webots
  track.py        pista paramétrica (tamaño S/M/L, salida, meta, sentido) + campo de distancias
  sensing.py      acciones calibradas, rayos virtuales, observación de 27 dimensiones
  safety.py       safe learning: escudo de seguridad + costo + Lagrangiano
  sim.py          gemelo cinemático vectorizado (entrenamiento y carreras masivas)
  policy_numpy.py ejecución de políticas exportadas (.npz) sin PyTorch
rl/algos/         dac.py  ppo.py  dqn.py  a2c.py
rl/train.py       entrenamiento, evaluación y carreras
scripts/
  run_experiments.py   los 20 experimentos (tamaño, salida, meta, parámetros, seguridad)
  analyze.py           tablas y figuras -> results/tables, results/figures
  make_world.py        genera los .wbt desde la misma pista
  deploy_to_webots.py  mundos + políticas entrenadas -> webots/
  webots_batch.sh      carreras en Webots sin interfaz
webots/
  worlds/*.wbt                       un mundo por configuración de pista
  controllers/darwin_walker/         C++: marcha con RobotisOp2GaitManager + levantarse tras caída
  controllers/race_brain/            Python: supervisor (políticas + escudo + medición + marcador)
  controllers/race_pedestrian/       Python: peatón que cruza y cede el paso
  policies/<mundo>/<ALGO>.npz        políticas entrenadas
results/          experimentos, tablas, figuras, carreras en Webots
```

## Requisitos

* Webots **R2025a** (los PROTO se descargan de GitHub la primera vez).
* Python 3.9+ con `numpy` (para Webots). Para entrenar: `pip install torch numpy pandas matplotlib`.
* Compilador C++ (Linux: `g++`/`make`; macOS: Xcode Command Line Tools; Windows: el MSYS2 que trae Webots).

## Ejecutar una carrera en Webots

1. Compilar el controlador de marcha (una sola vez): abrir `webots/worlds/carrera_S.wbt`,
   seleccionar el controlador `darwin_walker` en el editor de texto de Webots y pulsar **Build** (o
   `cd webots/controllers/darwin_walker && make` con `WEBOTS_HOME` definido).
   Se incluye el binario para Linux x86-64.
2. Abrir cualquier mundo de `webots/worlds/` y pulsar ▶. El marcador en pantalla muestra
   posición, progreso, colisiones, caídas e intervenciones del escudo.
3. Al terminar, `results/webots/<mundo>_g<parrilla>_<fecha>.json` guarda el resultado y
   `..._trayectoria.csv` cada decisión (pose, acción propuesta/ejecutada, escudo).

Lote sin interfaz: `scripts/webots_batch.sh carrera_S 4` (4 carreras rotando la parrilla).

**Google Colab:** abrir `Parcial2_Carrera_Colab.ipynb`; instala Webots sin ventana, corre la carrera de los 4 agentes y muestra resultados y video.

Argumentos del supervisor (`controllerArgs` del nodo `RACE_BRAIN`):
`--mode=explore` (muestreo con más temperatura para ver la exploración), `--no-shield`,
`--max-time=900`, `--grid=k`.

## Reproducir el entrenamiento y los experimentos

```bash
python scripts/run_experiments.py --steps 400000 --workers 2   # ≈ 3 h en 2 núcleos CPU
python scripts/analyze.py
python scripts/deploy_to_webots.py
```

## Calibración Webots → gemelo

Las velocidades de cada acción se midieron ejecutando el `RobotisOp2GaitManager` en Webots
(8 s por comando tras 3 s de transitorio): X=1.0 → 0.084 m/s; X=0.5 → 0.049 m/s;
(X=0.6, A=±0.5) → 0.034 m/s y ±0.48 rad/s; A=±1.0 → ±0.88 rad/s; (X=1, A=±0.3) → 0.080 m/s y ±0.29 rad/s.

## Resultados incluidos

* `results/experiments/E01…E20/`: políticas `.npz`, episodios de entrenamiento, evaluación (48 episodios) y carreras (`races_final.json`, 20 por experimento).
* `results/tables/*.csv`, `results/figures/*.png`: tablas y figuras del informe.
* `results/webots/`: 7 carreras ejecutadas en Webots R2025a (JSON + trayectoria CSV).
* `results/descartados/`: primera corrida de E05 y E11, invalidada porque un obstáculo ocupaba un puesto de la parrilla de salida (corregido en `Track` y repetido).
* `informe/Informe_Parcial2_IEEE.docx`: informe en formato IEEE.
