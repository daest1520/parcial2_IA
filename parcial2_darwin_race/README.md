# Parcial 2 — Carrera de robots DARwIn-OP en Webots con aprendizaje por refuerzo seguro

Primera versión del proyecto: entorno, agentes y controladores de Webots.

| Agente | Algoritmo | Implementación |
|---|---|---|
| **DAC** (propio) | Discovery Actor-Critic del Taller 2 | propia (PyTorch) |
| PPO | Proximal Policy Optimization | propia (PyTorch) |
| DQN | Deep Q-Network | propia (PyTorch) |
| A2C | Advantage Actor-Critic | propia (PyTorch) |

## Contenido

```
race_core/   pista, sensores, escudo de seguridad, simulador de entrenamiento (numpy)
rl/          algoritmos DAC, PPO, DQN y A2C + entrenamiento, evaluación y carreras
scripts/     generación de mundos y experimentos
webots/controllers/
  darwin_walker/    marcha de cada robot (C++, RobotisOp2GaitManager)
  race_brain/       supervisor de la carrera (Python)
  race_pedestrian/  peatón que cruza y cede el paso
```

## Uso rápido

```bash
pip install -r requirements.txt
python scripts/make_world.py --size S --name carrera_S      # genera el mundo de Webots
python scripts/run_experiments.py --steps 400000 --workers 2
```

Pendiente: resultados de los experimentos, políticas entrenadas, mundos de todas las pistas e informe.
