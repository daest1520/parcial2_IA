# Basado en el controlador "pedestrian" de Webots (Copyright 1996-2024 Cyberbotics Ltd.,
# Apache License 2.0). Modificado para el Parcial 2:
#   * cruza la pista de A a B, espera `pause` s, vuelve y espera (mismo modelo que el gemelo),
#   * CEDE EL PASO: si su siguiente posición quedaría a menos de 0.52 m de un robot,
#     se detiene (los humanos reales no atropellan robots).
"""Peatón que cruza la pista y cede el paso a los robots."""
from controller import Supervisor
import optparse
import math

ROBOT_DEFS = ["RACER_DAC", "RACER_PPO", "RACER_DQN", "RACER_A2C"]
YIELD_DIST = 0.25 + 0.15 + 0.12


class Pedestrian(Supervisor):
    BODY_PARTS_NUMBER = 13
    WALK_SEQUENCES_NUMBER = 8
    ROOT_HEIGHT = 1.27
    CYCLE_TO_DISTANCE_RATIO = 0.22
    joint_names = ["leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
                   "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
                   "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
                   "rightLegAngle", "rightLowerLegAngle", "rightFootAngle", "headAngle"]
    height_offsets = [-0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03]
    angles = [
        [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74],
        [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],
        [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0],
        [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],
        [-0.47, -0.3, -0.58, -0.21, 0.0, -0.16, -0.7, -0.38],
        [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],
        [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4],
        [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],
        [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07],
        [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],
        [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49],
        [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],
        [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09]]

    def run(self):
        op = optparse.OptionParser()
        op.add_option("--a", default="0 0"); op.add_option("--b", default="1 0")
        op.add_option("--speed", type=float, default=0.25); op.add_option("--pause", type=float, default=6.0)
        op.add_option("--phase", type=float, default=0.0)
        o, _ = op.parse_args()
        a = [float(v) for v in o.a.split()]; b = [float(v) for v in o.b.split()]
        L = math.hypot(b[0] - a[0], b[1] - a[1]); Tc = L / o.speed; Tp = o.pause
        ts = int(self.getBasicTimeStep())
        me = self.getSelf()
        tf, rf = me.getField("translation"), me.getField("rotation")
        joints = [me.getField(n) for n in self.joint_names]
        robots = [self.getFromDef(d) for d in ROBOT_DEFS]
        robots = [r for r in robots if r is not None]
        phase, walked = o.phase, 0.0

        def pos_at(ph):
            u = ph % (2 * (Tc + Tp))
            if u < Tc: f, d = u / Tc, 1
            elif u < Tc + Tp: f, d = 1.0, 0
            elif u < 2 * Tc + Tp: f, d = 1.0 - (u - Tc - Tp) / Tc, -1
            else: f, d = 0.0, 0
            return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, d

        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        while self.step(ts) != -1:
            dt = ts / 1000.0
            nx, ny, d = pos_at(phase + dt)
            blocked = False
            if d != 0:
                for r in robots:
                    p = r.getPosition()
                    if math.hypot(p[0] - nx, p[1] - ny) < YIELD_DIST:
                        blocked = True; break
            if not blocked:
                phase += dt
                if d != 0:
                    walked += o.speed * dt
                    heading = math.atan2(b[1] - a[1], b[0] - a[0]) + (0 if d > 0 else math.pi)
            x, y, _ = pos_at(phase)
            seq = int((walked / self.CYCLE_TO_DISTANCE_RATIO) % self.WALK_SEQUENCES_NUMBER)
            ratio = walked / self.CYCLE_TO_DISTANCE_RATIO - int(walked / self.CYCLE_TO_DISTANCE_RATIO)
            for i in range(self.BODY_PARTS_NUMBER):
                ang = self.angles[i][seq] * (1 - ratio) + self.angles[i][(seq + 1) % 8] * ratio
                joints[i].setSFFloat(ang)
            h = self.height_offsets[seq] * (1 - ratio) + self.height_offsets[(seq + 1) % 8] * ratio
            tf.setSFVec3f([x, y, self.ROOT_HEIGHT + h])
            rf.setSFRotation([0, 0, 1, heading])


Pedestrian().run()
