// =====================================================================
//  darwin_walker  —  controlador de bajo nivel de cada DARwIn-OP
//
//  Recibe comandos de alto nivel del supervisor "race_brain" por el
//  Receiver del robot (canal = campo `channel` del PROTO) y los ejecuta
//  con el RobotisOp2GaitManager oficial de Webots.
//
//  Protocolo (texto):
//     "CMD <x_amp> <a_amp>"   amplitud de avance [-1,1] y de giro [-1,1]
//     "STOP"                  detiene la marcha (queda de pie)
//  El robot informa por su Emitter:
//     "FALL <id>"  cuando detecta caída (acelerómetro)  y  "UP <id>" al levantarse.
//
//  Capa de seguridad de bajo nivel (safe learning):
//     * saturación de amplitudes (límite de velocidad),
//     * limitación de la tasa de cambio (sin tirones que tumben al robot),
//     * watchdog: si no llegan comandos en 1.5 s, se detiene,
//     * rutina automática de levantarse tras una caída.
// =====================================================================
#include <RobotisOp2GaitManager.hpp>
#include <RobotisOp2MotionManager.hpp>
#include <webots/Accelerometer.hpp>
#include <webots/Emitter.hpp>
#include <webots/LED.hpp>
#include <webots/Receiver.hpp>
#include <webots/Robot.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using webots::Robot; using webots::Accelerometer; using webots::Receiver; using webots::Emitter; using webots::LED;
using managers::RobotisOp2GaitManager; using managers::RobotisOp2MotionManager;

static double clampd(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

int main(int argc, char **argv) {
  webots::Robot *robot = new webots::Robot();
  const int ts = (int)robot->getBasicTimeStep();
  std::string name = robot->getName();

  Accelerometer *acc = robot->getAccelerometer("Accelerometer");
  acc->enable(ts);
  Receiver *rx = robot->getReceiver("receiver");
  rx->enable(ts);
  Emitter *tx = robot->getEmitter("emitter");
  LED *eye = robot->getLED("EyeLed");
  LED *head = robot->getLED("HeadLed");

  RobotisOp2MotionManager motion(robot);
  RobotisOp2GaitManager gait(robot, "config.ini");

  robot->step(ts);
  motion.playPage(9);  // postura inicial
  for (int i = 0; i < 20; ++i) robot->step(ts);
  gait.start();

  const double MAX_X = 1.0, MAX_A = 1.0;
  const double RATE_X = 0.05, RATE_A = 0.08;  // máximo cambio por paso de control (16 ms)
  double cmdX = 0.0, cmdA = 0.0, curX = 0.0, curA = 0.0;
  double lastCmd = robot->getTime();
  bool walking = true;
  int fup = 0, fdown = 0;

  while (robot->step(ts) != -1) {
    // ------------------ comandos ------------------
    while (rx->getQueueLength() > 0) {
      std::string msg((const char *)rx->getData(), rx->getDataSize());
      rx->nextPacket();
      double x, a;
      if (sscanf(msg.c_str(), "CMD %lf %lf", &x, &a) == 2) {
        cmdX = clampd(x, -MAX_X, MAX_X);
        cmdA = clampd(a, -MAX_A, MAX_A);
        lastCmd = robot->getTime();
        if (!walking) { gait.start(); walking = true; }
      } else if (msg.rfind("STOP", 0) == 0) {
        cmdX = cmdA = 0.0;
        lastCmd = robot->getTime();
      }
    }
    if (robot->getTime() - lastCmd > 1.5) cmdX = cmdA = 0.0;  // watchdog

    // ------------------ rampa suave ------------------
    curX += clampd(cmdX - curX, -RATE_X, RATE_X);
    curA += clampd(cmdA - curA, -RATE_A, RATE_A);
    gait.setXAmplitude(curX);
    gait.setAAmplitude(curA);
    gait.step(ts);
    eye->set(std::fabs(curX) > 0.05 ? 0x00FF00 : 0xFFAA00);

    // ------------------ detección de caída ------------------
    const double *v = acc->getValues();
    fup = (v[1] < 512.0 - 80.0) ? fup + 1 : 0;
    fdown = (v[1] > 512.0 + 80.0) ? fdown + 1 : 0;
    if (fup > 60 || fdown > 60) {
      std::string m = "FALL " + name;
      tx->send(m.c_str(), (int)m.size() + 1);
      head->set(0xFF0000);
      gait.stop();
      motion.playPage(fup > 60 ? 10 : 11);  // levantarse (boca abajo / boca arriba)
      motion.playPage(9);
      gait.start();
      curX = curA = 0.0;
      fup = fdown = 0;
      m = "UP " + name;
      tx->send(m.c_str(), (int)m.size() + 1);
      head->set(0x00FF00);
    }
  }
  delete robot;
  return 0;
}
