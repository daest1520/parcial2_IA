// Informe IEEE — Parcial 2 IA. Uso: node make_report.js <datos.json> <salida.docx>
const fs = require("fs");
const L = require("./lib");
const { d, P, H1, H2, EQ, BUL, FIG, TAB, FONT, COL_W, FULL_W, runs } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, SectionType, LevelFormat, Footer, PageNumber } = d;

const D = JSON.parse(fs.readFileSync(process.argv[2]));
const R = D.root;
const fig = (p) => `${R}/results/figures/${p}`;
const pct = (x) => (x * 100).toFixed(0) + "%";
const f1 = (x) => (x === null || Number.isNaN(x) ? "—" : Number(x).toFixed(1));
const f2 = (x) => (x === null || Number.isNaN(x) ? "—" : Number(x).toFixed(2));
const ALG = ["DAC", "PPO", "DQN", "A2C"];

// ------------------------------------------------------------------ autores (bloque IEEE)
function authorsTable(authors) {
  const NB = { style: d.BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const w = Math.floor(FULL_W / authors.length);
  const line = (t, opt = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
    children: [new TextRun({ text: t, font: FONT, size: opt.size || 20, italics: !!opt.it })] });
  return new d.Table({
    width: { size: w * authors.length, type: d.WidthType.DXA }, columnWidths: authors.map(() => w),
    alignment: AlignmentType.CENTER,
    rows: [new d.TableRow({ children: authors.map(([name, code, mail]) => new d.TableCell({
      width: { size: w, type: d.WidthType.DXA },
      borders: { top: NB, bottom: NB, left: NB, right: NB },
      children: [line(name, { size: 22 }), line("Facultad Tecnológica", { it: true }),
        line("Universidad Distrital Francisco José de Caldas"), line("Bogotá, Colombia"),
        line(`Código: ${code}`), line(mail)] })) })],
  });
}

// ------------------------------------------------------------------ título
const title = [
  new Paragraph({ children: [new TextRun({ text: "Carrera de humanoides DARwIn-OP en Webots con aprendizaje por refuerzo seguro: Discovery Actor-Critic frente a PPO, DQN y A2C", font: FONT, size: 44 })],
    alignment: AlignmentType.CENTER, spacing: { after: 160 } }),
  authorsTable([
    ["Cristian Castro Tapiero", "20222578114", "ccastrot@udistrital.edu.co"],
    ["Daniel Esteban Pardo Moreno", "20222578072", "depardom@udistrital.edu.co"],
  ]),
  new Paragraph({ children: [], spacing: { after: 60 } }),
  new Paragraph({ children: [new TextRun({ text: "Inteligencia Artificial — Parcial 02, semestre 2026-3. Profesor: Jorge Eduardo Hernández Rodríguez", font: FONT, size: 18, italics: true })],
    alignment: AlignmentType.CENTER, spacing: { after: 240 } }),
];

// ------------------------------------------------------------------ cuerpo
const S = D.summary;   // números agregados
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));   // {full:[...]} = bloque a 1 columna

add(new Paragraph({ children: [new TextRun({ text: "Resumen", font: FONT, size: 18, bold: true, italics: true }),
  new TextRun({ text: "—", font: FONT, size: 18, bold: true }), ...runs(D.text.abstract, { size: 18, bold: true })],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 } }));
add(new Paragraph({ children: [new TextRun({ text: "Palabras clave", font: FONT, size: 18, bold: true, italics: true }),
  new TextRun({ text: "—DARwIn-OP, Webots, aprendizaje por refuerzo, actor-crítico, PPO, DQN, exploración epistémica, aprendizaje seguro, escudo de seguridad.", font: FONT, size: 18, bold: true })],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 } }));

// ================================================================== I
add(H1("Introducción"));
add(P("El DARwIn-OP es un humanoide abierto de 20 grados de libertad y 0.45 m de altura [12]. Hacerlo competir en una carrera exige resolver a la vez locomoción bípeda estable, navegación por una pista con curvas, evasión de obstáculos fijos y móviles y convivencia segura con personas y con otros robots. El objetivo del trabajo es construir en Webots [11] un entorno de carreras propio y entrenar cuatro agentes —el algoritmo propio del Taller 2 (Discovery Actor-Critic, DAC) y tres líneas base: PPO [3], DQN [1] y un Actor-Critic síncrono (A2C) [2]— que compiten entre sí: gana el robot que llega primero a la meta."));
add(P("La solución separa dos niveles. El nivel bajo (caminar, girar, levantarse tras una caída) lo resuelve el ¦RobotisOp2GaitManager¦ oficial del robot, calibrado en Webots. El nivel alto —qué paso dar cada segundo— lo aprenden los agentes de refuerzo a partir de un vector de 27 sensores. Como entrenar cientos de miles de pasos con la física completa de cuatro humanoides es inviable, se construyó un gemelo cinemático rápido de la misma pista, con las velocidades medidas en Webots; las políticas entrenadas se ejecutan después, sin cambios, sobre los DARwIn-OP en Webots. Todo el sistema incorpora aprendizaje seguro: un escudo que veta acciones peligrosas y una restricción de costo con multiplicador de Lagrange."));

// ================================================================== II
add(H1("Entorno de carreras en Webots"));
add(H2("Pista, obstáculos y condiciones de recorrido"));
add(P("La pista se genera de forma paramétrica (clase ¦Track¦) y el mismo código produce el mundo de Webots (¦.wbt¦) y el gemelo de entrenamiento, por lo que ambos son geométricamente idénticos. Es un corredor de 1.0 m de ancho con forma de serpentina: tres rectas, dos curvas de 180° y una chicana sinusoidal en la recta central (Fig. 1). Incluye: paredes de 0.30 m; obstáculos fijos (cajas y cilindros, uno cada ≈2.5 m) alternados a izquierda y derecha; un peatón (modelo ¦Pedestrian¦ de Webots) que cruza la pista de lado a lado, espera 6 s y vuelve, y que en pistas M y L son dos; una zona resbaladiza (fricción de Coulomb 0.25) y una zona rugosa (fricción 1.2) definidas con ¦ContactProperties¦; una línea de salida y una meta a cuadros. Se definieron tres tamaños de espacio (Tabla I) y la salida y la meta pueden ubicarse en cualquier fracción del recorrido, en ambos sentidos."));
add(FIG(fig("pistas.png"), 7.0, "Pistas S, M y L generadas. Paredes en negro, obstáculos en gris, zona resbaladiza en azul, zona rugosa en marrón, cruce del peatón punteado, salida (triángulo verde) y meta (estrella).", true));
add(TAB("Tamaños del espacio de carrera", ["Pista", "Área (m)", "Longitud", "Obstáculos", "Peatones"],
  D.tracks.map((t) => [t.size, t.area, `${t.length.toFixed(1)} m`, t.obst, t.ped]), [900, 1200, 1100, 1000, 840]));
add(H2("Robots, parrilla de salida y medición"));
add(P("Cuatro DARwIn-OP (PROTO ¦Darwin-op¦ de Webots R2025a, agregado como indica el enunciado) se identifican con un color: DAC rojo, PPO azul, DQN verde y A2C amarillo. Salen de una parrilla escalonada 2×2 cuya asignación rota entre carreras para que ningún agente se beneficie de la posición. La arquitectura del sistema se muestra en la Fig. 2. Cada robot ejecuta el controlador C++ ¦darwin_walker¦, que recibe por su ¦Receiver¦ el comando de marcha (amplitudes X de avance y A de giro), lo aplica con rampa suave y lo ejecuta con el ¦GaitManager¦; si el acelerómetro detecta una caída, reproduce la rutina de levantarse del ¦MotionManager¦ (páginas 10/11). Un robot supervisor (¦race_brain¦, Python) lee cada segundo las poses, construye la observación, consulta las cuatro políticas, aplica el escudo de seguridad y envía los comandos. También registra el desempeño: tiempo de llegada y orden, progreso, contactos, caídas e intervenciones del escudo; muestra un marcador en pantalla (Fig. 3) y guarda un JSON con el resultado y un CSV con la trayectoria."));
add(FIG(fig("arquitectura.png"), 7.0, "Arquitectura: entrenamiento en el gemelo, exportación de políticas y ejecución en Webots con el supervisor y el controlador de marcha.", true));
add(FIG(fig("webots_cenital.jpg"), 3.45, "Carrera real en Webots (pista S, parrilla g1) vista desde arriba en t = 30 s: los cuatro DARwIn-OP tras la salida, obstáculos, peatón, zonas resbaladiza (azul) y rugosa (marrón) y meta a cuadros. Arriba a la izquierda, el marcador en vivo (el DAC, rojo, va primero)."));
add(FIG(fig("webots_persecucion.jpg"), 3.45, "Cámara de persecución en la misma carrera: el DARwIn-OP del DAC entra a la zona resbaladiza; ganó esta carrera en 293 s."));

add(H2("Espacio de acciones y calibración de la marcha"));
add(P("Las nueve acciones discretas de alto nivel corresponden a pares de amplitudes del ¦GaitManager¦. Sus velocidades se **midieron en Webots** con un supervisor de calibración: 3 s de transitorio y 8 s de medición por comando (Tabla II). La velocidad máxima del humanoide es 0.084 m/s, por lo que cada decisión (Δt = 1 s, ≈1.7 pasos de marcha) avanza como máximo 8.4 cm."));
add(TAB("Acciones calibradas en Webots", ["Acción", "X", "A", "v (m/s)", "ω (rad/s)"], D.actions, [1500, 600, 600, 1050, 1050]));
add(H2("Observación y recompensa"));
add(P("La observación (27 valores) contiene: 9 rayos de distancia virtuales entre −90° y +90° (alcance 1.5 m, contra paredes, obstáculos, humanos y robots); seno y coseno del rumbo hacia dos puntos de la línea central a 0.6 y 1.5 m; desplazamiento lateral; fracción de recorrido restante; velocidades v y ω; tipo de piso (resbaladizo o rugoso); y, para el humano y el robot más cercanos, un indicador de presencia, la distancia y la dirección (seno y coseno). La recompensa por paso es"));
add(EQ("r~t~ = Δs~t~/(v~max~Δt) − 0.05 + 20·1[meta] − 10·1[colisión] − 3·1[caída] − 0.5·1[escudo]", 1));
add(P("donde Δs es el avance sobre la línea central. El episodio termina al cruzar la meta, al colisionar o al agotar el tiempo (3.5 veces el tiempo ideal). En el gemelo, la marcha sigue un modelo de primer orden con ruido de deriva. Las caídas son estocásticas, más probables en piso resbaladizo y ante cambios bruscos de comando, y dejan al robot inmóvil 12 s mientras se levanta, como en Webots.", { noIndent: true }));

// ================================================================== III
add(H1("Algoritmos"));
add(P("Los cuatro agentes reciben la misma observación, el mismo espacio de acciones, la misma recompensa, la misma capa de seguridad y el mismo presupuesto de 4×10^5^ pasos con 16 entornos en paralelo. Cada uno explora de manera distinta: ε-greedy en DQN, muestreo de una política estocástica con bonificación de entropía en PPO y A2C, y descubrimiento epistémico en DAC."));
add(H2("Agente propio: Discovery Actor-Critic (Taller 2)"));
add(P("El DAC agrega dos conjuntos de redes a un actor-crítico. Primero, un **ensamble de K críticos** V^E^~k~ entrenados con máscaras bootstrap [9]. Su desacuerdo σ~V~(s) mide qué tanto se puede confiar en el crítico, y el gradiente del actor se pondera con w(s) = 1/(1+κσ̃~V~(s)) para que un crítico inseguro no \"engañe\" al actor. Segundo, un **módulo de descubrimiento**: un codificador aleatorio fijo ψ y un ensamble de M modelos de dinámica g~m~(ψ(s),a) → ψ(s′). La varianza entre modelos es una recompensa intrínseca de novedad epistémica [7], [8] con crítico propio V^I^ no episódico. Cuando σ~V~ supera su media móvil (o con probabilidad ε~ask~), el agente \"pregunta\" por la novedad de **todas** las acciones y actúa con μ(a|s) ∝ π~θ~(a|s)·exp(ηÛ(s,a)). La corrección fuera de política usa ρ̄ = min(1, π/b) [10]:"));
add(EQ("L~π~ = −E[ρ̄~t~ w(s~t~) Â~t~ log π~θ~(a~t~|s~t~)] − βH(π~θ~),  Â = A^E^ + c~I~A^I^", 2));
add(P("Las ventajas se estiman con GAE(λ) [4]. **Adaptación al problema:** como la observación es un vector y no una imagen de Atari, el codificador Nature-CNN del taller se reemplazó por un MLP de 128-128 y ψ por un MLP aleatorio de 64 salidas. Se usan K = 3 críticos y M = 5 modelos. **Por qué podría funcionar:** la pista tiene zonas raras (el cruce del peatón, el piso resbaladizo, la chicana) donde el crítico se equivoca. El DAC detecta allí el desacuerdo, reduce el peso de esas actualizaciones y fuerza la exploración de acciones cuyas consecuencias aún no conoce.", { noIndent: true }));
add(H2("Agente base 1: PPO"));
add(P("PPO [3] maximiza el objetivo recortado L^CLIP^ = E[min(r~t~Â~t~, clip(r~t~, 1−ε, 1+ε)Â~t~)], con r~t~ = π~θ~/π~θold~, y reutiliza cada lote durante varias épocas de minilotes. El recorte evita actualizaciones destructivas y la reutilización de datos le da buena eficiencia de muestras. Además, es el estándar de facto para control con políticas estocásticas, por lo que se espera que aprenda con rapidez a seguir la pista."));
add(H2("Agente base 2: DQN"));
add(P("DQN [1] aproxima Q(s,a) con una red neuronal entrenada con memoria de repetición, red objetivo y pérdida de Huber. Es adecuado porque las acciones son discretas y pocas (9). Al ser fuera de política, reutiliza toda la experiencia (incluida la vetada por el escudo) y en evaluación usa ε = 0.05, como en [1]. Su debilidad esperada es la exploración ε-greedy no dirigida en una pista larga."));
add(H2("Agente base 3: Actor-Critic (A2C)"));
add(P("A2C [2] entrena un actor π~θ~ y un crítico V~φ~ con la ventaja Â~t~ = R~t~ − V(s~t~) estimada con GAE, en lotes síncronos de 16 entornos. Es el actor-crítico \"puro\" del que parte el DAC y comparte con él la red y los hiperparámetros, de modo que la comparación DAC vs A2C aísla el efecto del ensamble y del descubrimiento."));

// ================================================================== IV
add(H1("Aprendizaje seguro (safe learning)"));
add(P("El robot debe aprender sin dañarse ni dañar a humanos u otros robots. Se combinaron cuatro mecanismos, activos durante el entrenamiento **y** en la carrera:"));
add(BUL("**Escudo de seguridad** [5]: antes de ejecutar la acción propuesta se predice, con el modelo de marcha calibrado, la pose a 0.5 s y a 1 s para las 9 acciones. Una acción es segura si respeta los márgenes de 0.08 m a paredes y obstáculos, 0.25 m de \"espacio personal\" a los humanos y 0.08 m a otros robots. Si ya se está dentro de un margen, también es segura cuando no lo empeora. Si la propuesta es insegura, se ejecuta la acción segura que la política prefiere y el agente recibe −0.5. Una regla de desbloqueo evita que dos robots en contacto queden inmóviles para siempre."));
add(BUL("**Restricción de costo (CMDP)** [6]: cada paso genera un costo c~t~ por invadir márgenes, colisionar (2) o caer (1). Los agentes optimizan r~t~ − λc~t~ y λ se ajusta por ascenso dual, λ ← max(0, λ + α(E[c] − d)), con límite d = 0.02 por paso."));
add(BUL("**Seguridad de bajo nivel** en ¦darwin_walker¦: saturación de amplitudes (límite de velocidad), rampa de cambio para evitar tirones que tumben al robot, ¦watchdog¦ que detiene la marcha si no llegan comandos en 1.5 s y rutina automática de levantarse."));
add(BUL("**Humanos cooperativos**: el peatón cede el paso si su siguiente posición quedaría a menos de 0.52 m de un robot. Aun así, el escudo mantiene el margen por si el humano no lo hiciera."));
add(P(D.text.safety_result));

// ================================================================== V
add(H1("Implementación y ejecución"));
add(P("**Todos los algoritmos son de implementación propia** en PyTorch 2 (sin bibliotecas de RL), sobre un arnés común. Así las cuatro implementaciones comparten exactamente el entorno, la red base y la medición. El gemelo (¦race_core¦) sólo usa numpy y acelera las consultas geométricas con un campo de distancias precalculado (≈4 500 pasos/s en CPU). Tras entrenar, cada política se exporta a ¦.npz¦ y el supervisor de Webots la ejecuta sólo con numpy (diferencia con PyTorch < 10^−6^). Cada algoritmo **explora el entorno** durante el entrenamiento con su propio mecanismo y **participa en la carrera** como uno de los cuatro DARwIn-OP. La Tabla III resume los parámetros base."));
add(TAB("Parámetros base de cada algoritmo", ["Parámetro", "DAC", "PPO", "DQN", "A2C"], D.params, [1500, 900, 900, 900, 840], { size: 14 }));
add(P(D.text.impl_note));

// ================================================================== VI
add(H1("Experimentos y comparación"));
add(H2("Diseño experimental"));
add(P("Se realizaron **20 experimentos** (Tabla IV): E01 es la configuración base. E02–E03 varían el **tamaño del espacio**; E04–E07, la **posición inicial**, incluido el sentido inverso; E08–E11, la **posición de la meta**; E12–E16 y E19–E20, los **parámetros de los algoritmos**; y E17–E18 son ablaciones de seguridad. En cada experimento se entrenan los cuatro agentes desde cero, se evalúan en 48 episodios nuevos (semillas no vistas, dos robots de tráfico) y compiten en 20 carreras de cuatro robots, rotando la parrilla, con la misma lógica de despliegue que Webots. En total son 80 entrenamientos (32 millones de pasos), 3 840 episodios de evaluación y 400 carreras. Se comparan la **recompensa acumulada**, la **tasa de éxito** (llegar a la meta), el tiempo de llegada, las victorias en carrera y las métricas de seguridad."));
add(TAB("Diseño de los 20 experimentos", ["Exp.", "Factor", "Variación"], D.design, [520, 1050, 3470], { size: 14 }));
add(H2("Resultados globales"));
add(P(D.text.results_global));
add(TAB("Métricas promedio sobre los 20 experimentos", ["Métrica", ...ALG], D.global_table, [1800, 810, 810, 810, 810], { size: 14 }));
add(FIG(fig("curvas_aprendizaje.png"), 7.0, "Curvas de aprendizaje promedio (E01–E16, E19–E20; banda = ±1 error estándar entre experimentos). Izquierda: recompensa acumulada por episodio; derecha: tasa de éxito.", true));
add(FIG(fig("exito_por_experimento.png"), 7.0, "Tasa de éxito de evaluación (48 episodios) por experimento y algoritmo.", true));
add(TAB("Tasa de éxito (%) y recompensa media por experimento", ["Exp.", "DAC", "PPO", "DQN", "A2C"], D.per_exp, [700, 1085, 1085, 1085, 1085], { size: 13 }));
add(H2("Efecto de cada factor"));
add(P(D.text.factors));
add(FIG(fig("exito_por_factor.png"), 3.45, "Éxito medio agrupado por factor experimental."));
add(FIG(fig("boxplot_retorno.png"), 3.45, "Distribución de la recompensa acumulada en todos los episodios de evaluación (960 por algoritmo)."));
add(H2("Carreras"));
add(P(D.text.races));
add(FIG(fig("carreras.png"), 3.45, "Izquierda: porcentaje de carreras ganadas (número sobre la barra) en 400 carreras. Derecha: distribución de la posición final."));
add(FIG(fig("trayectoria_carrera_E01.png"), 3.45, "Trayectorias de los cuatro robots en una carrera de la pista base (E01)."));
add(H2("Validación en Webots"));
add(P(D.text.webots));
add(TAB("Carreras con la física completa de Webots", ["Mundo / parrilla", "1.°", "Tiempo", "Llegan", "Contactos (D/P/Q/A)"], D.webots_table, [1300, 650, 750, 1100, 1240], { size: 13 }));
add(H2("Seguridad"));
add(P(D.text.safety_exp));
add(FIG(fig("diagnostico_dac.png"), 3.45, "Diagnóstico del DAC en E01: desacuerdo del ensamble de críticos σ~V~, fracción de pasos en que el agente \"pregunta\" al módulo de descubrimiento y multiplicador de Lagrange λ."));

// ================================================================== VII
add(H1("Conclusiones y recomendaciones"));
D.text.conclusions.forEach((t) => add(P(t)));

// ================================================================== Referencias
add(new Paragraph({ children: [new TextRun({ text: "Referencias", font: FONT, size: 20, smallCaps: true })],
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 } }));
const refs = [
  "V. Mnih et al., \"Human-level control through deep reinforcement learning,\" ¦Nature¦, vol. 518, pp. 529–533, 2015.",
  "V. Mnih et al., \"Asynchronous methods for deep reinforcement learning,\" in ¦Proc. ICML¦, 2016, pp. 1928–1937.",
  "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, \"Proximal policy optimization algorithms,\" arXiv:1707.06347, 2017.",
  "J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, \"High-dimensional continuous control using generalized advantage estimation,\" in ¦Proc. ICLR¦, 2016.",
  "M. Alshiekh, R. Bloem, R. Ehlers, B. Könighofer, S. Niekum, and U. Topcu, \"Safe reinforcement learning via shielding,\" in ¦Proc. AAAI¦, 2018, pp. 2669–2678.",
  "A. Ray, J. Achiam, and D. Amodei, \"Benchmarking safe exploration in deep reinforcement learning,\" OpenAI, Tech. Rep., 2019.",
  "D. Pathak, D. Gandhi, and A. Gupta, \"Self-supervised exploration via disagreement,\" in ¦Proc. ICML¦, 2019, pp. 5062–5071.",
  "Y. Burda, H. Edwards, A. Storkey, and O. Klimov, \"Exploration by random network distillation,\" in ¦Proc. ICLR¦, 2019.",
  "I. Osband, C. Blundell, A. Pritzel, and B. Van Roy, \"Deep exploration via bootstrapped DQN,\" in ¦Proc. NeurIPS¦, 2016.",
  "L. Espeholt et al., \"IMPALA: Scalable distributed deep-RL with importance weighted actor-learner architectures,\" in ¦Proc. ICML¦, 2018.",
  "O. Michel, \"Webots: Professional mobile robot simulation,\" ¦Int. J. Advanced Robotic Systems¦, vol. 1, no. 1, pp. 39–42, 2004.",
  "I. Ha, Y. Tamura, H. Asama, J. Han, and D. W. Hong, \"Development of open humanoid platform DARwIn-OP,\" in ¦Proc. SICE Annual Conf.¦, 2011, pp. 2178–2181.",
  "J. García and F. Fernández, \"A comprehensive survey on safe reinforcement learning,\" ¦J. Mach. Learn. Res.¦, vol. 16, pp. 1437–1480, 2015.",
];
refs.forEach((r, i) => add(new Paragraph({ children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 16 }), ...runs(r, { size: 16 })],
  alignment: AlignmentType.JUSTIFIED, indent: { left: 360, hanging: 360 }, tabStops: [{ type: d.TabStopType.LEFT, position: 360 }], spacing: { after: 30 } })));

// ------------------------------------------------------------------ documento
const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 900, right: 900 } };
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] });

// separar figuras/tablas de ancho completo en secciones de 1 columna
const sections = [{ properties: { page, type: SectionType.NEXT_PAGE }, footers: { default: footer }, children: title }];
let cur = [];
const flush = () => { if (cur.length) sections.push({ properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 360 } }, footers: { default: footer }, children: cur }); cur = []; };
body.forEach((el) => {
  if (el && el.full) {
    flush();
    sections.push({ properties: { page, type: SectionType.CONTINUOUS, column: { count: 1 } }, footers: { default: footer }, children: el.full });
  } else cur.push(el);
});
flush();

const doc = new Document({
  creator: "Parcial 2 IA", title: "Carrera DARwIn-OP con RL seguro",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 240, hanging: 160 } } } }] }] },
  sections,
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(process.argv[3], b); console.log("ok", process.argv[3]); });
