# Cómo se automejora una IA, y qué toma NAVROS de ello

Este documento resume las técnicas **públicamente documentadas** con las que se
entrenan y mejoran los modelos de lenguaje modernos (incluidos los de la familia
de Claude) y cómo NAVROS implementa cada una a su escala.

## Lo que no se puede extraer

- **Pesos ni detalles internos de Claude.** Claude no tiene acceso a sus propios
  pesos ni a su código de entrenamiento, y sabe de su entrenamiento lo mismo que
  está publicado. Este documento se basa solo en lo publicado.
- **Destilar a Claude.** Los términos de Anthropic (igual que los de OpenAI y
  Google) prohíben usar las salidas de sus modelos para entrenar modelos que
  compitan con ellos. Por eso NAVROS no trae conectores hacia esas APIs y exige
  que todo maestro declare que su licencia permite la destilación
  (`navros/distill.py`).

Lo valioso no son unos pesos concretos, sino **los métodos**, y esos sí están
publicados y aquí implementados.

## Las técnicas, una por una

| # | Técnica (referencia) | Idea | En NAVROS |
|---|---|---|---|
| 1 | Preentrenamiento autosupervisado (Radford et al. 2018–19) | El modelo aprende a predecir el siguiente token en texto sin etiquetar: se "entrena solo" con datos crudos. | `trainer.text_examples`, `Navros.ingest` |
| 2 | Tokenizador BPE a nivel de bytes (Sennrich et al. 2016; GPT-2) | El vocabulario se aprende del corpus; cualquier texto es representable con 256 bytes base. | `tokenizer.BPETokenizer.train` |
| 3 | Vocabulario extensible | Al asimilar un dominio nuevo, se añaden fusiones y se inicializan sus embeddings a partir de sus componentes. | `BPETokenizer.extend`, `NavrosLM.grow_vocab` |
| 4 | Ajuste supervisado (SFT) | Se entrena con pares pregunta→respuesta; la pérdida solo cuenta en la respuesta. | `trainer.skill_example` (máscara de pérdida) |
| 5 | RLHF (Christiano et al. 2017; Ouyang et al. 2022) | Un modelo de recompensa aprendido de preferencias humanas guía la política. | Sustituido por verificadores automáticos (fila 7). |
| 6 | Constitutional AI / RLAIF (Bai et al. 2022, Anthropic) | El propio modelo critica y revisa sus respuestas según unos principios; la retroalimentación de IA reemplaza parte de la humana. | El modo `consensus` es la versión mínima: el modelo se juzga a sí mismo sin respuestas humanas. |
| 7 | RL con recompensas verificables; STaR (Zelikman et al. 2022); ReST (Gulcehre et al. 2023); *expert iteration* | El modelo genera muchos intentos, un verificador conserva los correctos y el modelo se entrena con ellos. Es el motor de los modelos de razonamiento actuales. | Modo `verifier` de `Navros.improve_round` |
| 8 | Autoconsistencia (Wang et al. 2022); "LLMs can self-improve" (Huang et al. 2022) | Sin respuesta de referencia, la respuesta en la que coinciden varias muestras es una etiqueta fiable. | Modo `consensus`, `Navros._agreement` |
| 9 | Currículo de dificultad (Lee et al. 2025, *Self-Improving Transformers*) | Dominar el nivel n y autoetiquetar el n+1 permite superar lo que se enseñó. | Frontera de nivel con avance automático |
| 10 | Selección con reversión / *population-based training* (Jaderberg et al. 2017) | Se entrenan variantes de hiperparámetros, se queda la mejor y nunca se acepta una regresión. | Candidatos por tasa de aprendizaje + compuerta de aceptación |
| 11 | Crecimiento que preserva la función: Net2Net (Chen et al. 2015), *progressive stacking* (Gong et al. 2019) | Añadir capas o neuronas inicializadas para no cambiar la salida conserva lo aprendido y añade capacidad. | `grow_depth`, `grow_heads`, `grow_mlp` |
| 12 | Leyes de escala (Kaplan et al. 2020; Hoffmann et al. 2022) | La pérdida baja de forma predecible con más parámetros, datos y cómputo, y hay que escalar los tres juntos. | Presets `tiny`→`large` y `max_params` |
| 13 | Destilación (Hinton et al. 2015) | El estudiante imita la distribución de probabilidad del maestro. | `distill.logit_distill_loss`, `Navros.assimilate` |
| 14 | *Model soups* (Wortsman et al. 2022) | Promediar pesos de modelos afines combina lo que aprendió cada uno. | `distill.merge_state_dicts`, `Navros.merge_with` |
| 15 | Abacus (McLeish et al. 2024) + acoplamiento de posiciones (Cho et al. 2024) | Darle a cada dígito su posición dentro del número permite generalizar a números más largos que los vistos. | `NavrosLM.digit_positions` |

## Por qué la fila 15 fue necesaria (medido, no supuesto)

Con un transformer pequeño entrenado en sumas de 1–3 dígitos:

| Codificación posicional | 3 díg. | 4 díg. (nunca visto) | pass@16 en 4 díg. |
|---|---|---|---|
| RoPE | 92 % | 0 % | 0 % |
| NoPE | 98 % | 0 % | 2 % |
| NoPE + Abacus acoplado | 100 % | 26 % | 32 % |

La automejora solo funciona si la frontera es **alcanzable**, es decir, si al
menos algunos intentos aciertan. Con RoPE el bucle se atasca; con Abacus
acoplado arranca.

## Resultado de automejora (CPU, preset `mini`, 181k parámetros)

Solo se le enseñaron sumas de 1 a 3 dígitos. Todo lo demás lo aprendió de sus
propios intentos verificados:

| Versión | Frontera | 4 díg. | 6 díg. | 8 díg. | 10 díg. | 12 díg. |
|---|---|---|---|---|---|---|
| v0: solo se le enseñó 1–3 díg. | 4 | ~0 % | 0 % | 0 % | 0 % | 0 % |
| v1 | 4 | 56 % | — | — | — | — |
| v4 | 4 → 5 | 99 % | 0 % | — | — | — |
| v7 | 6 | 100 % | 87 % | — | — | — |
| v13 | 8 | 100 % | 100 % | 99 % | 50 % | — |
| **v16** | **11** | **100 %** | **100 %** | **100 %** | **89 %** | **75 %** |

16 rondas, 5,5 minutos de CPU. Los 11.997 ejemplos con los que aprendió más
allá de los 3 dígitos los **generó y verificó él mismo** (`pool.jsonl`, origen
`self:verifier`). Sin crecer: 181k parámetros todo el tiempo.

### Dos lecciones de la primera versión del bucle

La primera ejecución se estancó en 5 dígitos con ~10 % de exactitud. El
diagnóstico:

1. El número de muestras *k* se adaptaba con el rendimiento **total**, que
   dominaban los problemas de repaso (casi siempre correctos). Por eso *k* bajaba
   justo cuando la frontera necesitaba más intentos. Ahora se adapta con el
   rendimiento **en la frontera**.
2. Los pocos ejemplos nuevos de la frontera se diluían entre miles de ejemplos
   viejos. Ahora tienen un peso propio en la mezcla (`frontier_weight`).

Con esos dos cambios, la frontera de 5 dígitos pasó de 22 % a 82 % en una sola
ronda, y el modelo siguió avanzando hasta los 11 dígitos.

## Límites reales

- **Conocimiento.** Un modelo sabe lo que contienen sus datos y lo que puede
  almacenar en sus parámetros. NAVROS amplía ambos (`ingest`, crecimiento), pero
  el techo lo ponen el cómputo y los datos disponibles, no el código.
  `max_params` es solo una protección contra quedarse sin memoria de GPU:
  súbelo cuando tengas más.
- **Verificadores.** La automejora sin supervisión es fiable donde la respuesta
  se puede comprobar (matemáticas, código con pruebas, lógica). Para habilidades
  abiertas hace falta señal externa: datos, maestros con licencia o evaluación
  humana.
- **Conocimiento general.** Para eso hace falta preentrenar con corpus grandes
  (`navros ingest`) en GPU, o asimilar modelos de pesos abiertos cuya licencia
  lo permita.

## Diseño auditable

Cada ronda guarda checkpoint, historial y linaje (`state.json`) y la procedencia
de cada dato (`pool.jsonl`: `self:verifier`, `self:consensus`, `teacher:<nombre>`).
Una versión que empeora se rechaza. Así cualquier mejora se puede comprobar,
reproducir y revertir.
