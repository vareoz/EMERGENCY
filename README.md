# NAVROS

**Un modelo de lenguaje que se entrena, se evalúa y crece por sí mismo.**

NAVROS empieza sabiendo casi nada. Crea su propio tokenizador, inicializa sus
pesos y, a partir de ahí, mejora en un bucle: se propone problemas más difíciles
de los que domina, los resuelve con sus propios pesos, filtra lo que hizo bien y
se entrena con eso. Si deja de mejorar, crece. Puede asimilar otros modelos
cuya licencia lo permita, y tiene una capa cuántica híbrida lista para
ejecutarse en hardware real.

Todo es código propio en PyTorch, sin modelos preentrenados de terceros.

## Qué hace, pieza por pieza

| Capacidad | Cómo | Dónde |
|---|---|---|
| **Crea su tokenizador** | BPE a nivel de bytes entrenado con su corpus. Amplía el vocabulario cuando asimila texto nuevo, sin cambiar los ids existentes. | `navros/tokenizer.py` |
| **Genera sus pesos** | Transformer decodificador propio (RMSNorm, NoPE/RoPE, posiciones numéricas Abacus). | `navros/model.py` |
| **Se entrena a sí misma** | Bucle proponer → resolver → verificar o votar → entrenar → seleccionar, con reversión si empeora. | `navros/core.py` |
| **Crece** | Añade capas, cabezas de atención, neuronas del MLP y vocabulario **sin cambiar su salida** (Net2Net), y sigue aprendiendo con la capacidad extra. | `NavrosLM.grow_*` |
| **Ajusta sus parámetros** | Población de tasas de aprendizaje por ronda, nº de muestras *k* adaptativo y avance automático de dificultad. | `Navros.improve_round` |
| **Asimila otros modelos** | Destilación de logits (mismo tokenizador), destilación de secuencias (cualquier modelo) y fusión de pesos. | `navros/distill.py`, `Navros.assimilate` |
| **Absorbe conocimiento** | `ingest` añade textos al corpus, amplía el tokenizador si hace falta y entrena con ellos. | `Navros.ingest` |
| **Lista para lo cuántico** | Circuito variacional diferenciable, capa híbrida, exportación a OpenQASM, gradientes *parameter-shift* y backends intercambiables. | `navros/quantum.py` |

Cómo se relaciona cada pieza con las técnicas publicadas detrás de los modelos
actuales: [`docs/AUTOMEJORA.md`](docs/AUTOMEJORA.md). Estado real de la
computación cuántica y la hoja de ruta: [`docs/CUANTICA.md`](docs/CUANTICA.md).

## Resultado medido

Primera habilidad verificable: **suma de enteros de longitud arbitraria**. Solo
se le enseñan sumas de 1 a 3 dígitos; lo demás lo aprende de sus propios
intentos. Validación en CPU con el preset `mini` (≈180k parámetros, ~15 s por
ronda):

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

```
$ python -m navros solve 12345678+87654321
12345678 + 87654321 = 99999999 ✓
$ python -m navros solve 99999999+1
99999999 + 1 = 100000000 ✓
```

En GPU, los presets `small`/`base`/`large` ejecutan el mismo bucle con modelos
de 3M a 100M+ parámetros que crecen hasta 1B.

## Inicio rápido

```bash
pip install -r requirements.txt
python -m navros init --preset mini            # CPU: tokenizador + pesos + preentrenamiento (~1 min)
python -m navros improve --rounds 10           # automejora (reanudable en cualquier momento)
python -m navros status                        # versión, linaje, exactitud por nivel
python -m navros solve 123456+987654
python -m pytest -q                            # 31 pruebas, ~5 s
```

## Dónde correrlo

| Plataforma | Preset sugerido | Cómo |
|---|---|---|
| **Kaggle** (T4 ×2 / P100, 12 h por sesión) | `small` | Importa [`deploy/kaggle/navros_kaggle.ipynb`](deploy/kaggle/navros_kaggle.ipynb). Reanuda entre sesiones con la salida anterior como input. |
| **Modal** (T4 → B200) | `base` (A10G/A100), `large` (H100) | `NAVROS_GPU=A100 modal run deploy/modal_app.py --action init --preset base`, luego `--action improve --rounds 20`. Volumen persistente; guarda cada ronda. |
| **Azure VM** | `small` (NC T4), `base`/`large` (NC A100 v4, NC H100 v5) | Ver abajo. |

### Modal

```bash
pip install modal && modal setup
NAVROS_GPU=A100 modal run deploy/modal_app.py --action init --preset base
NAVROS_GPU=A100 modal run deploy/modal_app.py --action improve --rounds 0 --hours 23
modal run deploy/modal_app.py --action status
# Automejora continua: una sesión cada 6 h que retoma donde quedó
NAVROS_SCHEDULE_HOURS=6 NAVROS_GPU=A100 modal deploy deploy/modal_app.py
```

### Azure

```bash
az group create -n navros-rg -l eastus
az vm create -g navros-rg -n navros-vm \
  --size Standard_NC24ads_A100_v4 \
  --image microsoft-dsvm:ubuntu-hpc:2204:latest \
  --priority Spot --eviction-policy Deallocate --max-price -1 \
  --os-disk-size-gb 256 --admin-username azureuser --generate-ssh-keys
# en la VM:
git clone <este repo> && cd EMERGENCY
NAVROS_PRESET=base ./deploy/azure/setup.sh --service   # automejora continua con systemd
journalctl -u navros -f
```

Una VM **Spot** cuesta mucho menos. Si Azure la desaloja, el servicio retoma
desde el último checkpoint al volver a encenderla. Tamaños: `Standard_NC4as_T4_v3`
(T4, `small`), `Standard_NC24ads_A100_v4` (A100 80 GB, `base`/`large`) y
`Standard_NC40ads_H100_v5` (H100, `large`). La imagen Ubuntu-HPC ya trae los
drivers NVIDIA.

## Comandos

| Comando | Qué hace |
|---|---|
| `init --preset {tiny,mini,small,base,large} [--mode consensus]` | Crea NAVROS desde cero |
| `improve --rounds N [--hours H]` | Rondas de automejora (`N=0`: continuo) |
| `status` | Estado, crecimiento e historial con la exactitud por nivel |
| `solve 123+456` / `generate "texto"` | Usar el modelo |
| `ingest archivos/ carpetas/` | Asimila textos y amplía el tokenizador si conviene |
| `grow {depth,mlp,heads}` | Crecimiento manual (preserva la función) |
| `assimilate --teacher runs/otro` | Aprende de otro NAVROS (destilación) |
| `merge --with runs/otro` | Fusiona pesos con otro NAVROS del mismo linaje |
| `quantum demo` / `quantum attach --qubits 4` | Simulador y QASM / capa cuántica en el modelo |
| `dashboard [--port 8777] [--host 127.0.0.1]` | Panel web local: observa en vivo y arranca/detén/borra el modelo |

Todos aceptan `--run DIR` (por defecto `runs/navros`).

### Panel local

```bash
python -m navros dashboard            # abre http://127.0.0.1:8777
```

Sin dependencias extra (solo la librería estándar) y sin recursos externos, así
que funciona offline. Se refresca solo y muestra versión, frontera, parámetros,
exactitud real por nivel, la gráfica de crecimiento y el registro en vivo. Tres
botones: **Arrancar**, **Detener** y **Desaparecer** (borra la ejecución
entera). El panel es el dueño del proceso que arranca, detecta un entrenamiento
externo ya en curso y evita lanzar un segundo escritor sobre la misma ejecución.
La caja «Ingestar» la dispara una persona (pega un texto o una URL): NAVROS no
hace peticiones de red por su cuenta.

### Modos de automejora

- **`verifier`** (por defecto): un verificador automático decide qué intentos
  son correctos. Equivale al aprendizaje por refuerzo con recompensas
  verificables de los modelos de razonamiento.
- **`consensus`**: **sin ninguna respuesta de referencia**. El modelo solo se
  queda con las respuestas en las que sus propias muestras coinciden
  (autoconsistencia). La exactitud real se mide y se informa, pero no interviene
  en ninguna decisión.

## Estructura

```
navros/
  tokenizer.py   BPE propio, extensible
  model.py       transformer + crecimiento que preserva la función + Abacus
  skills.py      habilidades verificables (suma; añade las tuyas)
  trainer.py     datos, entrenamiento (AMP, torch.compile), evaluación
  core.py        el motor de automejora, presets, ingesta y asimilación
  distill.py     destilación, licencias y fusión de pesos
  quantum.py     simulador, circuito variacional, QASM, backends
  cli.py         python -m navros …
deploy/          Modal, Azure (setup + systemd), Kaggle (notebook)
docs/            AUTOMEJORA.md, CUANTICA.md
tests/           31 pruebas
```

### Añadir una habilidad

Implementa el protocolo `Skill` de `navros/skills.py` (`make_problem`,
`prompt`, `target`, `answer_of`, `verify`, …) y regístrala en `SKILLS`.
Cualquier tarea con verificador automático sirve: aritmética, álgebra, lógica,
código con pruebas unitarias, etc.

## Límites, con franqueza

- Lo que un modelo sabe y puede hacer está acotado por sus datos, sus
  parámetros y el cómputo con el que se entrena. NAVROS automatiza cómo
  amplía las tres cosas, pero el techo lo pone el hardware. `max_params` es solo
  una protección para no agotar la memoria de la GPU.
- La automejora autónoma es fiable donde hay verificador. Para conocimiento
  general hace falta ingerir corpus grandes en GPU o destilar modelos de pesos
  abiertos con licencia permisiva.
- NAVROS no extrae ni destila a Claude ni a otros modelos comerciales: sus
  términos lo prohíben. Lo que sí implementa son los métodos publicados.
- La capa cuántica funciona en simulación y exporta circuitos a hardware real,
  pero hoy no hay ventaja cuántica demostrada para entrenar redes neuronales
  ([`docs/CUANTICA.md`](docs/CUANTICA.md)).
