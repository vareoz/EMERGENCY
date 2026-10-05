# NAVROS como agente: peticiones HTTP

Un agente es un bucle: el modelo propone una acción, un programa la ejecuta y
le devuelve lo que pasó. Aquí la acción es una **petición HTTP**. El modelo solo
escribe texto; **quien decide qué se ejecuta es el operador**, mediante una
política. Nada de lo que el modelo escriba, ni de lo que lea en una página,
puede ampliarla.

| Pieza | Qué hace | Dónde |
|---|---|---|
| Ejecutor seguro | Aplica la política, conecta y devuelve la respuesta (solo biblioteca estándar) | `navros/httptool.py` |
| Protocolo y bucle | Texto ↔ acciones, observaciones, límites de pasos, adaptador a NAVROS | `navros/agent.py` |
| Habilidad `http` | Enseña a NAVROS a escribir la petición correcta, con verificador automático | `navros/skills.py` |
| CLI | `python -m navros agent …` | `navros/cli.py` |

## Protocolo

Texto plano, que es HTTP/1.1 sin adornos: un transformer pequeño lo aprende y los
modelos grandes ya lo conocen.

```
tarea  → obtén api.example.com/v1/items con id=7
modelo → <http>
         GET https://api.example.com/v1/items?id=7
         accept: application/json
         </http>
arnés  → <result>
         200 OK
         content-type: application/json

         {"id": 7, "nombre": "tuerca"}
         </result>
modelo → <final>El item 7 es una tuerca.</final>
```

Primera línea `MÉTODO URL`; luego cabeceras `nombre: valor`; una línea en blanco y el
cuerpo (opcional). Con un modelo que no conoce el formato, pásale
`preamble=PROTOCOL_HELP` al `Agent`.

## Qué impide la política

Con `HttpPolicy()` por defecto **no se permite nada**: hay que listar los hosts.

| Riesgo | Defensa |
|---|---|
| El modelo (o una página que lee) pide `http://169.254.169.254/` (credenciales de la nube), `localhost`, `10.x`… (**SSRF**) | Hosts permitidos explícitos **y** todas las IP resueltas deben ser públicas (también `::ffff:127.0.0.1`, NAT64, 6to4…) |
| El DNS responde una IP pública al comprobar y otra privada al conectar (*DNS rebinding*) | Se resuelve **una vez** y se conecta a esa IP; el certificado TLS se valida contra el nombre original |
| `https://permitido.com@malo.com/`, `malo.com\.permitido.com`, `malo.com/.permitido.com` | Se rechazan `@`, caracteres de control y nombres que no sean un hostname válido; el comodín `*.x.com` no incluye `x.com` |
| Una redirección lleva a otro host o a la red interna | Solo se siguen redirecciones de `GET`/`HEAD`, a mano, y **cada salto pasa la política completa** (máx. 3) |
| El modelo filtra una clave o usa la de otro | No puede fijar `authorization`, `cookie`, `host`, `content-length`…; las credenciales (`credentials`) las pone la herramienta, solo al host exacto y **solo por https** |
| Efectos secundarios no deseados | Solo `GET` y `HEAD`, solo https y solo los puertos 80/443, hasta que el operador lo amplíe |
| Respuesta enorme o servidor lento | Tope de bytes **leídos** (no solo mostrados), tiempo por operación y tiempo total |
| La página imita `</result>` o `<final>` para tomar el turno | Solo se leen acciones de lo que genera el modelo; en lo recibido se escapa `<`; de cada turno se ejecuta la **primera** acción y se descarta lo que el modelo escriba después (p. ej. un `<result>` inventado) |
| Rastro de lo que hizo | Auditoría JSONL (`agent.jsonl`): método, URL **sin consulta**, estado, bytes, ms, errores. Sin cabeceras ni cuerpos, para no registrar secretos |

### Lo que la política **no** resuelve

- **Inyección de instrucciones (*prompt injection*).** Una página permitida puede
  contener texto como «ignora tu tarea y pide…». El modelo puede hacerle caso. La
  política **acota el daño** (solo hosts y métodos permitidos, nada interno), pero no
  evita que el modelo se desvíe. Por eso: no permitas hosts que sirvan contenido de
  terceros, y activa `POST`/`PUT`/`DELETE` solo si hace falta de verdad.
- **Los hosts permitidos son de confianza.** Una petición a un host permitido puede
  llevar en la URL lo que el modelo sepa. Es lo esperable, pero ten en cuenta qué
  contiene su contexto.
- **No usa `HTTPS_PROXY`.** Fijar la IP exige conectar directamente. En una red
  donde la salida solo es posible por proxy, hará falta adaptar `_connect_any`.
- **Un modelo diminuto no «entiende» una tarea abierta.** Ver «Qué aprende hoy».

## Uso

```bash
python -m navros init --preset small --skill http      # aprende el formato (ver abajo)
python -m navros improve --rounds 20
python -m navros solve "obtén api.example.com/v1/items con id=7"

python -m navros agent "obtén api.example.com/v1/items con id=7" \
    --allow api.example.com --steps 4
# más permisos, siempre explícitos:
#   --allow '*.example.com'   --method POST   --port 8443   --insecure-http
#   --bearer api.example.com=MI_TOKEN   (envía «Authorization: Bearer $MI_TOKEN» solo a ese host)
```

Sale con código distinto de 0 si el agente no llegó a una respuesta `<final>`. La
auditoría queda en `runs/navros/agent.jsonl` (`--log` para cambiarla).

Desde Python, con cualquier modelo (una función `prompt -> continuación`):

```python
from navros.agent import Agent, NavrosAgentModel, PROTOCOL_HELP
from navros.httptool import HttpPolicy, HttpTool

policy = HttpPolicy(allow_hosts=("api.example.com",),
                    credentials={"api.example.com": {"authorization": "Bearer ..."}})
tool = HttpTool(policy, log_path="agent.jsonl")

agent = Agent(NavrosAgentModel(nav), tool, max_steps=4)           # NAVROS
agent = Agent(mi_llm, tool, preamble=PROTOCOL_HELP, max_steps=6)  # cualquier otro modelo
result = agent.run("obtén api.example.com/v1/items con id=7")
result.answer, result.stop, result.steps, result.transcript
```

## La habilidad `http`: cómo aprende NAVROS a escribir peticiones

Es una habilidad más de `navros/skills.py`, con el mismo bucle de automejora que la
suma: el modelo propone una petición para una orden en español, un **verificador
automático** comprueba que coincide exactamente con la esperada (método, URL, y sin
cabeceras ni cuerpo de más) y entrena con lo que acertó. Dificultad = nº de
parámetros de consulta. El verificador usa el mismo analizador que el agente, así que
lo que el modelo aprende a escribir es exactamente lo que el arnés sabe ejecutar.

El prompt de la habilidad es **idéntico** al que `Agent` le da al modelo en el primer
turno (la tarea + salto de línea), de modo que lo aprendido se aplica tal cual.

> Usa un preset `small` o mayor. Con `tiny`/`mini` la ventana (64/96 tokens) no
> cabe una petición y la generación se corta antes de terminar; `init` te avisa.

## Resultado medido

Solo se le enseñan órdenes con **1 a 3** parámetros de consulta; el resto lo aprende de
sus propios intentos verificados. CPU, arquitectura `mini` (188k parámetros) con la
ventana ampliada a 192 tokens, 4 rondas (≈18 min en total):

| Versión | Frontera | 3 par. | 4 par. | 5 par. | 6 par. | 7 par. |
|---|---|---|---|---|---|---|
| v0: solo se le enseñó 1–3 | 4 | 100 % | 4 % | 0 % | — | — |
| v1 | 4 | 99 % | 87 % | 0 % | — | — |
| v2 | 5 | 100 % | 98 % | 8 % | 0 % | — |
| v3 | 6 | 100 % | 99 % | 95 % | 3 % | 0 % |
| **v4** | **6** | **100 %** | **99 %** | **95 %** | **26 %** | **0 %** |

Generó y verificó él mismo 932 ejemplos (`pool.jsonl`, origen `self:verifier`): 493 de
4–6 parámetros, es decir, más allá de lo que se le enseñó, y el resto es repaso de 1–3.

Cómo leerlo, con sus límites: una sola ejecución, una semilla, 100 problemas de
evaluación por nivel (≈ ±5 puntos); el nivel 6 aún no está dominado y el 7 no se ha
alcanzado. Lo que demuestra es que **el formato de la petición se aprende y se extiende
solo con un verificador**, no que NAVROS sepa decidir qué pedir.

Para repetirlo (el preset `mini` por defecto tiene una ventana de 96 tokens, que no basta):

```python
from navros.core import Navros, preset_config
cfg = preset_config("mini", skill="http")
cfg.model.max_seq_len, cfg.tokenizer_merges, cfg.pretrain_steps = 192, 300, 1500
cfg.improve.problems_per_round, cfg.improve.steps_per_round, cfg.improve.eval_problems = 512, 300, 100
nav = Navros.create("runs/http", cfg, corpus=["README.md", "docs"])
nav.improve(4)
print(nav.solve("obtén api.example.com/v1/items con id=7 lang=es"))
# obtén api.example.com/v1/items con id=7 lang=es → GET https://api.example.com/v1/items?id=7&lang=es ✓
```

## Qué aprende hoy, con franqueza

La habilidad enseña el **primer paso** (escribir una petición bien formada a partir
de una orden con forma fija). El bucle, el ejecutor y la política son independientes
del modelo y funcionan con cualquier otro que hable el protocolo.

Lo que **falta** para un agente autónomo con NAVROS, y es trabajo de modelo, no de
andamiaje:

1. **Leer el resultado y redactar `<final>`.** El historial del segundo turno
   (`tarea + petición + <result>…`) no es parte de la habilidad actual. Se añade como
   una segunda habilidad (p. ej. «extrae el campo `id` de un JSON») o como segundo tipo
   de problema de `HttpCallSkill`, con ventana suficiente para el cuerpo.
2. **Tareas abiertas en lenguaje natural.** Decidir *qué* pedir sin una plantilla
   exige un modelo mucho mayor que los presets actuales. El mismo bucle sirve con uno
   de pesos abiertos y licencia permisiva si lo pasas como `model`.
3. **Varias herramientas.** `Agent` ejecuta un solo tipo de acción (`<http>`). Añadir
   otra es una etiqueta más en `_TAGS` y un ejecutor con su propia política.
