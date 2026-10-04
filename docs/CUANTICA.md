# NAVROS y la computación cuántica

## Dónde está la tecnología (2026)

- Los procesadores cuánticos actuales tienen de cientos a algunos miles de qubits
  físicos con ruido, y los primeros qubits lógicos con corrección de errores en
  fase experimental.
- **No hay ninguna ventaja cuántica demostrada para entrenar redes neuronales.**
  Hay dos obstáculos de fondo: cargar datos clásicos en estados cuánticos es caro,
  y leer resultados exige repetir el circuito miles de veces (disparos).
- Los algoritmos con aceleración probada (Shor, Grover, simulación de sistemas
  cuánticos) no se traducen directamente en entrenar transformers más rápido.

Así que ningún sistema puede prometer hoy "ser la primera IA cuántica" en un
sentido que importe. Lo que sí se puede es **tener la arquitectura lista** para
que, cuando el hardware madure, la parte cuántica se ejecute en un QPU real sin
reescribir nada. Eso es lo que hace NAVROS.

## Qué incluye NAVROS (`navros/quantum.py`)

| Componente | Para qué |
|---|---|
| Simulador de vector de estado diferenciable (PyTorch, complejos) | Entrenar circuitos de hasta ~12–14 qubits en CPU/GPU |
| `VariationalCircuit` | Codificación por ángulos + capas RY/RZ + anillo de CNOT → ⟨Z_i⟩ |
| `QuantumAdapter` + `NavrosLM.attach_quantum` | Capa híbrida clásico-cuántica conectada de forma residual al transformer. Al conectarla no cambia la salida, y el entrenamiento decide cuánto usarla. |
| `to_qasm` | Exporta el circuito a **OpenQASM 2.0**, el formato que aceptan IBM Quantum, IonQ, Rigetti, Quantinuum, Amazon Braket, etc. |
| `parameter_shift_grad` | Gradientes por *parameter-shift*, el método que funciona en hardware real, donde no hay retropropagación. Las pruebas comprueban que coincide con autograd. |
| `QuantumBackend` / `SimulatorBackend` / `CallableBackend` | Una sola interfaz: QASM entra, conteos salen. Se cambia el simulador por hardware sin tocar el modelo. |

```bash
python -m navros quantum demo --qubits 4          # simulación, muestreo, gradientes, QASM
python -m navros quantum attach --run runs/navros # conecta el adaptador al modelo
```

## Conectar hardware real (ejemplo con IBM Quantum)

No probado en este repositorio (requiere cuenta y `pip install qiskit qiskit-ibm-runtime`):

```python
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from navros.quantum import CallableBackend, VariationalCircuit

service = QiskitRuntimeService()                       # credenciales guardadas antes
device = service.least_busy(operational=True, simulator=False)

def run_ibm(qasm: str, shots: int) -> dict[str, int]:
    qc = transpile(QuantumCircuit.from_qasm_str(qasm), device)
    result = SamplerV2(mode=device).run([qc], shots=shots).result()
    return result[0].data.c.get_counts()               # 'c' = registro clásico del QASM

backend = CallableBackend(run_ibm, qubit0_last=True)   # Qiskit ordena los bits al revés
z = VariationalCircuit(4).run_on_backend(x, backend, shots=4000)
```

## Hoja de ruta cuántica

1. **Ahora:** adaptador híbrido simulado y entrenado junto al modelo; inferencia
   puntual en hardware real vía QASM.
2. **Con QPUs de 100+ qubits lógicos:** entrenar el adaptador en hardware con
   *parameter-shift* (ya implementado) y medir si aporta algo frente a una capa
   clásica del mismo tamaño. Si no aporta, desconectarlo. La decisión la toman
   los datos, no el entusiasmo.
3. **Investigación abierta:** kernels cuánticos para atención, muestreo cuántico
   para la exploración durante la automejora, y optimización combinatoria de
   arquitecturas.
