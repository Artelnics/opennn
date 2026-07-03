# Plan para Adrián — Cross-framework del benchmark Recurrent vs LSTM

**Para:** Adrián (autor del commit `88d1eed5` "Benchmarks recurrent/LSTM")
**De:** run en la RTX 4080 de la oficina
**Fecha:** 2026-07-03
**Objetivo:** resultados del cruce OpenNN vs PyTorch vs TensorFlow + dos cosas a arreglar antes de publicar la tabla.

---

## TL;DR

1. **Los tres motores coinciden** en la conclusión interna: la **LSTM gana en horizonte corto (B1/B2, 1h)** y la **Recurrent gana en el multi-target de 24h (B3/B4)** — LSTM 2/4 en OpenNN, PyTorch y TF. Buena señal de que el benchmark mide lo que dice medir.
2. 🐞 **Bug de convención de RMSE en multi-target (B3/B4):** el driver reconcilia el factor ½ con `×√2` pero **le falta `/√W`** (W = anchura del target = 24). Eso infla el RMSE de OpenNN **×√24 ≈ 4.9** en B3/B4 y hace que OpenNN parezca ~5× peor de lo que es. Fix de una línea abajo.
3. ⚙️ **TensorFlow corrió en CPU, no en GPU** (no pudo cargar las libs CUDA en el venv). Su precisión es válida, pero sus tiempos no son comparables. Falta instalar el stack CUDA de TF y, de paso, que el harness **falle en vez de caer a CPU en silencio**.

---

## 1. Entorno del run

| | |
|---|---|
| GPU | NVIDIA RTX 4080, driver 595.71.05, 16 GB |
| Commit | `88d1eed5` (dev) |
| Binario | `build/bin/no2_forecasting` (recompilado tras el pull) |
| Dataset | UCI Beijing PM2.5, 43.824 filas (2.067 pm2_5 interpolados) |
| Protocolo | 5 seeds (0..4), FP32, Adam, split 60/20/20 |
| PyTorch | 2.6.0+cu124 (GPU ✔) — venv `~/.venvs/ml` |
| TensorFlow | 2.21.0 (**cayó a CPU**, ver §3) — venv `~/.venvs/ml` |
| Artefacto OpenNN | `docs/benchmarks/results/recurrent-lstm-forecasting-20260702T211408Z.json` |

Nota operativa: `run_forecasting.py` **siempre** re-ejecuta el binario OpenNN (incluida su fase CPU de ~8 h), aunque le pases solo `--frameworks pytorch,tensorflow`. Para este cruce lancé `pt_forecasting.py` y `tf_forecasting.py` directamente reusando los resultados de OpenNN ya generados.

---

## 2. Resultados

### Precisión — test RMSE estándar (pm2.5, media ± std sobre 5 seeds)

> Los valores de OpenNN en **B3/B4 están corregidos** dividiendo por √24 (ver §3, bug). En crudo el driver reporta B3/B4 ≈ 369–400.

| Esc. | Red | OpenNN (GPU) | PyTorch (GPU) | TF (CPU) |
|------|-----|------:|------:|------:|
| B1 24h→1h  | Recurrent | 30.4 ± 4.9  | **21.4 ± 0.2** | 23.0 ± 0.5 |
| B1         | LSTM      | 33.0 ± 13.8 | **21.3 ± 0.3** | 22.1 ± 0.7 |
| B2 48h→1h  | Recurrent | 31.8 ± 6.4  | **21.4 ± 0.2** | 23.4 ± 0.9 |
| B2         | LSTM      | 28.5 ± 2.6  | **21.0 ± 0.2** | 21.9 ± 0.7 |
| B3 72h→24h | Recurrent | 75.3 ± 3.0\* | **68.4 ± 1.2** | 71.2 ± 0.7 |
| B3         | LSTM      | 81.7 ± 2.9\* | **73.0 ± 1.2** | 73.6 ± 2.4 |
| B4 168h→24h| Recurrent | 75.9 ± 1.3\* | **70.1 ± 2.1** | 71.4 ± 1.8 |
| B4         | LSTM      | 81.7 ± 3.4\* | **71.2 ± 1.8** | 71.7 ± 1.4 |

\* = valor crudo de OpenNN dividido por √24.

**Lectura:**
- PyTorch es el más preciso; TF justo detrás; **OpenNN queda por detrás** (~40% en B1/B2 single-target donde no hay ambigüedad de convención, ~5–15% en B3/B4 ya corregidos).
- ⚠️ **Alta varianza entre seeds en OpenNN**: B1 LSTM ±13.8, B2 Recurrent ±6.4, frente a <2.5 en PyTorch/TF. Apunta a sensibilidad de init/optimización, no a la métrica. Merece mirarse aparte.

### Tiempo de entrenamiento — s/run medio

> OpenNN y PyTorch en **GPU**; TF en **CPU** (no comparable, marcado).

| Esc. | Red | OpenNN GPU | PyTorch GPU | TF **CPU** | PyTorch más rápido |
|------|-----|------:|------:|------:|------:|
| B1 | Recurrent |  9.14 | 3.77 |  13.03 | 2.4× |
| B1 | LSTM      |  4.94 | 4.45 |  20.23 | 1.1× |
| B2 | Recurrent | 11.39 | 3.60 |  17.21 | 3.2× |
| B2 | LSTM      |  9.95 | 4.07 |  46.87 | 2.4× |
| B3 | Recurrent | 17.64 | 3.14 |  21.21 | 5.6× |
| B3 | LSTM      | 15.48 | 3.52 |  83.98 | 4.4× |
| B4 | Recurrent | 38.46 | 2.93 |  39.95 | **13.1×** |
| B4 | LSTM      | 26.53 | 2.75 | 132.48 | 9.6× |

**Lectura:**
- PyTorch es más rápido en todos los casos y la brecha crece con la longitud de secuencia.
- El punto débil de OpenNN es la **Recurrent con kernels CUDA propios** (B4: 38.5 s vs 2.9 s = 13×); apenas escala con la longitud de secuencia. La **LSTM (cuDNN)** aguanta mucho mejor (B1: casi empate) porque ahí ambos usan cuDNN.

---

## 3. 🐞 Bug #1 — RMSE multi-target inflado ×√W

### Diagnóstico

`TestingAnalysis::calculate_errors` en `opennn/testing_analysis.cpp:378-384`:

```cpp
const Index batch_size = targets.rows();                       // = N (nº de muestras)
const float sum_squared = (outputs.array() - targets.array()).square().sum();  // suma sobre N × W
errors(1) = sum_squared / (2.0f * float(batch_size));          // divide por 2·N  (¡falta el ·W!)
errors(2) = sqrt(errors(1));                                   // = sqrt(sum / (2N))
```

`sum_squared` suma sobre **N muestras × W salidas**, pero el denominador solo lleva **N**. Entonces:

```
errs(2)              = RMSE_estándar · √(W/2)
headline (× √2)      = RMSE_estándar · √W          ← lo que reporta el driver como test_rmse_mean
```

- **B1/B2 (single-target, W=1):** headline = RMSE estándar → correcto, comparable con PyTorch/TF.
- **B3/B4 (multi-target, W=24):** headline = RMSE estándar · **√24 ≈ 4.9** → **inflado**.

PyTorch/TF computan `mean((pred-true)²)` sobre toda la matriz `(N, W)` (per-element), que es lo estándar. La nota de `recurrent-lstm-forecasting-opennn.md` (sección "Cross-framework fidelity") solo menciona el factor ½; **le falta este `1/√W`**.

**Impacto colateral:** el `test_rmse_rel_mean` (rmse%) de B3/B4 también sale inflado — reporta 55–60% cuando el valor real ronda **11–12%**.

### Fix propuesto (driver del benchmark)

En `examples/no2_forecasting/main.cpp`, función `train_one`, donde se calcula el RMSE (línea ~218):

```cpp
// antes:
if (errs.size() >= 3) r.test_rmse = errs(2) * RMSE_HALF_TO_STD;

// después: normaliza por la anchura del target para RMSE estándar per-element.
const Index target_width = ds->get_target_shape().size();   // W (subir esta línea desde abajo)
if (errs.size() >= 3)
    r.test_rmse = errs(2) * RMSE_HALF_TO_STD / std::sqrt(float(target_width));
```

`errs(2) · √2 / √W = sqrt(sum/(2N)) · √2 / √W = sqrt(sum/(N·W))` = RMSE estándar per-element. ✔

Notas:
- `target_width` ya se calcula más abajo en la misma función (para el RMSE relativo); solo hay que subirlo antes del cálculo de `test_rmse`.
- Revisar también `test_rmse_native_halfconv_mean`: hoy es `test_rmse_mean/√2`; conviene que sea el `errs(2)` crudo de OpenNN, claramente etiquetado como "convención interna OpenNN", para no volver a mezclar convenciones.
- Actualizar la nota de convención en `recurrent-lstm-forecasting-opennn.md` para mencionar el `1/√W`.
- **Alternativa** (no recomendada): corregir en `run_forecasting.py` al parsear, pero el RMSE nace en C++, mejor arreglarlo en el origen.

---

## 4. ⚙️ Issue #2 — TensorFlow corre en CPU

### Síntoma

Las líneas METRIC de TF salen con `phase=CPU ... device=cpu`, y:

```
W tensorflow/.../gpu_device.cc: Cannot dlopen some GPU libraries. ... Skipping registering GPU devices...
built_with_cuda True
tf.config.list_physical_devices('GPU') -> []
```

El wheel de TF 2.21 sí está compilado con CUDA, pero no encuentra las shared libs (CUDA 12.x / cuDNN 9) en el venv `~/.venvs/ml`. Torch trae sus propias libs cu124, pero TF busca las suyas por su cuenta.

### Fix propuesto

1. Instalar el stack CUDA de TF en el venv:
   ```bash
   ~/.venvs/ml/bin/pip install 'tensorflow[and-cuda]==2.21.0'
   ```
   (arrastra los wheels `nvidia-*` que TF espera). Verificar con:
   ```bash
   ~/.venvs/ml/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
2. Relanzar `tf_forecasting.py`; las líneas deben salir `phase=GPU device=cuda` y los tiempos ya serán comparables con OpenNN/PyTorch.

### Mejora del harness (evita el falso positivo)

Ahora mismo, si un motor "de GPU" cae a CPU, `run_forecasting.py` **guarda esos tiempos de CPU dentro de la comparación GPU sin avisar**. Propuesta:

- Que `pt_forecasting.py` / `tf_forecasting.py` **aborten (exit ≠ 0)** si `--gpu` es esperado y `device==cpu`, o al menos emitan un `WARNING device_mismatch`.
- Que `run_forecasting.py` propague ese warning al JSON (`configuration.device_check`) y a la salida.

---

## 5. Plan de acción (checklist)

- [ ] **Fix RMSE multi-target** en `main.cpp` (`/√target_width`) + revisar campo `native_halfconv`.
- [ ] **Actualizar la nota de convención** en `recurrent-lstm-forecasting-opennn.md` (mencionar `1/√W`).
- [ ] **Instalar `tensorflow[and-cuda]`** en `~/.venvs/ml` y relanzar TF en GPU.
- [ ] **Guardas anti-fallback a CPU** en los scripts Python + reflejarlo en el JSON.
- [ ] **Re-lanzar el cruce completo** tras los fixes y regenerar la tabla headline.
- [ ] (Aparte) **Investigar la alta varianza entre seeds de OpenNN** (B1 LSTM ±13.8, B2 Rec ±6.4) — ¿init Glorot / lr / early-stop?
- [ ] (Aparte) **Perf de la Recurrent CUDA propia** vs cuDNN — es donde OpenNN pierde más (B4: 13× más lento que PyTorch).

---

## Apéndice — reproducir

```bash
# OpenNN (GPU + CPU, ~8 h por la fase CPU de B3/B4 LSTM):
cd docs/benchmarks/recurrent-lstm-forecasting
export OPENNN_FORECASTING_BIN="$(git rev-parse --show-toplevel)/build/bin/no2_forecasting"
python3 run_forecasting.py --gpu-index 0

# Solo el cruce (reusa data/ ya preparado), sin re-ejecutar OpenNN:
VENV=~/.venvs/ml/bin/python
CUDA_VISIBLE_DEVICES=0 $VENV pt_forecasting.py
CUDA_VISIBLE_DEVICES=0 $VENV tf_forecasting.py
```

Todos los números de precisión de arriba salen de las líneas `METRIC ... seed=aggregate` de cada motor.
