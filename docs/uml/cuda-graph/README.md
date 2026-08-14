# CUDA Graph real de entrenamiento Transformer en OpenNN

Este directorio contiene una captura real, dependiente de la configuracion, del
`cudaGraph_t` que OpenNN instancia durante el entrenamiento de un Transformer.
No es una reconstruccion conceptual: el archivo DOT procede directamente de
`cudaGraphDebugDotPrint(..., cudaGraphDebugDotFlagsVerbose)`.

## Artefactos

- `opennn-transformer-training.cuda.dot`: salida original y detallada de CUDA.
- `opennn-transformer-training.full.puml`: UML integral con todos los campos
  emitidos por CUDA, incluidas direcciones, identificadores, dimensiones de
  lanzamiento y parametros de las operaciones de memoria.
- `opennn-transformer-training.topology.puml`: el mismo UML integral con
  etiquetas compactas. Conserva exactamente todos los nodos y dependencias.
- `parts/*.puml` y `parts/*.svg`: vistas UML renderizables del staging H2D y de
  cada paso de entrenamiento. Son subgrafos inducidos exactos, no resumenes.

El UML integral tiene 1.744 nodos y 1.750 dependencias:

| Tipo de nodo CUDA | Cantidad |
| --- | ---: |
| `KERNEL` | 1.440 |
| `MEMCPY` | 64 |
| `MEMSET` | 176 |
| `MEM_ALLOC` | 32 |
| `MEM_FREE` | 32 |

La particion observada es:

| Seccion capturada | Nodos CUDA | Nodos | Dependencias internas |
| --- | ---: | ---: | ---: |
| Staging H2D | 0..23 | 24 | 23 |
| Paso 1 | 24..238 | 215 | 214 |
| Paso 2 | 239..453 | 215 | 214 |
| Paso 3 | 454..668 | 215 | 214 |
| Paso 4 | 669..883 | 215 | 214 |
| Paso 5 | 884..1098 | 215 | 214 |
| Paso 6 | 1099..1313 | 215 | 214 |
| Paso 7 | 1314..1528 | 215 | 214 |
| Paso 8 | 1529..1743 | 215 | 214 |

Hay 15 dependencias entre secciones: ocho enlazan el staging con el primer nodo
de cada paso y siete serializan el final de un paso con el comienzo del
siguiente. Estas dependencias estan en los dos UML integrales. Al ser subgrafos
inducidos, no aparecen dentro de los SVG individuales.

## Captura reproducible

La captura se realizo en una NVIDIA GeForce RTX 5070 Ti con driver 595.71.05,
usando el benchmark de entrenamiento de OpenNN y esta configuracion concreta:

- BF16 y atencion SDPA de cuDNN activadas.
- 1.024 muestras; secuencias de entrada y decoder de longitud 256.
- Transformer encoder-decoder con `d_model=32`, 4 cabezas, FFN 64 y una capa.
- Batch 128: ocho batches, por lo que se activa el mega-graph de ocho pasos.
- Cross-entropy 3D, clipping de gradiente y actualizacion Adam capturable.

Comandos:

```bash
cmake -S . -B build-refactor-cuda \
  -DOpenNN_BUILD_ATTENTION_SPEED_BENCHMARKS=ON
cmake --build build-refactor-cuda \
  --target opennn_transformer_train -j2

OPENNN_CUDA_GRAPH_DOT="$PWD/docs/uml/cuda-graph/opennn-transformer-training.cuda.dot" \
OPENNN_BF16=1 OPENNN_SDPA_MIN=1 \
build-refactor-cuda/bin/opennn_transformer_train \
  docs/benchmarks/attention-speed/corpus.txt 32 4 64 1 128 0
```

`epochs=0` ejecuta la epoca de indice cero; OpenNN recorre los ocho batches y
captura el grafo durante ese entrenamiento.

Para convertir de nuevo el DOT sin alterar su topologia:

```bash
python3 tools/cuda_dot_to_plantuml.py \
  docs/uml/cuda-graph/opennn-transformer-training.cuda.dot \
  docs/uml/cuda-graph/opennn-transformer-training.topology.puml \
  --labels compact --split-dir docs/uml/cuda-graph/parts

python3 tools/cuda_dot_to_plantuml.py \
  docs/uml/cuda-graph/opennn-transformer-training.cuda.dot \
  docs/uml/cuda-graph/opennn-transformer-training.full.puml \
  --labels full
```

El diagrama integral excede la profundidad que el layout Smetana de PlantUML
1.2026.6 puede resolver en una sola imagen. Por eso se entrega como fuente UML
integral y como nueve SVG exactos y navegables. La limitacion es del layout, no
de la captura ni de la conversion.

## Integridad

SHA-256 de los artefactos integrales:

```text
259377db2c4fa09d62dbea36174e0d16a609c491e3fd5a552f659c742b50b90b  opennn-transformer-training.cuda.dot
1826faa4db256ade7a59dffba7087a40296e78df3ba2e80859c4f309db0a5bfc  opennn-transformer-training.topology.puml
892399ab50946b407f3d373f8e0b91fce4387c6f6b9331de3a1169ae1139a303  opennn-transformer-training.full.puml
```
