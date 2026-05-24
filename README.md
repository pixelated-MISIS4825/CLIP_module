# Modulo CLIP/OpenCLIP para retrieval de ROIs X-ray

Este modulo implementa la etapa CLIP del proyecto para recuperacion semantica de regiones ROI en imagenes X-ray de equipaje. El enfoque usa exclusivamente bounding boxes ground-truth del dataset de Kaggle, por lo que esta etapa evalua la capacidad de CLIP/OpenCLIP para representar y recuperar objetos recortados sin depender todavia de detecciones predichas.

La propuesta separa dos espacios de representacion:

- Embeddings visuales generados desde crops ROI de objetos anotados.
- Embeddings textuales generados desde prompts con nombres naturales de clase.

Con estos vectores se realizan busquedas semanticas texto-a-imagen e imagen-a-imagen sobre objetos como `gun`, `knife`, `pliers`, `scissors` y `wrench`.

## Notebooks

### `01_roi_crop_extraction.ipynb`

Prepara los datos visuales del modulo. Descarga o localiza el dataset, lee las particiones `train`, `valid` y `test`, interpreta las anotaciones YOLO ground-truth y convierte las cajas normalizadas a coordenadas absolutas en pixeles. Luego aplica padding y clipping para extraer crops ROI validos. Cada crop queda asociado a su `class_id`, `class_name`, split, ruta original y propiedades geometricas de la caja.

### `02_clip_preprocessing_embeddings.ipynb`

Construye las representaciones CLIP/OpenCLIP. Carga el modelo configurado, aplica el preprocesamiento oficial a los crops y genera embeddings visuales para el banco de recuperacion usando solo imagenes de `train + valid`. Tambien genera embeddings textuales por clase usando prompts naturales, promediando los embeddings de varios prompts por cada clase y normalizando el vector final.

### `03_qdrant_indexing.ipynb`

Crea el indice semantico de busqueda. Carga los embeddings visuales y su metadata, construye una coleccion en Qdrant con distancia coseno y almacena cada vector junto con informacion util del crop. El notebook incluye funciones para buscar por texto libre, por nombre de clase natural y por imagen crop.

### `04_retrieval_evaluation.ipynb`

Evalua el comportamiento del modulo de retrieval. Incluye evaluacion `text_to_image`, donde una consulta textual de clase recupera crops visuales del banco, y evaluacion `image_to_image`, donde crops del split `test` se usan como consultas visuales contra el banco `train + valid`.

Las metricas reportadas se organizan por tipo de busqueda:

- `text_to_image` con `precision` y `recall` para `top_1`, `top_5`, `top_10`.
- `text_to_image` con `precision` y `recall` para K porcentual: `10pct_relevant`, `15pct_relevant`, `20pct_relevant`.
- `image_to_image` con `precision` y `recall` promediados por clase para `top_1`, `top_5`, `top_10`.
- `image_to_image` con `precision` y `recall` promediados por clase para K porcentual: `10pct_relevant`, `15pct_relevant`, `20pct_relevant`.

Tambien se incluyen visualizaciones de resultados recuperados, matriz de clases consultadas contra clases recuperadas y distribucion de similitud intra-clase e inter-clase.

## Clases

```python
CLASS_PROMPT_NAMES = {
    "0": "gun",
    "1": "knife",
    "2": "pliers",
    "3": "scissors",
    "4": "wrench",
}
```

