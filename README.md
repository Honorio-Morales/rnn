# 🌐 Traductor Automático Español-Inglés con RNN

**Proyecto académico:** Implementación de un traductor automático usando Redes Neuronales Recurrentes (RNN/LSTM) bajo la metodología **CRISP-ML(Q)**.

## 📋 Descripción

Sistema de traducción automática bidireccional (Español ↔ Inglés) basado en arquitectura **Seq2Seq** con atención de Bahdanau. Optimizado para ejecutarse en GPUs con ≤ 6GB de memoria.

**Objetivo:** Alcanzar BLEU Score ≥ 0.35

---

## 📁 Estructura del Proyecto

```
traductor-rnn/
├── docs/
│   └── proyecto_crisp_ml.tex          # Documento principal (6 fases CRISP-ML)
├── data/
│   ├── raw/                           # Dataset original
│   ├── processed/                     # Datos procesados y vocabularios
│   └── splits/                        # Train/Val/Test codificados
├── notebooks/
│   ├── 01_exploration.ipynb           # EDA y preparación de datos
│   ├── 02_model_training.ipynb        # Entrenamiento (TO DO)
│   └── 03_evaluation.ipynb            # Evaluación (TO DO)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Limpieza, tokenización
│   ├── model.py                       # Arquitectura Seq2Seq
│   ├── metrics.py                     # BLEU Score, análisis de errores
│   └── inference.py                   # Predicción
├── api/
│   ├── app.py                         # FastAPI (TO DO)
│   └── requirements.txt
├── models/
│   └── (modelos entrenados)
├── requirements.txt                   # Dependencias
└── README.md                          # Este archivo
```

---

## 🚀 Quick Start

### 1. Instalación

```bash
# Clonar repositorio
cd /home/honorio/IA/rnn

# Crear entorno virtual (opcional)
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Preparación de Datos

Ejecutar el notebook `notebooks/01_exploration.ipynb`:

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

Esto generará:
- `data/processed/vocab_spanish.json`
- `data/processed/vocab_english.json`
- `data/splits/X_train.npy`, `y_train.npy`, etc.

### 3. Entrenar Modelo

```bash
# (Próximo paso: Crear notebook 02_model_training.ipynb)
jupyter notebook notebooks/02_model_training.ipynb
```

### 4. Evaluar y Traducir

```bash
# (Próximo paso: Crear notebook 03_evaluation.ipynb)
jupyter notebook notebooks/03_evaluation.ipynb
```

---

## 🏗️ Arquitectura del Modelo

### Componentes

**Encoder:** LSTM Bidireccional
- Capas: 2
- Unidades: 256 cada una
- Embeddings: 128 dimensiones
- Dropout: 0.3

**Decoder:** LSTM Unidireccional + Atención Bahdanau
- Capas: 2
- Unidades: 256 cada una
- Atención: Mecanismo de Bahdanau
- Dropout: 0.3

**Parámetros Globales**
- Batch size: 32
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Max epochs: 50
- Early stopping: Paciencia 5

### Optimizaciones para GPU 6GB
- Embeddings pequeños (128 vs 512)
- Hidden units reducidos (256 vs 512)
- Batch size controlado (32)
- Secuencias truncadas (max 30 tokens)
- Vocabulario limitado (10k palabras)

---

## 📊 Metodología: CRISP-ML(Q)

### Fase 1: Business & Data Understanding
- Métrica: BLEU Score ≥ 0.35
- Dataset: Pares Español-Inglés (Tatoeba)
- Usuarios: Comunidad UAC

### Fase 2: Data Preparation
- Limpieza de texto
- Tokenización
- Construcción de vocabularios
- Padding/Truncamiento

### Fase 3: Modeling
- Arquitectura Seq2Seq
- Encoder-Decoder
- Mecanismo de Atención

### Fase 4: Evaluation
- BLEU Score
- Análisis de errores
- Matriz de confusión

### Fase 5: Deployment
- Exportación a SavedModel
- API REST (FastAPI)
- Containerización (Docker)

### Fase 6: Monitoring & Maintenance
- Detección de drift
- Feedback loop
- Reentrenamiento automático

---

## 📝 Módulos Principales

### `preprocessing.py`
```python
from src.preprocessing import TextPreprocessor, ParallelDataProcessor

# Crear preprocesador
processor = ParallelDataProcessor(vocab_size=10000, max_length=30)
processor.build_vocabs(spanish_texts, english_texts)

# Codificar pares
source_idx, target_idx = processor.encode_pair("hola", "hello")
```

### `model.py`
```python
from src.model import create_seq2seq_model

# Crear modelo
model = create_seq2seq_model(
    source_vocab_size=10000,
    target_vocab_size=10000,
    max_length=30
)

# Entrenar
model.fit((X_train, y_train), y_train, epochs=50, validation_split=0.1)
```

### `metrics.py`
```python
from src.metrics import BLEUScore, ErrorAnalyzer

# Calcular BLEU
bleu = BLEUScore()
score = bleu.compute_corpus_bleu(generated_corpus, reference_corpus)

# Analizar errores
analyzer = ErrorAnalyzer()
stats = analyzer.analyze_batch(generated_batch, reference_batch)
```

### `inference.py`
```python
from src.inference import TranslationInference

# Cargar modelo
infer = TranslationInference(
    'models/traductor_v1.h5',
    'data/processed/vocab_spanish.json',
    'data/processed/vocab_english.json'
)

# Traducir
translation, confidence = infer.translate("Hola, ¿cómo estás?")
```

---

## 📈 Métricas de Evaluación

| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| BLEU Score | ≥ 0.35 | Similitud con referencia |
| Exact Match | ≥ 10% | Coincidencia exacta |
| Latencia | < 500 ms | Tiempo por traducción |
| Cobertura OOV | ≤ 20% | Palabras desconocidas |

---

## 🔧 Próximos Pasos

- [ ] Crear `02_model_training.ipynb`
- [ ] Crear `03_evaluation.ipynb`
- [ ] Implementar API REST (FastAPI)
- [ ] Crear Dockerfile
- [ ] Agregar logging y monitoring
- [ ] Implementar feedback loop
- [ ] Agregar soporte para más idiomas

---

## 📚 Referencias

- Vaswani et al. (2017). *Attention is All You Need*. NeurIPS
- Papineni et al. (2002). *BLEU: Automatic Evaluation of Machine Translation*. ACL
- [Tatoeba Project](https://tatoeba.org/)
- [CRISP-ML(Q)](https://crisp-ml.github.io/)

---

## 👤 Autor

Universidad Andina del Cusco (UAC)  
Proyecto Académico - Traducción Automática con RNN

---

## 📄 Licencia

MIT License - Proyecto educativo
