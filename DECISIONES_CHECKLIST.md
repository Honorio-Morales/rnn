# 📋 Decisiones Clave del Proyecto + Checklist de Entregables

## 🤔 Preguntas Clave a Responder

### 1️⃣ ¿Qué Corpus Tienes? (¿Dónde consigues pares Español-Inglés/Quechua?)

#### OPCIONES:

**A) DATASET PÚBLICO (RECOMENDADO)**
- **Tatoeba Project** - https://tatoeba.org/
  - ✓ Gratis, 50K+ pares Español-Inglés
  - ✓ Fácil de descargar
  - ✓ Oraciones cortas-medianas
  - Descargar: `https://downloads.tatoeba.org/exports/`

- **OPUS Corpus** - https://opus.nlpl.eu/
  - ✓ Millones de pares paralelos
  - ✓ Multilingüe (Español-Quechua disponible limitado)
  - ✓ Dominio variado

- **CCNet** - Facebook Research
  - ✓ Web crawl multilingüe
  - ✓ Alta calidad
  - ✓ Costo: tiempo de descarga

**B) DATASET PEQUEÑO LOCAL (PARA PROTOTIPO)**
- ✓ `data/raw/translation_pairs.csv` (100 pares de ejemplo)
- ✓ Generado automáticamente en `01_exploration.ipynb`
- ⚠️ Solo para demostración (BLEU muy bajo)

**C) DATASET PERSONALIZADO**
- Recolectar manualmente de libros, películas, documentos
- Usar Google Translate + validación manual
- ⚠️ Más tiempo, mejor calidad

**RECOMENDACIÓN:** Usar **Tatoeba (Opción A)** para proyecto serio
- Descarga rápida (~50MB)
- Calidad garantizada
- BLEU Score alcanzable (0.35-0.45)

---

### 2️⃣ ¿Español ↔ Inglés o Quechua?

#### ANÁLISIS COMPARATIVO:

| Aspecto | Español-Inglés | Español-Quechua |
|---------|----------------|-----------------|
| **Corpus disponible** | 50K+ pares | <5K pares |
| **Dificultad** | Fácil ✓ | Difícil ✗ |
| **BLEU esperado** | 0.35-0.50 | 0.10-0.25 |
| **Tiempo entrenamiento** | 10-20 min | 20-40 min |
| **Éxito proyecto** | Alto | Medio-Bajo |
| **Impacto académico** | General | Regional (UAC) |

**RECOMENDACIÓN:** **Español-Inglés**
- ✓ Corpus abundante
- ✓ Modelo más robusto
- ✓ Cumple BLEU Score objetivo
- ✓ Demostración clara para rúbrica

**Nota:** Puedes entrenar Español-Inglés y luego experimentar con Quechua si tienes tiempo.

---

### 3️⃣ ¿BLEU Score Objetivo?

#### ESCALA DE BLEU SCORES:

```
BLEU < 0.15  →  Muy pobre (inaceptable)
BLEU 0.15-0.25  →  Pobre (literal)
BLEU 0.25-0.35  →  Aceptable (nuestro mínimo)
BLEU 0.35-0.50  →  Bueno ✓ (RECOMENDADO)
BLEU 0.50-0.70  →  Muy bueno (difícil de lograr)
BLEU > 0.70  →  Excelente (producción, Google Translate ~0.80)
```

**RECOMENDACIÓN:** **BLEU ≥ 0.35**
- ✓ Realista con 6GB GPU
- ✓ Alcanzable con Español-Inglés + Tatoeba
- ✓ Suficiente para demostrar competencia
- ✓ Según rúbrica PDF: "aceptable" = 17-20

---

## ✅ CHECKLIST: ¿Qué Falta del Proyecto?

### 📄 **ENTREGABLES SEGÚN PDF:**

```
1. Documento Maestro CRISP-ML(Q)
   ✅ HECHO: docs/proyecto_crisp_ml.tex (437 líneas)
   - 6 fases documentadas
   - Parámetros, arquitectura, métricas
   - Plan de monitoreo
   
2. Notebook de Entrenamiento
   ✅ PARCIAL: notebooks/01_exploration.ipynb (EDA)
   ❌ FALTA: notebooks/02_model_training.ipynb
      - Cargar datos preparados
      - Entrenar modelo (50 epochs)
      - Graficar Loss/Accuracy
      - Salvar modelo .h5
   
   ❌ FALTA: notebooks/03_evaluation.ipynb
      - Cargar modelo
      - Calcular BLEU Score
      - Análisis de errores (OOV, contexto, orden)
      - Ejemplos de traducción
   
3. Prototipo Funcional
   ❌ FALTA: api/app.py (FastAPI)
      - Endpoint POST /translate
      - Validación de entrada
      - Manejo de errores
      - Respuesta JSON
   
   ❌ FALTA: api/requirements.txt
   ❌ FALTA: Dockerfile (opcional)
   ❌ FALTA: Demo web simple (HTML/JS)
```

### 🏗️ **COMPONENTES TÉCNICOS:**

```
CÓDIGO BASE (✅ LISTO)
├── preprocessing.py ✅
│   └── TextPreprocessor, ParallelDataProcessor
├── model.py ✅
│   └── Encoder, Decoder, Seq2SeqModel
├── metrics.py ✅
│   └── BLEUScore, ErrorAnalyzer
└── inference.py ✅
    └── TranslationInference

DATOS (✅ PARCIAL)
├── data/raw/translation_pairs.csv ✅ (ejemplo 100 pares)
├── data/processed/ ❌ (FALTA generar)
│   ├── vocab_spanish.json (se genera en 01)
│   └── vocab_english.json (se genera en 01)
└── data/splits/ ❌ (FALTA generar)
    ├── X_train.npy, y_train.npy
    ├── X_val.npy, y_val.npy
    └── X_test.npy, y_test.npy

MODELOS (❌ FALTA)
├── models/traductor_v1.h5
├── models/metrics_training.json
└── models/attention_weights.pkl

DOCUMENTACIÓN (✅ CASI LISTA)
├── docs/proyecto_crisp_ml.tex ✅
├── README.md ✅
└── docs/API_specification.md ❌
```

---

## 📊 **PLAN DE ACCIÓN (orden recomendado):**

### **SEMANA 1: Datos + Entrenamiento**
1. [ ] Descargar Tatoeba (~30 min)
2. [ ] Ejecutar `01_exploration.ipynb` (~20 min)
   - Genera vocabularios y datos codificados
3. [ ] Crear + ejecutar `02_model_training.ipynb` (~60 min)
   - Entrenar modelo 50 epochs
   - Guardar modelo.h5

### **SEMANA 2: Evaluación + API**
4. [ ] Crear + ejecutar `03_evaluation.ipynb` (~40 min)
   - Calcular BLEU Score
   - Análisis de errores
5. [ ] Crear `api/app.py` (~60 min)
   - Endpoint /translate
   - Validación
   - Testing con curl

### **SEMANA 3: Refinamiento**
6. [ ] Crear demo web simple (opcional) (~30 min)
7. [ ] Dockerfile (opcional) (~20 min)
8. [ ] Subir a GitHub (~10 min)

---

## 🎯 **DECISIÓN RECOMENDADA:**

**COMBINA ESTAS OPCIONES:**

| Decisión | Opción | Razón |
|----------|--------|-------|
| Corpus | **A) Tatoeba** | Abundante, fácil, garantizado BLEU |
| Idioma | **Español-Inglés** | Máximas probabilidades de éxito |
| BLEU | **≥ 0.35** | Realista, suficiente rúbrica |

**Esto te asegura:**
- ✅ Rúbrica 17-20 en arquitectura RNN
- ✅ Rúbrica 17-20 en metodología CRISP-ML
- ✅ Rúbrica 17-20 en calidad de traducción
- ✅ Rúbrica 17-20 en despliegue y QA (API funcional)

---

## 📝 **RESUMEN: LO QUE FALTA (PRIORIDAD)**

### 🔴 CRÍTICO (sin esto, proyecto incompleto):
- [ ] **02_model_training.ipynb** - Entrenar modelo
- [ ] **03_evaluation.ipynb** - BLEU Score + análisis
- [ ] **api/app.py** - API funcional

### 🟡 IMPORTANTE (para máxima rúbrica):
- [ ] Descargar corpus Tatoeba
- [ ] Documentar resultados en .tex
- [ ] Tests de la API

### 🟢 OPCIONAL (nice-to-have):
- [ ] Demo web HTML/JS
- [ ] Dockerfile
- [ ] GitHub Actions CI/CD

---

## 🚀 **PRÓXIMO PASO:**

¿Cuál es tu decisión?

**Opción A:** Confirmar corpus, idioma, BLEU → Creo los notebooks 02 y 03
**Opción B:** Descargar Tatoeba primero → Te ayudo con descarga y preparación
**Opción C:** Empezar ya con corpus pequeño → Entrenamos modelo demo ahora
