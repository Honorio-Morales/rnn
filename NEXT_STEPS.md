# 🚀 INSTRUCCIONES PARA GIT Y ENTRENAMIENTO

## PASO 1: Subir a GitHub

```bash
cd /home/honorio/IA/rnn

# Inicializar repo git
git init

# Agregar archivos
git add .

# Primer commit
git commit -m "Initial commit: Estructura CRISP-ML + código base"

# Conectar a GitHub (reemplaza con tu repo)
git remote add origin https://github.com/Honorio-Morales/rnn.git

# Push a main branch
git branch -M main
git push -u origin main
```

## PASO 2: Ejecutar Entrenamiento

### 2a. Verificar datos preparados
```bash
# Ejecutar 01_exploration.ipynb para generar datos
jupyter notebook notebooks/01_exploration.ipynb
# Esto genera:
# - data/processed/vocab_spanish.json
# - data/processed/vocab_english.json
# - data/splits/{X_train, y_train, X_val, y_val, X_test, y_test}.npy
```

### 2b. Entrenar modelo
```bash
# Ejecutar notebook de entrenamiento
jupyter notebook notebooks/02_model_training.ipynb
# SALIDA:
# - models/traductor_v1/ (modelo guardado)
# - models/training_metrics.json (métricas de entrenamiento)
# - models/training_history.png (gráficas Loss/Accuracy)
```

### 2c. Evaluar modelo
```bash
# Ejecutar notebook de evaluación
jupyter notebook notebooks/03_evaluation.ipynb
# SALIDA:
# - models/evaluation_metrics.json (BLEU Score, Exact Match)
# - Print de ejemplos de traducción
```

## PASO 3: Actualizar Documento .tex

Abrir `docs/proyecto_crisp_ml.tex` y completar:

### Sección 4: Evaluation
- Agregar tabla con BLEU Score obtenido
- Incluir captura de training_history.png
- Copiar ejemplos de traducción del notebook 03

### Sección 3: Modeling  
- Agregar captura de model.summary()
- Número de parámetros totales

### Sección 6: Monitoring & Maintenance
- Agregar JSON de métricas
- Plan de feedback

## PASO 4: Guardar cambios en Git

```bash
# Ver cambios
git status

# Agregar archivos (excepto modelos - ignorados en .gitignore)
git add docs/ notebooks/

# Commit con métricas
git commit -m "feat: Entrenamiento completado + métricas BLEU"

# Push
git push origin main
```

## ✅ CHECKLIST FINAL

- [ ] Git inicializado y subido a GitHub
- [ ] Notebook 01 ejecutado (datos preparados)
- [ ] Notebook 02 ejecutado (modelo entrenado)
- [ ] Notebook 03 ejecutado (evaluación BLEU)
- [ ] Documento .tex actualizado con métricas
- [ ] Cambios subidos a GitHub
- [ ] README.md describe el proyecto

---

## 📊 MÉTRICAS A CAPTURAR

Después de ejecutar 02 y 03, tendrás:

**En models/training_metrics.json:**
- final_train_loss
- final_val_loss
- final_train_acc
- final_val_acc
- epochs

**En models/evaluation_metrics.json:**
- bleu_score ← **CRÍTICO**
- exact_match

**Gráficas:**
- models/training_history.png ← Agregar al .tex

---

## 🔗 GITHUB WORKFLOW

```
LOCAL                              GITHUB
├─ Código base ─────────────────→ Repo inicial
├─ Datos preparados
├─ Modelo entrenado (no en git)
├─ Notebooks ejecutados ────────→ Push con métricas
└─ Documento actualizado ───────→ Final commit
```

El modelo (.h5) **NO se sube a GitHub** (demasiado grande).
Solo los notebooks, documento y código.
