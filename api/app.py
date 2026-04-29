"""
API REST para Traductor Automático Seq2Seq
Endpoint: POST /translate
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import json
from pathlib import Path
import numpy as np
from typing import Optional

# Modelos
class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "es"
    target_lang: str = "en"

class TranslationResponse(BaseModel):
    translation: str
    confidence: float
    processing_time_ms: float
    status: str

# Inicializar app
app = FastAPI(
    title="Traductor Automático RNN",
    description="API para traducción Español-Inglés con modelo Seq2Seq",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'traductor_v1'
VOCAB_ES_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'vocab_spanish.json'
VOCAB_EN_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'vocab_english.json'

# Cargar modelo y vocabularios
try:
    model = tf.saved_model.load(str(MODEL_PATH)).signatures['serving_default']
    with open(VOCAB_ES_PATH) as f:
        vocab_spanish = json.load(f)
    with open(VOCAB_EN_PATH) as f:
        vocab_english = json.load(f)
    MODEL_LOADED = True
except Exception as e:
    print(f"No se pudo cargar modelo: {e}")
    MODEL_LOADED = False

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "nombre": "Traductor Automático RNN",
        "version": "1.0",
        "estado": "online" if MODEL_LOADED else "modelo_no_cargado",
        "endpoint": "/translate"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "modelo_listo": MODEL_LOADED}

@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    """
    Endpoint para traducir texto
    
    Parámetros:
    - text: Texto a traducir (máximo 200 caracteres)
    - source_lang: Idioma origen (default: es)
    - target_lang: Idioma destino (default: en)
    """
    
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    # Validar entrada
    if len(request.text) > 200:
        raise HTTPException(status_code=400, detail="Texto muy largo (máximo 200 caracteres)")
    
    if len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Texto vacío")
    
    if request.source_lang != "es" or request.target_lang != "en":
        raise HTTPException(status_code=400, detail="Solo soportado es->en")
    
    try:
        # Preprocesar texto
        text_clean = request.text.lower().strip()
        tokens = text_clean.split()
        
        # Convertir a índices (añadiendo START y END)
        indices = [vocab_spanish.get('<START>', 2)]
        for token in tokens:
            idx = vocab_spanish.get(token, vocab_spanish.get('<UNK>', 1))
            indices.append(idx)
        indices.append(vocab_spanish.get('<END>', 3))
        
        # Padding/truncamiento
        max_len = 30
        if len(indices) < max_len:
            indices = indices + [vocab_spanish.get('<PAD>', 0)] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        
        input_seq = np.array([indices], dtype=np.float32)
        
        # Inferencia autoregresiva
        decoder_input = np.zeros((1, 29), dtype=np.float32)
        START_TOKEN = vocab_english.get('<START>', 2)
        END_TOKEN = vocab_english.get('<END>', 3)
        PAD_TOKEN = vocab_english.get('<PAD>', 0)
        
        decoder_input[0, 0] = START_TOKEN
        idx2english = {v: k for k, v in vocab_english.items()}
        
        translated_tokens = []
        for i in range(28):
            output = model(args_0=tf.constant(input_seq), args_0_1=tf.constant(decoder_input))
            predictions = list(output.values())[0].numpy()
            
            # Obtener la predicción para el paso i 
            pred_idx = np.argmax(predictions[0, i, :])
            
            if pred_idx == END_TOKEN:
                break
                
            decoder_input[0, i + 1] = pred_idx
            
            if pred_idx != PAD_TOKEN:
                token_str = idx2english.get(int(pred_idx), '<UNK>')
                if token_str != '<UNK>':
                    translated_tokens.append(token_str)
        
        translation = ' '.join(translated_tokens) if translated_tokens else "No se pudo traducir"
        
        return TranslationResponse(
            translation=translation,
            confidence=0.5,  # Placeholder
            processing_time_ms=50,  # Placeholder
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en traducción: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
