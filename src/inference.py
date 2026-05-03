"""
Módulo de inferencia para realizar predicciones con modelo entrenado.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List
import json


class TranslationInference:
    """Realiza inferencia (traducción) con modelo entrenado."""
    
    def __init__(self, model_path: str, source_vocab_path: str, 
                 target_vocab_path: str):
        """
        Args:
            model_path: Ruta al modelo guardado
            source_vocab_path: Ruta al vocabulario de origen
            target_vocab_path: Ruta al vocabulario de destino
        """
        # Cargar modelo con keras 3 / saved_model
        # Si termina en .keras le quitamos porque asumimos que es el exportado (SavedModel folder)
        if model_path.endswith('.keras'):
            model_path = model_path.replace('.keras', '')
            
        self.model = tf.saved_model.load(model_path).signatures['serving_default']
        
        # Cargar vocabularios
        with open(source_vocab_path, 'r') as f:
            self.source_vocab = json.load(f)
        with open(target_vocab_path, 'r') as f:
            self.target_vocab = json.load(f)
        
        # Crear mapeo inverso
        self.idx2source = {v: k for k, v in self.source_vocab.items()}
        self.idx2target = {v: k for k, v in self.target_vocab.items()}
        
        self.START_TOKEN_IDX = self.target_vocab.get('<START>', 2)
        self.END_TOKEN_IDX = self.target_vocab.get('<END>', 3)
        self.UNK_TOKEN_IDX = self.target_vocab.get('<UNK>', 1)
        self.PAD_TOKEN_IDX = self.target_vocab.get('<PAD>', 0)
    
    def preprocess_input(self, text: str, max_length: int = 30) -> np.ndarray:
        """Preprocesa texto de entrada."""
        text = text.lower().strip()
        tokens = text.split()
        
        indices = [self.START_TOKEN_IDX]
        for token in tokens:
            idx = self.source_vocab.get(token, self.source_vocab.get('<UNK>', 1))
            indices.append(idx)
        indices.append(self.source_vocab.get('<END>', 3))
        
        if len(indices) > max_length:
            indices = indices[:max_length]
        
        pad_length = max_length - len(indices)
        indices = indices + [self.PAD_TOKEN_IDX] * pad_length
        
        return np.array([indices], dtype=np.float32)
    
    def translate(self, text: str, max_length: int = 30, 
                 temperature: float = 1.0) -> Tuple[str, List[float]]:
        """Traduce texto."""
        source_seq = self.preprocess_input(text, max_length)
        
        try:
            decoder_input = np.zeros((1, 29), dtype=np.float32)
            decoder_input[0, 0] = self.START_TOKEN_IDX
            
            translated_tokens = []
            confidences = []
            for i in range(28):
                output = self.model(args_0=tf.constant(source_seq), args_0_1=tf.constant(decoder_input))
                predictions = list(output.values())[0].numpy()
                
                pred_probs = predictions[0, i, :]
                pred_idx = np.argmax(pred_probs)
                
                if pred_idx == self.END_TOKEN_IDX:
                    break
                    
                decoder_input[0, i + 1] = pred_idx
                
                if pred_idx != self.PAD_TOKEN_IDX:
                    token_str = self.idx2target.get(int(pred_idx), '<UNK>')
                    if token_str != '<UNK>':
                        translated_tokens.append(token_str)
            
            translation = ' '.join(translated_tokens) if translated_tokens else "No se pudo traducir"
            return translation, confidences
            
        except Exception as e:
            return f"Error en inferencia: {str(e)}", []
    
    def batch_translate(self, texts: List[str], max_length: int = 30) \
            -> List[Tuple[str, List[float]]]:
        """Traduce lote de textos."""
        results = []
        for text in texts:
            translation, conf = self.translate(text, max_length)
            results.append((translation, conf))
        return results
