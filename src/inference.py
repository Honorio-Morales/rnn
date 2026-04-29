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
        self.model = tf.keras.models.load_model(model_path)
        
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
        """Preprocesa texto de entrada.
        
        Args:
            text: Texto a traducir
            max_length: Longitud máxima
            
        Returns:
            Array de índices con padding
        """
        # Limpieza básica
        text = text.lower().strip()
        tokens = text.split()
        
        # Convertir a índices
        indices = [self.START_TOKEN_IDX]
        for token in tokens:
            idx = self.source_vocab.get(token, self.source_vocab.get('<UNK>', 1))
            indices.append(idx)
        indices.append(self.source_vocab.get('<END>', 3))
        
        # Truncar si es muy largo
        if len(indices) > max_length:
            indices = indices[:max_length]
        
        # Padding
        pad_length = max_length - len(indices)
        indices = indices + [self.PAD_TOKEN_IDX] * pad_length
        
        return np.array([indices])
    
    def translate(self, text: str, max_length: int = 30, 
                 temperature: float = 1.0) -> Tuple[str, List[float]]:
        """Traduce texto.
        
        Args:
            text: Texto a traducir
            max_length: Longitud máxima de traducción
            temperature: Temperatura para sampling
            
        Returns:
            (texto_traducido, confianza)
        """
        # Preprocesar
        source_seq = self.preprocess_input(text, max_length)
        
        # Inferencia
        try:
            translation_indices, attention_weights = self.model.generate_translation(
                source_seq, max_length=max_length, temperature=temperature
            )
        except Exception as e:
            return f"Error en inferencia: {str(e)}", []
        
        # Decodificar
        translated_tokens = []
        confidences = []
        
        for idx in translation_indices:
            if idx == self.END_TOKEN_IDX:
                break
            if idx == self.PAD_TOKEN_IDX:
                continue
            
            token = self.idx2target.get(int(idx), '<UNK>')
            if token != '<UNK>':
                translated_tokens.append(token)
        
        translation = ' '.join(translated_tokens)
        return translation, confidences
    
    def batch_translate(self, texts: List[str], max_length: int = 30) \
            -> List[Tuple[str, List[float]]]:
        """Traduce lote de textos.
        
        Args:
            texts: Lista de textos
            max_length: Longitud máxima
            
        Returns:
            Lista de (traducción, confianza)
        """
        results = []
        for text in texts:
            translation, conf = self.translate(text, max_length)
            results.append((translation, conf))
        return results
