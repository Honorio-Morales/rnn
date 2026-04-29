"""
Módulo de métricas para evaluación de traducción automática.
Implementa BLEU Score y otras métricas de calidad.
"""

import numpy as np
from collections import Counter
from typing import List, Tuple, Dict
import math


class BLEUScore:
    """Calcula BLEU Score para traducción automática."""
    
    def __init__(self, n_grams: int = 4):
        """
        Args:
            n_grams: Número máximo de n-gramas a considerar (default: 4)
        """
        self.n_grams = n_grams
    
    @staticmethod
    def _get_ngrams(sentence: List[str], n: int) -> Counter:
        """Extrae n-gramas de una oración.
        
        Args:
            sentence: Lista de tokens
            n: Tamaño del n-grama
            
        Returns:
            Counter con frecuencias de n-gramas
        """
        ngrams = []
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i:i+n])
            ngrams.append(ngram)
        return Counter(ngrams)
    
    def _brevity_penalty(self, generated_length: int, reference_length: int) -> float:
        """Calcula penalización por brevedad.
        
        Args:
            generated_length: Longitud de traducción generada
            reference_length: Longitud de referencia
            
        Returns:
            Brevity penalty (0-1)
        """
        if generated_length >= reference_length:
            return 1.0
        else:
            ratio = generated_length / reference_length
            return math.exp(1 - 1/ratio) if ratio > 0 else 0.0
    
    def _ngram_precision(self, generated: List[str], reference: List[str], n: int) -> float:
        """Calcula precisión de n-gramas.
        
        Args:
            generated: Oración generada (tokens)
            reference: Oración de referencia (tokens)
            n: Tamaño del n-grama
            
        Returns:
            Precisión de n-gramas (0-1)
        """
        generated_ngrams = self._get_ngrams(generated, n)
        reference_ngrams = self._get_ngrams(reference, n)
        
        if not generated_ngrams:
            return 0.0
        
        # Contar matches (máximo: frecuencia en referencia)
        matches = 0
        for ngram, count in generated_ngrams.items():
            matches += min(count, reference_ngrams.get(ngram, 0))
        
        return matches / sum(generated_ngrams.values())
    
    def compute(self, generated: List[str], reference: List[str], 
               weights: List[float] = None) -> float:
        """Calcula BLEU score.
        
        Args:
            generated: Oración generada (tokens)
            reference: Oración de referencia (tokens)
            weights: Pesos para cada n-grama (default: iguales)
            
        Returns:
            BLEU score (0-1)
        """
        if weights is None:
            weights = [1.0 / self.n_grams] * self.n_grams
        
        # Calcular brevity penalty
        bp = self._brevity_penalty(len(generated), len(reference))
        
        # Calcular precision de n-gramas
        precisions = []
        for n in range(1, self.n_grams + 1):
            p_n = self._ngram_precision(generated, reference, n)
            precisions.append(p_n)
        
        # Calcular BLEU
        if any(p == 0 for p in precisions):
            # Si alguna precisión es 0, BLEU = 0 (penalización por falta de n-gramas)
            bleu = 0.0
        else:
            log_precisions = [math.log(p) for p in precisions]
            geo_mean = sum(w * lp for w, lp in zip(weights, log_precisions))
            bleu = bp * math.exp(geo_mean)
        
        return bleu
    
    def compute_corpus_bleu(self, generated_corpus: List[List[str]], 
                           reference_corpus: List[List[str]]) -> float:
        """Calcula BLEU a nivel de corpus.
        
        Args:
            generated_corpus: Lista de oraciones generadas
            reference_corpus: Lista de oraciones de referencia
            
        Returns:
            BLEU score del corpus
        """
        # Acumular estadísticas
        total_matches = {n: 0 for n in range(1, self.n_grams + 1)}
        total_count = {n: 0 for n in range(1, self.n_grams + 1)}
        generated_length = 0
        reference_length = 0
        
        for generated, reference in zip(generated_corpus, reference_corpus):
            generated_length += len(generated)
            reference_length += len(reference)
            
            for n in range(1, self.n_grams + 1):
                gen_ngrams = self._get_ngrams(generated, n)
                ref_ngrams = self._get_ngrams(reference, n)
                
                for ngram, count in gen_ngrams.items():
                    total_matches[n] += min(count, ref_ngrams.get(ngram, 0))
                    total_count[n] += count
        
        # Calcular brevity penalty
        bp = self._brevity_penalty(generated_length, reference_length)
        
        # Calcular precisiones y BLEU
        precisions = []
        for n in range(1, self.n_grams + 1):
            if total_count[n] > 0:
                p_n = total_matches[n] / total_count[n]
                precisions.append(p_n)
            else:
                precisions.append(0.0)
        
        if any(p == 0 for p in precisions):
            bleu = 0.0
        else:
            weights = [1.0 / self.n_grams] * self.n_grams
            log_precisions = [math.log(p) for p in precisions]
            geo_mean = sum(w * lp for w, lp in zip(weights, log_precisions))
            bleu = bp * math.exp(geo_mean)
        
        return bleu


class ErrorAnalyzer:
    """Analiza errores en traducción automática."""
    
    def __init__(self, unk_token: str = "<UNK>", pad_token: str = "<PAD>"):
        """
        Args:
            unk_token: Token para palabras desconocidas
            pad_token: Token para padding
        """
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.errors = {
            'oov': 0,      # Out of Vocabulary
            'context': 0,  # Contexto perdido
            'order': 0,    # Orden incorrecto
            'grammatical': 0  # Error gramatical
        }
    
    def detect_oov(self, generated: List[str]) -> int:
        """Detecta tokens desconocidos.
        
        Args:
            generated: Oración generada
            
        Returns:
            Número de tokens OOV
        """
        return sum(1 for token in generated if token == self.unk_token)
    
    def compare_length(self, generated: List[str], reference: List[str]) \
            -> Tuple[int, str]:
        """Compara longitud de oraciones.
        
        Args:
            generated: Oración generada
            reference: Oración de referencia
            
        Returns:
            (diferencia, tipo)
        """
        diff = len(generated) - len(reference)
        if diff > 2:
            error_type = "too_long"
        elif diff < -2:
            error_type = "too_short"
        else:
            error_type = "appropriate"
        return diff, error_type
    
    def word_overlap(self, generated: List[str], reference: List[str]) -> float:
        """Calcula overlap de palabras (recall).
        
        Args:
            generated: Oración generada
            reference: Oración de referencia
            
        Returns:
            Porcentaje de palabras que coinciden
        """
        if not reference:
            return 0.0
        
        gen_set = set(generated) - {self.pad_token, self.unk_token}
        ref_set = set(reference) - {self.pad_token, self.unk_token}
        
        if not ref_set:
            return 0.0
        
        overlap = len(gen_set & ref_set)
        return overlap / len(ref_set)
    
    def analyze_sample(self, generated: List[str], reference: List[str]) \
            -> Dict[str, any]:
        """Analiza un par generado-referencia.
        
        Args:
            generated: Oración generada
            reference: Oración de referencia
            
        Returns:
            Diccionario con análisis
        """
        analysis = {
            'oov_count': self.detect_oov(generated),
            'length_diff': self.compare_length(generated, reference)[0],
            'length_type': self.compare_length(generated, reference)[1],
            'word_overlap': self.word_overlap(generated, reference),
            'bleu': BLEUScore().compute(generated, reference)
        }
        return analysis
    
    def analyze_batch(self, generated_batch: List[List[str]], 
                     reference_batch: List[List[str]]) -> Dict[str, any]:
        """Analiza un lote de pares.
        
        Args:
            generated_batch: Lista de oraciones generadas
            reference_batch: Lista de oraciones de referencia
            
        Returns:
            Diccionario con estadísticas agregadas
        """
        analyses = []
        for gen, ref in zip(generated_batch, reference_batch):
            analyses.append(self.analyze_sample(gen, ref))
        
        # Agregar estadísticas
        stats = {
            'avg_oov': np.mean([a['oov_count'] for a in analyses]),
            'avg_length_diff': np.mean([abs(a['length_diff']) for a in analyses]),
            'avg_word_overlap': np.mean([a['word_overlap'] for a in analyses]),
            'avg_bleu': np.mean([a['bleu'] for a in analyses]),
            'short_samples': sum(1 for a in analyses if a['length_type'] == 'too_short'),
            'long_samples': sum(1 for a in analyses if a['length_type'] == 'too_long'),
        }
        return stats


def exact_match_accuracy(generated_corpus: List[List[str]], 
                        reference_corpus: List[List[str]]) -> float:
    """Calcula exactitud de match exacto.
    
    Args:
        generated_corpus: Lista de oraciones generadas
        reference_corpus: Lista de oraciones de referencia
        
    Returns:
        Porcentaje de exactos (0-1)
    """
    matches = sum(1 for g, r in zip(generated_corpus, reference_corpus) 
                 if g == r)
    return matches / len(generated_corpus) if generated_corpus else 0.0
