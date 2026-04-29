"""
Módulo de preprocesamiento para traducción automática.
Limpieza de texto, tokenización y manejo de vocabulario.
"""

import re
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict


class TextPreprocessor:
    """Preprocesa texto para entrenamiento de RNN."""
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 30):
        """
        Args:
            vocab_size: Número máximo de palabras únicas a mantener
            max_length: Longitud máxima de secuencias
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size_actual = 0
        
        # Tokens especiales
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.START_TOKEN = "<START>"
        self.END_TOKEN = "<END>"
        
    def _clean_text(self, text: str) -> str:
        """Limpia texto: minúsculas, puntuación, espacios."""
        # Convertir a minúsculas
        text = text.lower().strip()
        
        # Remover URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Mantener puntuación básica
        text = re.sub(r'([.!?,;])', r' \1 ', text)
        
        # Remover caracteres especiales
        text = re.sub(r'[^a-záéíóúñ\s.,!?;]', '', text)
        
        # Remover espacios duplicados
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Divide texto en tokens (palabras)."""
        return text.split()
    
    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """Construye vocabulario a partir de textos.
        
        Args:
            texts: Lista de textos para construir vocabulario
            min_freq: Frecuencia mínima para incluir palabra
        """
        # Tokenizar todos los textos
        all_tokens = []
        for text in texts:
            cleaned = self._clean_text(text)
            tokens = self._tokenize(cleaned)
            all_tokens.extend(tokens)
        
        # Contar frecuencias
        counter = Counter(all_tokens)
        
        # Inicializar con tokens especiales
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.START_TOKEN: 2,
            self.END_TOKEN: 3
        }
        
        # Agregar palabras más frecuentes
        idx = 4
        for word, freq in counter.most_common(self.vocab_size - 4):
            if freq >= min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        # Crear mapeo inverso
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size_actual = len(self.word2idx)
        
        print(f"Vocabulario construido: {self.vocab_size_actual} palabras")
    
    def encode(self, text: str, add_tokens: bool = True) -> List[int]:
        """Convierte texto a secuencia de índices.
        
        Args:
            text: Texto a encodificar
            add_tokens: Si True, agrega START y END tokens
            
        Returns:
            Lista de índices
        """
        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)
        
        # Convertir tokens a índices
        indices = []
        if add_tokens:
            indices.append(self.word2idx[self.START_TOKEN])
        
        for token in tokens:
            idx = self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
            indices.append(idx)
        
        if add_tokens:
            indices.append(self.word2idx[self.END_TOKEN])
        
        # Truncar si es muy largo
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        
        return indices
    
    def pad_sequence(self, sequence: List[int], length: int = None) -> List[int]:
        """Aplica padding a una secuencia.
        
        Args:
            sequence: Secuencia de índices
            length: Longitud objetivo (default: max_length)
            
        Returns:
            Secuencia con padding
        """
        if length is None:
            length = self.max_length
        
        pad_idx = self.word2idx[self.PAD_TOKEN]
        
        if len(sequence) < length:
            sequence = sequence + [pad_idx] * (length - len(sequence))
        else:
            sequence = sequence[:length]
        
        return sequence
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Convierte secuencia de índices a texto.
        
        Args:
            indices: Lista de índices
            remove_special: Si True, remueve START y END tokens
            
        Returns:
            Texto decodificado
        """
        tokens = []
        special_tokens = {self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN}
        
        for idx in indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                
                if remove_special and token in special_tokens:
                    continue
                
                if token == self.UNK_TOKEN:
                    continue
                
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def save_vocab(self, filepath: str) -> None:
        """Guarda vocabulario a archivo."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, filepath: str) -> None:
        """Carga vocabulario desde archivo."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size_actual = len(self.word2idx)


class ParallelDataProcessor:
    """Procesa pares paralelos de textos (origen-destino)."""
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 30):
        """
        Args:
            vocab_size: Tamaño de vocabulario
            max_length: Longitud máxima de secuencias
        """
        self.source_processor = TextPreprocessor(vocab_size, max_length)
        self.target_processor = TextPreprocessor(vocab_size, max_length)
        self.max_length = max_length
    
    def build_vocabs(self, source_texts: List[str], target_texts: List[str]) -> None:
        """Construye vocabularios para ambos idiomas."""
        print("Construyendo vocabulario de origen...")
        self.source_processor.build_vocab(source_texts)
        
        print("Construyendo vocabulario de destino...")
        self.target_processor.build_vocab(target_texts)
    
    def encode_pair(self, source_text: str, target_text: str) \
            -> Tuple[List[int], List[int]]:
        """Encoda par de textos.
        
        Returns:
            Tupla (source_indices, target_indices)
        """
        source_idx = self.source_processor.encode(source_text)
        target_idx = self.target_processor.encode(target_text)
        
        # Aplicar padding
        source_idx = self.source_processor.pad_sequence(source_idx)
        target_idx = self.target_processor.pad_sequence(target_idx)
        
        return source_idx, target_idx
    
    def create_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 32) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Crea batch de datos para entrenamiento.
        
        Args:
            pairs: Lista de tuplas (source, target)
            batch_size: Tamaño del batch
            
        Returns:
            (X_batch, y_batch) como arrays de numpy
        """
        X_batch = []
        y_batch = []
        
        for source_text, target_text in pairs:
            source_idx, target_idx = self.encode_pair(source_text, target_text)
            X_batch.append(source_idx)
            y_batch.append(target_idx)
        
        return np.array(X_batch), np.array(y_batch)
    
    def save_vocabs(self, source_path: str, target_path: str) -> None:
        """Guarda ambos vocabularios."""
        self.source_processor.save_vocab(source_path)
        self.target_processor.save_vocab(target_path)
    
    def load_vocabs(self, source_path: str, target_path: str) -> None:
        """Carga ambos vocabularios."""
        self.source_processor.load_vocab(source_path)
        self.target_processor.load_vocab(target_path)


def load_parallel_corpus(filepath: str, limit: int = None) -> List[Tuple[str, str]]:
    """Carga corpus paralelo desde archivo TSV.
    
    Args:
        filepath: Ruta al archivo (formato: source<TAB>target)
        limit: Número máximo de pares a cargar
        
    Returns:
        Lista de tuplas (source, target)
    """
    pairs = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
    except FileNotFoundError:
        print(f"Archivo no encontrado: {filepath}")
    
    return pairs
