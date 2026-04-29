"""
Arquitectura Seq2Seq (Encoder-Decoder) para traducción automática.
Implementado con TensorFlow/Keras, optimizado para GPU 6GB.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


class Encoder(layers.Layer):
    """Encoder bidireccional LSTM para Seq2Seq."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_units: int = 256, num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            vocab_size: Tamaño del vocabulario
            embedding_dim: Dimensión de embeddings (default: 128)
            hidden_units: Unidades LSTM (default: 256)
            num_layers: Número de capas LSTM (default: 2)
            dropout: Tasa de dropout (default: 0.3)
        """
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        
        self.lstm_layers = []
        for i in range(num_layers):
            self.lstm_layers.append(
                layers.Bidirectional(
                    layers.LSTM(hidden_units, return_sequences=(i < num_layers - 1), 
                               return_state=True, dropout=dropout)
                )
            )
    
    def call(self, x, training=False):
        """Procesa secuencia de entrada.
        
        Args:
            x: Tensor de entrada (batch_size, seq_length)
            training: Flag para dropout
            
        Returns:
            outputs, state_h, state_c: Salida y estados finales
        """
        x = self.embedding(x)
        
        for lstm in self.lstm_layers:
            if isinstance(lstm, layers.Bidirectional):
                x, forward_h, forward_c, backward_h, backward_c = lstm(x, training=training)
            else:
                x, state_h, state_c = lstm(x, training=training)
        
        # Concatenar estados forward y backward
        state_h = tf.concat([forward_h, backward_h], axis=-1)
        state_c = tf.concat([forward_c, backward_c], axis=-1)
        
        return x, state_h, state_c


class BahdanauAttention(layers.Layer):
    """Mecanismo de atención de Bahdanau."""
    
    def __init__(self, hidden_units: int):
        """
        Args:
            hidden_units: Dimensión de vectores internos
        """
        super().__init__()
        self.W1 = layers.Dense(hidden_units)
        self.W2 = layers.Dense(hidden_units)
        self.V = layers.Dense(1)
    
    def call(self, decoder_hidden, encoder_outputs):
        """Calcula pesos de atención.
        
        Args:
            decoder_hidden: Estado actual del decoder (batch_size, hidden_units)
            encoder_outputs: Salidas del encoder (batch_size, seq_length, 2*hidden_units)
            
        Returns:
            context: Vector de contexto (batch_size, 2*hidden_units)
            weights: Pesos de atención (batch_size, seq_length)
        """
        # Expandir dimensiones para broadcasting
        decoder_hidden = tf.expand_dims(decoder_hidden, 1)
        
        # Calcular energía
        energy = tf.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden))
        energy = self.V(energy)
        
        # Calcular pesos (atención)
        weights = tf.nn.softmax(energy, axis=1)
        
        # Calcular contexto como promedio ponderado
        context = weights * encoder_outputs
        context = tf.reduce_sum(context, axis=1)
        
        return context, weights


class Decoder(layers.Layer):
    """Decoder LSTM con atención para Seq2Seq."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_units: int = 256, num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            vocab_size: Tamaño del vocabulario destino
            embedding_dim: Dimensión de embeddings
            hidden_units: Unidades LSTM
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout
        """
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.attention = BahdanauAttention(hidden_units)
        
        self.lstm_layers = []
        for i in range(num_layers):
            self.lstm_layers.append(
                layers.LSTM(hidden_units, return_sequences=True, 
                           return_state=True, dropout=dropout)
            )
        
        self.dense = layers.Dense(vocab_size, activation='softmax')
    
    def call(self, x, encoder_outputs, decoder_state_h, decoder_state_c, training=False):
        """Decodifica una secuencia.
        
        Args:
            x: Token de entrada (batch_size, 1)
            encoder_outputs: Salidas del encoder
            decoder_state_h: Estado h del decoder
            decoder_state_c: Estado c del decoder
            training: Flag para dropout
            
        Returns:
            output: Logits de salida (batch_size, vocab_size)
            state_h, state_c: Nuevos estados
            attention_weights: Pesos de atención
        """
        x = self.embedding(x)
        
        # Atención
        context, attention_weights = self.attention(decoder_state_h, encoder_outputs)
        
        # Concatenar entrada con contexto
        x = tf.concat([tf.squeeze(x, 1), context], axis=-1)
        x = tf.expand_dims(x, 1)
        
        # Pasar por capas LSTM
        for lstm in self.lstm_layers:
            x, decoder_state_h, decoder_state_c = lstm(x, initial_state=[decoder_state_h, decoder_state_c], training=training)
        
        # Producir distribución de probabilidad
        output = self.dense(x)
        
        return output, decoder_state_h, decoder_state_c, attention_weights


class Seq2SeqModel(Model):
    """Modelo completo Seq2Seq para traducción automática."""
    
    def __init__(self, source_vocab_size: int, target_vocab_size: int,
                 embedding_dim: int = 128, hidden_units: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            source_vocab_size: Tamaño vocabulario origen
            target_vocab_size: Tamaño vocabulario destino
            embedding_dim: Dimensión de embeddings (optimizado para 6GB)
            hidden_units: Unidades LSTM (optimizado para 6GB)
            num_layers: Número de capas (optimizado para 6GB)
            dropout: Tasa de dropout
        """
        super().__init__()
        self.encoder = Encoder(source_vocab_size, embedding_dim, 
                              hidden_units, num_layers, dropout)
        self.decoder = Decoder(target_vocab_size, embedding_dim,
                              hidden_units, num_layers, dropout)
        self.enc_to_dec_h = layers.Dense(hidden_units, activation='tanh')
        self.enc_to_dec_c = layers.Dense(hidden_units, activation='tanh')
        self.target_vocab_size = target_vocab_size
        self.START_TOKEN_IDX = 2  # Token especial <START>
        self.END_TOKEN_IDX = 3    # Token especial <END>
    
    def call(self, inputs, training=False):
        """Forward pass.
        
        Args:
            inputs: Tupla (encoder_input, decoder_input)
            training: Flag para dropout
            
        Returns:
            predictions: Logits de salida
        """
        encoder_input, decoder_input = inputs
        
        # Encode
        encoder_outputs, state_h, state_c = self.encoder(encoder_input, training=training)

        state_h = self.enc_to_dec_h(state_h)
        state_c = self.enc_to_dec_c(state_c)
        
        # Decode
        all_outputs = []
        decoder_state_h = state_h
        decoder_state_c = state_c
        
        # Procesar cada token de entrada del decoder
        for t in range(decoder_input.shape[1]):
            decoder_single_input = decoder_input[:, t:t+1]
            
            output, decoder_state_h, decoder_state_c, _ = self.decoder(
                decoder_single_input, encoder_outputs, decoder_state_h, 
                decoder_state_c, training=training
            )
            all_outputs.append(output)
        
        # Concatenar todas las salidas
        predictions = tf.concat(all_outputs, axis=1)
        return predictions
    
    def generate_translation(self, source_sequence: np.ndarray, 
                            max_length: int = 30, temperature: float = 1.0):
        """Genera traducción (inference mode).
        
        Args:
            source_sequence: Secuencia codificada (1, seq_length)
            max_length: Longitud máxima de traducción
            temperature: Temperatura para sampling (1.0 = greedy)
            
        Returns:
            translation_indices: Índices de la traducción
            attention_weights: Pesos de atención por paso
        """
        # Encode
        encoder_outputs, state_h, state_c = self.encoder(source_sequence, training=False)

        state_h = self.enc_to_dec_h(state_h)
        state_c = self.enc_to_dec_c(state_c)
        
        # Inicializar decoder con START token
        decoder_input = np.array([[self.START_TOKEN_IDX]])
        translation_indices = []
        attention_weights_all = []
        
        decoder_state_h = state_h
        decoder_state_c = state_c
        
        for _ in range(max_length):
            # Predict siguiente token
            output, decoder_state_h, decoder_state_c, attention_weights = self.decoder(
                decoder_input, encoder_outputs, decoder_state_h, decoder_state_c, 
                training=False
            )
            
            # Muestrear o seleccionar token más probable
            logits = output[0, 0, :] / temperature
            probs = tf.nn.softmax(logits)
            
            if temperature == 1.0:
                # Greedy: token más probable
                token_idx = tf.argmax(probs).numpy()
            else:
                # Sampling según distribución
                token_idx = np.random.choice(len(probs), p=probs.numpy())
            
            translation_indices.append(token_idx)
            attention_weights_all.append(attention_weights.numpy()[0, 0, :])
            
            # Si generamos END token, parar
            if token_idx == self.END_TOKEN_IDX:
                break
            
            # Usar token generado como próxima entrada
            decoder_input = np.array([[token_idx]])
        
        return np.array(translation_indices), np.array(attention_weights_all)


def create_seq2seq_model(source_vocab_size: int, target_vocab_size: int,
                        max_length: int = 30) -> Seq2SeqModel:
    """Factory function para crear modelo optimizado.
    
    Parámetros optimizados para GPU 6GB:
    - embedding_dim: 128
    - hidden_units: 256
    - num_layers: 2
    - batch_size: 32
    """
    model = Seq2SeqModel(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        embedding_dim=128,
        hidden_units=256,
        num_layers=2,
        dropout=0.0
    )
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
        run_eagerly=True
    )
    
    return model
