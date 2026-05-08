import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_deepfake_model():
    # 1. Feature Extractor (CNN)
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False, 
        weights='imagenet', 
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    # 2. Input for Sequence (10 frames)
    inputs = layers.Input(shape=(10, 224, 224, 3)) 
    
    # Process each frame through CNN using TimeDistributed
    x = layers.TimeDistributed(base_model)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    
    # Project features to a fixed dimension (Transformer Embedding)
    projection_dim = 128
    x = layers.Dense(projection_dim)(x)
    
    # --- TRANSFORMER BLOCK ---
    # Multi-Head Self-Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=8, key_dim=projection_dim
    )(x, x)
    
    # Skip Connection 1
    x1 = layers.Add()([x, attention_output])
    x1 = layers.LayerNormalization()(x1)
    
    # Feed-Forward Network
    ffn = layers.Dense(projection_dim, activation="relu")(x1)
    ffn = layers.Dense(projection_dim)(ffn)
    
    # Skip Connection 2
    x2 = layers.Add()([x1, ffn])
    x = layers.LayerNormalization()(x2)
    # --------------------------
    
    # 3. Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize model
deepfake_detector = build_deepfake_model()