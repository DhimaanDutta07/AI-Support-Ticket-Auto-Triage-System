import tensorflow as tf
from src.config.config import CONFIG
import keras


class MultiTaskModel:

    def build(self, num_classes):

        inputs = keras.Input(shape=(CONFIG["max_len"],), dtype=tf.int32)

        x = keras.layers.Embedding(CONFIG["max_vocab"], 128)(inputs)
        x = keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dropout(0.2)(x)

        out_category = keras.layers.Dense(num_classes["category"], activation="softmax", name="category")(x)
        out_priority = keras.layers.Dense(num_classes["priority"], activation="softmax", name="priority")(x)
        out_sentiment = keras.layers.Dense(num_classes["sentiment"], activation="softmax", name="sentiment")(x)

        model = keras.Model(inputs=inputs, outputs=[out_category, out_priority, out_sentiment])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=CONFIG["lr"]),
            loss={
                "category": "sparse_categorical_crossentropy",
                "priority": "sparse_categorical_crossentropy",
                "sentiment": "sparse_categorical_crossentropy"
            },
            metrics={
                "category": ["accuracy"],
                "priority": ["accuracy"],
                "sentiment": ["accuracy"]
            }
        )

        return model