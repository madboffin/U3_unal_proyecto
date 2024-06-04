from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np
import pandas as pd


def show_example(tokenized_data_train):
    input_ids_sample = tokenized_data_train["input_ids"][0]
    decoded_string = tokenizer.decode(input_ids_sample)
    print(decoded_string)


def explore_vocab_size(tokenizer):
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")


def load_base_model(learning_rate: float):
    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss)
    model.optimizer.learning_rate.assign(learning_rate)
    return model


def warming_up_model(model, tokenized_data_train, labels_train):
    for i in range(len(model.layers) - 2):
        model.layers[i].trainable = False

    stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        restore_best_weights=True,
    )

    history = model.fit(
        dict(tokenized_data_train),
        labels_train,
        epochs=6,
        batch_size=16,
        validation_split=0.2,
        callbacks=[stopping],
    )
    return model


def finetune_model(model, tokenized_data_train, labels_train):
    for i in range(len(model.layers) - 2):
        model.layers[i].trainable = True

    stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        restore_best_weights=True,
    )

    history = model.fit(
        dict(tokenized_data_train),
        labels_train,
        epochs=6,
        batch_size=16,
        validation_split=0.2,
        callbacks=[stopping],
    )
    return model


path = "src/txt_comments/jigsaw_data_prep_sampled.parquet"
comments = pd.read_parquet(path)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
max_length = 32
tokenized_data_train = tokenizer(
    comments["comment_text_prep"].to_list(),
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="tf",
)
labels_train = tf.keras.utils.to_categorical(np.array(comments["toxicity_class"]))

show_example(tokenized_data_train)

explore_vocab_size(tokenizer)

new_learning_rate = 5e-6
model = load_base_model(new_learning_rate)

model = warming_up_model(model, tokenized_data_train, labels_train)

model = finetune_model(model, tokenized_data_train, labels_train)

model.save("src/models/bert_model.keras")
