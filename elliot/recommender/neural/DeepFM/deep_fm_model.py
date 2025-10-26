"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Antonio Ferrara'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it,' \
            'daniele.malitesta@poliba.it, antonio.ferrara@poliba.it'

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepFMModel(keras.Model):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_mf_size,
                 hidden_layers,
                 lambda_weights,
                 learning_rate=0.01,
                 random_seed=42,
                 name="DeepFM",
                 num_features=0,
                 item_features=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)
        self.num_users = num_users
        self.num_items = num_items
        self.embed_mf_size = embed_mf_size
        self.hidden_layers = hidden_layers  # Specify as ((5, 'sigmoid'), (10, 'relu'))
        self.lambda_weights = lambda_weights
        self.num_features = num_features
        self.num_fields = 2 + (1 if self.num_features > 0 else 0)

        self.initializer = tf.initializers.GlorotUniform()

        self.user_mf_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=self.embed_mf_size,
                                                        embeddings_initializer=self.initializer, name='U_MF',
                                                        embeddings_regularizer=keras.regularizers.l2(
                                                            self.lambda_weights),
                                                        dtype=tf.float32)
        self.item_mf_embedding = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embed_mf_size,
                                                        embeddings_regularizer=keras.regularizers.l2(
                                                            self.lambda_weights),
                                                        embeddings_initializer=self.initializer, name='I_MF',
                                                        dtype=tf.float32)

        self.user_linear = keras.layers.Embedding(self.num_users, 1,
                                                  embeddings_initializer=self.initializer,
                                                  embeddings_regularizer=keras.regularizers.l2(self.lambda_weights),
                                                  name="U_linear",
                                                  dtype=tf.float32)
        self.item_linear = keras.layers.Embedding(self.num_items, 1,
                                                  embeddings_initializer=self.initializer,
                                                  embeddings_regularizer=keras.regularizers.l2(self.lambda_weights),
                                                  name="I_linear",
                                                  dtype=tf.float32)
        if self.num_features:
            self.feature_embedding = keras.layers.Embedding(self.num_features + 1, self.embed_mf_size,
                                                            embeddings_initializer=self.initializer,
                                                            embeddings_regularizer=keras.regularizers.l2(
                                                                self.lambda_weights),
                                                            mask_zero=True,
                                                            name="F_MF",
                                                            dtype=tf.float32)
            self.feature_linear = keras.layers.Embedding(self.num_features + 1, 1,
                                                         embeddings_initializer=self.initializer,
                                                         embeddings_regularizer=keras.regularizers.l2(
                                                             self.lambda_weights),
                                                         mask_zero=True,
                                                         name="F_linear",
                                                         dtype=tf.float32)
            if item_features is None:
                raise ValueError("item_features must be provided when num_features > 0")
            self.item_feature_ids = tf.constant(item_features, dtype=tf.int32)
        else:
            self.feature_embedding = None
            self.feature_linear = None
            self.item_feature_ids = None

        self.bias_ = tf.Variable(0., name='GB')

        self.user_mf_embedding(0)
        self.item_mf_embedding(0)
        self.user_linear(0)
        self.item_linear(0)
        if self.feature_embedding is not None:
            self.feature_embedding(0)
            self.feature_linear(0)

        deep_layers = [
            tf.keras.layers.Dense(self.hidden_layers[0][0],
                                  activation=self.hidden_layers[0][1],
                                  input_dim=self.num_fields * self.embed_mf_size)
        ]
        deep_layers.extend(
            [tf.keras.layers.Dense(n, activation=act) for n, act in self.hidden_layers[1:]]
        )
        self.hidden = tf.keras.Sequential(deep_layers)

        self.prediction_layer = tf.keras.layers.Dense(1, activation=None)

        self.loss = keras.losses.BinaryCrossentropy(from_logits=True)

        self.optimizer = tf.optimizers.Adam(learning_rate)

    def _gather_item_features(self, item_ids):
        if self.item_feature_ids is None:
            return None, None
        features = tf.gather(self.item_feature_ids, item_ids)
        mask = tf.cast(features > 0, tf.float32)
        return features, mask

    def _aggregate_feature_embeddings(self, feature_ids, mask):
        if feature_ids is None or self.feature_embedding is None:
            return None, None
        feature_embeddings = self.feature_embedding(feature_ids)
        masked_embeddings = feature_embeddings * mask[..., None]
        summed_embeddings = tf.reduce_sum(masked_embeddings, axis=1)
        feature_count = tf.reduce_sum(mask, axis=1, keepdims=True)
        feature_count = tf.maximum(feature_count, 1.0)
        averaged_embeddings = summed_embeddings / feature_count
        linear_terms = tf.reduce_sum(
            tf.squeeze(self.feature_linear(feature_ids), axis=-1) * mask,
            axis=1
        )
        return averaged_embeddings, linear_terms

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        user = tf.reshape(user, (-1,))
        item = tf.reshape(item, (-1,))

        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)

        field_embeddings = [user_mf_e, item_mf_e]
        linear_terms = tf.squeeze(self.user_linear(user), axis=-1) + tf.squeeze(self.item_linear(item), axis=-1)

        feature_embeddings = None
        if self.num_features > 0:
            feature_ids, mask = self._gather_item_features(item)
            feature_embeddings, feature_linear = self._aggregate_feature_embeddings(feature_ids, mask)
            field_embeddings.append(feature_embeddings)
            linear_terms += feature_linear

        stacked_embeddings = tf.stack(field_embeddings, axis=1)
        sum_embeddings = tf.reduce_sum(stacked_embeddings, axis=1)
        sum_square = tf.square(sum_embeddings)
        square_sum = tf.reduce_sum(tf.square(stacked_embeddings), axis=1)
        fm_interaction = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1)

        nn_input = tf.concat(field_embeddings, axis=1)
        hidden_output = self.hidden(nn_input)
        nn_output = tf.squeeze(self.prediction_layer(hidden_output), axis=-1)

        logits = self.bias_ + linear_terms + fm_interaction + nn_output
        return tf.nn.sigmoid(logits)

    @tf.function
    def train_step(self, batch):
        user, pos, label = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            user = tf.reshape(user, (-1,))
            pos = tf.reshape(pos, (-1,))

            user_mf_e = self.user_mf_embedding(user)
            item_mf_e = self.item_mf_embedding(pos)

            field_embeddings = [user_mf_e, item_mf_e]
            linear_terms = tf.squeeze(self.user_linear(user), axis=-1) + tf.squeeze(self.item_linear(pos), axis=-1)

            if self.num_features > 0:
                feature_ids, mask = self._gather_item_features(pos)
                feature_embeddings, feature_linear = self._aggregate_feature_embeddings(feature_ids, mask)
                field_embeddings.append(feature_embeddings)
                linear_terms += feature_linear

            stacked_embeddings = tf.stack(field_embeddings, axis=1)
            sum_embeddings = tf.reduce_sum(stacked_embeddings, axis=1)
            sum_square = tf.square(sum_embeddings)
            square_sum = tf.reduce_sum(tf.square(stacked_embeddings), axis=1)
            fm_interaction = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1)

            nn_input = tf.concat(field_embeddings, axis=1)
            hidden_output = self.hidden(nn_input, training=True)
            nn_output = tf.squeeze(self.prediction_layer(hidden_output), axis=-1)

            logits = self.bias_ + linear_terms + fm_interaction + nn_output
            loss = self.loss(label, logits)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    @tf.function
    def predict(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        user, item = inputs
        user = tf.reshape(user, (-1,))
        item = tf.reshape(item, (-1,))

        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)

        field_embeddings = [user_mf_e, item_mf_e]
        linear_terms = tf.squeeze(self.user_linear(user), axis=-1) + tf.squeeze(self.item_linear(item), axis=-1)

        if self.num_features > 0:
            feature_ids, mask = self._gather_item_features(item)
            feature_embeddings, feature_linear = self._aggregate_feature_embeddings(feature_ids, mask)
            field_embeddings.append(feature_embeddings)
            linear_terms += feature_linear

        stacked_embeddings = tf.stack(field_embeddings, axis=1)
        sum_embeddings = tf.reduce_sum(stacked_embeddings, axis=1)
        sum_square = tf.square(sum_embeddings)
        square_sum = tf.reduce_sum(tf.square(stacked_embeddings), axis=1)
        fm_interaction = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1)

        nn_input = tf.concat(field_embeddings, axis=1)
        hidden_output = self.hidden(nn_input, training=training)
        nn_output = tf.squeeze(self.prediction_layer(hidden_output), axis=-1)

        logits = self.bias_ + linear_terms + fm_interaction + nn_output
        return tf.nn.sigmoid(logits)

    @tf.function
    def get_recs(self, inputs, training=False, **kwargs):
        """
        Get full predictions on the whole users/items matrix.

        Returns:
            The matrix of predicted values.
        """
        preds = self.predict(inputs, training=training)
        return tf.squeeze(preds)

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)
