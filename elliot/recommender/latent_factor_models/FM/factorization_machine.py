"""
Module description:

"""


__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Antonio Ferrara'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it,' \
            'daniele.malitesta@poliba.it, antonio.ferrara@poliba.it'

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from elliot.dataset.samplers import pointwise_pos_neg_ratings_sampler as pws
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.latent_factor_models.FM.factorization_machine_model import FactorizationMachineModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.recommender.base_recommender_model import init_charger


class FM(RecMixin, BaseRecommenderModel):
    r"""
    Factorization Machines

    For further details, please refer to the `paper <https://ieeexplore.ieee.org/document/5694074>`_

    Args:
        factors: Number of factors of feature embeddings
        lr: Learning rate
        reg: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        FM:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          reg: 0.1
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "factors", 10, int, None),
            ("_learning_rate", "lr", "lr", 0.001, float, None),
            ("_l_w", "reg", "reg", 0.1, float, None),
            ("_loader", "loader", "load", "ItemAttributes", None, None),
            ("_recs_user_chunk_size", "user_chunk_size", "uch", 8, int, None),
            ("_recs_item_chunk_size", "item_chunk_size", "ich", 256, int, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        if self._recs_user_chunk_size < 1:
            self._recs_user_chunk_size = 1
        if self._recs_item_chunk_size < 1:
            self._recs_item_chunk_size = 64

        if (hasattr(self._side, "nfeatures")) and (hasattr(self._side, "feature_map")):
            self._nfeatures = self._side.nfeatures
        else:
            self._nfeatures = 0

        self._item_feature_indices = [None] * self._num_items
        if self._nfeatures:
            feature_map = getattr(self._side, "feature_map", {})
            public_features = getattr(self._side, "public_features", {})
            for internal_item, original_item in self._data.private_items.items():
                feats = feature_map.get(original_item, [])
                mapped = [public_features[f] for f in feats if f in public_features]
                self._item_feature_indices[internal_item] = mapped if mapped else None

        self._field_dims = [self._num_users, self._num_items, self._nfeatures]

        self._sampler = pws.Sampler(self._data.i_train_dict, self._data.sp_i_train_ratings)

        self._model = FactorizationMachineModel(self._num_users,
                                                self._num_items,
                                                self._nfeatures,
                                                self._factors,
                                                self._l_w,
                                                self._learning_rate,
                                                self._seed)


    @property
    def name(self):
        return "FM" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def predict(self, u: int, i: int):
        pass

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    if self._nfeatures:
                        prepared_batch = self.prepare_fm_transaction(batch)
                        loss += self._model.train_step(prepared_batch)
                    else:
                        u,i,r = batch
                        loss += self._model.train_step(((u, i), r))
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss.numpy()/(it + 1))

    def prepare_fm_transaction(self, batch):
        batch_users = np.array(batch[0])
        batch_users = np.array(batch[0])
        batch_items = np.array(batch[1])
        user_array = np.zeros((batch_users.size, self._num_users), dtype=np.float32)
        user_array[np.arange(batch_users.size), batch_users] = 1

        item_array = np.zeros((batch_items.size, self._num_items), dtype=np.float32)
        item_array[np.arange(batch_items.size), batch_items] = 1

        if self._nfeatures:
            feature_array = np.zeros((batch_items.size, self._nfeatures), dtype=np.float32)
            for row_idx, item in enumerate(batch_items):
                feat_idx = self._item_feature_indices[item]
                if feat_idx:
                    feature_array[row_idx, feat_idx] = 1.0
            features = np.hstack((user_array, item_array, feature_array))
        else:
            features = np.hstack((user_array, item_array))

        return features.astype(np.float32, copy=False), batch[2].astype(np.float32, copy=False)

    def _build_item_block(self, item_ids):
        block = np.zeros((len(item_ids), self._num_items), dtype=np.float32)
        block[np.arange(len(item_ids)), item_ids] = 1
        if self._nfeatures:
            feature_block = np.zeros((len(item_ids), self._nfeatures), dtype=np.float32)
            for idx, internal_item in enumerate(item_ids):
                feat_idx = self._item_feature_indices[internal_item]
                if feat_idx:
                    feature_block[idx, feat_idx] = 1.0
            return np.hstack((block, feature_block))
        return block

    def get_user_full_array(self, user):
        user_oh = np.zeros(self._num_users, dtype=np.float32)  # user one-hot encoding
        user_oh[user] = 1
        return user_oh

    def get_recommendations(self, k: int = 100):
        predictions_top_k_val = {}
        predictions_top_k_test = {}
        for offset in range(0, self._num_users, self._recs_user_chunk_size):
            offset_stop = min(offset + self._recs_user_chunk_size, self._num_users)
            users = list(range(offset, offset_stop))
            chunk_predictions = []
            for user in users:
                user_repr = self.get_user_full_array(user)
                user_scores_parts = []
                for item_start in range(0, self._num_items, self._recs_item_chunk_size):
                    item_stop = min(item_start + self._recs_item_chunk_size, self._num_items)
                    item_ids = np.arange(item_start, item_stop)
                    item_block = self._build_item_block(item_ids)
                    user_block = np.repeat(user_repr[None, :], item_stop - item_start, axis=0)
                    model_input = np.hstack((user_block, item_block)).astype(np.float32, copy=False)
                    preds = self._model.predict(model_input, training=False)
                    if isinstance(preds, tf.Tensor):
                        preds = preds.numpy()
                    user_scores_parts.append(np.reshape(preds, -1))
                chunk_predictions.append(np.concatenate(user_scores_parts))
            predictions = np.stack(chunk_predictions).astype(np.float32, copy=False)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test
