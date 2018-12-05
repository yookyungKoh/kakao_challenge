# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import Dense, Input
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Activation

from misc import get_logger, Option
opt = Option('./config.json')


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)


class TextOnly:
    def __init__(self):
        self.logger = get_logger('textonly')

    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            cate1_embed = Embedding(57, 10)
            cate2_embed = Embedding(552, 10)
            cate3_embed = Embedding(3190, 10)
            
            t_uni = Input((max_len,), name="input_1")
            t_uni_embd = embd(t_uni)  # token
            print('embedded input:', t_unit_embd.shape)

            w_uni = Input((max_len,), name="input_2")
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1) # dot product (x, w) --> weighted token embedding
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)
            
            embd_out = Dropout(rate=0.5)(uni_embd)
            relu = Activation('relu', name='relu1')(embd_out)
            print('after relu1:', relu.shape)

            out1 = Dense(57, activation=activation)(relu)
            # cate1
            y1 = cate1_embed(argmax(out1))

            print('cate1 hidden:', relu.shape)
            print('concat with y1:', keras.backend.stack(relu, out1).shape)
            
            h1 = keras.backend.stack(relu, out1)
            out2 = Dense(552, activation=activation)(h1)
            y2 = cate2_embed(keras.backend.argmax(out2))
            h2 = keras.backend.stack(h1, out2)
            out3 = Dense(3190, activation=activation)(h2)
            y3 = cate3_embed(keras.backend.argmax(out3))
            h3 = keras.backend.stack(h2, out3)
            out4 = Dense(404, activation=activation)(h3)

            model = Model(inputs=[t_uni, w_uni], outputs=outputs)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(loss='binary_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model
