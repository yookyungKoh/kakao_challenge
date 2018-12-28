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
import keras.backend as K
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import Dense, Input, Activation, Lambda, concatenate
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout

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
                             name='text_embd')
            
            text = Input((max_len,), name="text")
            text_embed = embd(text)  # (L, 128)
            t_input = K.permute_dimensions(text_embed, (0,2,1)) #(128, L)
            img = Input((2048,), name="image")
            img_feat = Reshape((2048, 1))(img)  # img feature (2048,1)

            alpha_b = Dense(1, activation='softmax')(text_embed) #(L, 1)
            alpha_b = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(alpha_b)
            context_b = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1,2]))([text_embed, alpha_b])
            
            alpha_m = Dense(1, activation='softmax')(text_embed)
            alpha_m = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(alpha_m)
            context_m = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1,2]))([text_embed, alpha_m])

            alpha_s = Dense(1, activation='softmax')(text_embed)
            alpha_s = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(alpha_s)
            context_s = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1,2]))([text_embed, alpha_s])

            alpha_d = Dense(1, activation='softmax')(text_embed)
            alpha_d = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(alpha_d)
            context_d = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1,2]))([text_embed, alpha_d])

            h1 = concatenate([context_b, context_m, context_s, context_d, img_feat], axis=1)
            h1 = Reshape((2560,))(h1)
            outputs = Dense(num_classes, activation=activation)(h1) 

            model = Model(inputs=[text, img], outputs=outputs)
            optm = keras.optimizers.Adam(opt.lr)
            model.compile(loss='binary_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))

        return model
