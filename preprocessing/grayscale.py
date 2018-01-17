# Copyright 2017 reinforce.io. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorforce2.core.preprocessing import Preprocessor


class Grayscale(Preprocessor):
    """
    Turn 3D color state into grayscale.
    """

    def __init__(self, weights=(0.299, 0.587, 0.114), binarypixels=False, threshold=128):
        super(Grayscale, self).__init__()
        self.weights = weights
        self.binarypixels = binarypixels
        self.threshold = threshold

    def binary(self, image, threshold):
        image = (image.astype(np.float32) // threshold) * threshold
        return image

    def process(self, state):
        state = (self.weights * state).sum(-1)
        state = np.reshape(state, tuple(state.shape) + (1,))
        if self.binarypixels:
            state = self.binary(state, self.threshold)
        return state

    def processed_shape(self, shape):
        return tuple(shape[:-1]) + (1,)


