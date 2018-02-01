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

from tensorforce2 import util
from tensorforce2.core.preprocessing import Preprocessor


class Standardize(Preprocessor):
    """
    Standardize state. Subtract mean and divide by standard deviation.
    """

    def process(self, state):
        state = state.astype(np.float32)
        return (state - state.mean()) / (state.std() + util.epsilon)