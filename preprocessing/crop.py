from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorforce.core.preprocessing import Preprocessor




class Crop(Preprocessor):
    """
    Crop image to size.
    """
    def __init__(self, leftpx=0, rightpx=0, uppx=0, downpx=0):
        super(Crop, self).__init__()
        self.downpx = downpx
        self.uppx = uppx
        self.leftpx = leftpx
        self.rightpx = rightpx
        self.size = (rightpx-leftpx, downpx-uppx)

    def process(self, state):
        state = state[self.leftpx:self.rightpx, self.uppx:self.downpx]
        return state

    def processed_shape(self, shape):
        return self.size + (shape[-1],)