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

from tensorforce2.core.preprocessing.preprocessor import Preprocessor
from tensorforce2.core.preprocessing.sequence import Sequence
from tensorforce2.core.preprocessing.standardize import Standardize
from tensorforce2.core.preprocessing.running_standardize import RunningStandardize
from tensorforce2.core.preprocessing.normalize import Normalize
from tensorforce2.core.preprocessing.grayscale import Grayscale
from tensorforce2.core.preprocessing.image_resize import ImageResize
from tensorforce2.core.preprocessing.divide import Divide
from tensorforce2.core.preprocessing.clip import Clip
from tensorforce2.core.preprocessing.preprocessing import Preprocessing
from tensorforce2.core.preprocessing.crop import Crop


preprocessors = dict(
    sequence=Sequence,
    standardize=Standardize,
    running_standardize=RunningStandardize,
    normalize=Normalize,
    grayscale=Grayscale,
    image_resize=ImageResize,
    divide=Divide,
    clip=Clip,
    crop=Crop,
)


__all__ = [
    'Preprocessor',
    'Sequence',
    'Standardize',
    'RunningStandardize',
    'Normalize',
    'Grayscale',
    'ImageResize',
    'Preprocessing',
    'Divide',
    'Clip',
    'Crop',
    'preprocessors'
]
