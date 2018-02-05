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

import importlib
import numpy as np

from preprocess_error import PreprocessError


epsilon = 1e-6


def get_object(obj, predefined_objects=None, default_object=None, kwargs=None):
    """
    Utility method to map some kind of object specification to its content,
    e.g. optimizer or baseline specifications to the respective classes.

    Args:
        obj: A specification dict (value for key 'type' optionally specifies
                the object, options as follows), a module path (e.g.,
                my_module.MyClass), a key in predefined_objects, or a callable
                (e.g., the class type object).
        predefined_objects: Dict containing predefined set of objects,
                accessible via their key
        default_object: Default object is no other is specified
        kwargs: Arguments for object creation

    Returns: The retrieved object

    """
    args = ()
    kwargs = dict() if kwargs is None else kwargs

    if isinstance(obj, dict):
        kwargs.update(obj)
        obj = kwargs.pop('type', None)

    if predefined_objects is not None and obj in predefined_objects:
        obj = predefined_objects[obj]
    elif isinstance(obj, str):
        if obj.find('.') != -1:
            module_name, function_name = obj.rsplit('.', 1)
            module = importlib.import_module(module_name)
            obj = getattr(module, function_name)
        else:
            predef_obj_keys = list(predefined_objects.keys())
            raise PreprocessError("Error: object {} not found in predefined objects: {}".format(obj,predef_obj_keys))
    elif callable(obj):
        pass
    elif default_object is not None:
        args = (obj,)
        obj = default_object
    else:
        # assumes the object is already instantiated
        return obj

    return obj(*args, **kwargs)