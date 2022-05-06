# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from random import randint


def unpickle(file):
    with open(file, "rb") as fo:
        myDict = pickle.load(fo, encoding="latin1")
    return myDict


def get_labels():
    data = unpickle("imgs/cifar-100-python/meta")
    fine_labels = []
    coarse_labels = []
    # create array of fine labels (source of truth)
    for label in data["fine_label_names"]:
        fine_labels.append(label)
    for label in data["coarse_label_names"]:
        coarse_labels.append(label)
    return (fine_labels, coarse_labels)


def to_categorical(y, num_classes=None, dtype="float32"):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with `categorical_crossentropy`.
    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The class axis is placed
        last.
    Example:
    >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> a = tf.constant(a, shape=[4, 4])
    >>> print(a)
    tf.Tensor(
      [[1. 0. 0. 0.]
       [0. 1. 0. 0.]
       [0. 0. 1. 0.]
       [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)
    >>> b = tf.constant([.9, .04, .03, .03,
    ...                  .3, .45, .15, .13,
    ...                  .04, .01, .94, .05,
    ...                  .12, .21, .5, .17],
    ...                 shape=[4, 4])
    >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]
    >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


import tensorflow as tf
from keras.optimizer_v2 import optimizer_v2


class SGD(optimizer_v2.OptimizerV2):
    r"""Gradient descent (with momentum) optimizer.
    Update rule for parameter `w` with gradient `g` when `momentum` is 0:
    ```python
    w = w - learning_rate * g
    ```
    Update rule when `momentum` is larger than 0:
    ```python
    velocity = momentum * velocity - learning_rate * g
    w = w + velocity
    ```
    When `nesterov=True`, this rule becomes:
    ```python
    velocity = momentum * velocity - learning_rate * g
    w = w + momentum * velocity - learning_rate * g
    ```
    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to 0.01.
      momentum: float hyperparameter >= 0 that accelerates gradient descent
        in the relevant
        direction and dampens oscillations. Defaults to 0, i.e., vanilla gradient
        descent.
      nesterov: boolean. Whether to apply Nesterov momentum.
        Defaults to `False`.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to `"SGD"`.
      **kwargs: Keyword arguments. Allowed to be one of
        `"clipnorm"` or `"clipvalue"`.
        `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
        gradients by value.
    Usage:
    >>> opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    >>> var = tf.Variable(1.0)
    >>> loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
    >>> step_count = opt.minimize(loss, [var]).numpy()
    >>> # Step is `- learning_rate * grad`
    >>> var.numpy()
    0.9
    >>> opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    >>> var = tf.Variable(1.0)
    >>> val0 = var.value()
    >>> loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
    >>> # First step is `- learning_rate * grad`
    >>> step_count = opt.minimize(loss, [var]).numpy()
    >>> val1 = var.value()
    >>> (val0 - val1).numpy()
    0.1
    >>> # On later steps, step-size increases because of momentum
    >>> step_count = opt.minimize(loss, [var]).numpy()
    >>> val2 = var.value()
    >>> (val1 - val2).numpy()
    0.18
    Reference:
        - For `nesterov=True`, See [Sutskever et al., 2013](
          http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self, learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD", **kwargs
    ):
        super(SGD, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

        self._momentum = False
        if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError(
                f"`momentum` must be between [0, 1]. Received: "
                f"momentum={momentum} (of type {type(momentum)})."
            )
        self._set_hyper("momentum", momentum)

        self.nesterov = nesterov

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(SGD, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = tf.identity(
            self._get_hyper("momentum", var_dtype)
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        if self._momentum:
            momentum_var = self.get_slot(var, "momentum")
            return tf.raw_ops.ResourceApplyKerasMomentum(
                var=var.handle,
                accum=momentum_var.handle,
                lr=coefficients["lr_t"],
                grad=grad,
                momentum=coefficients["momentum"],
                use_locking=self._use_locking,
                use_nesterov=self.nesterov,
            )
        else:
            return tf.raw_ops.ResourceApplyGradientDescent(
                var=var.handle,
                alpha=coefficients["lr_t"],
                delta=grad,
                use_locking=self._use_locking,
            )

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, **kwargs):
        if self._momentum:
            return super(SGD, self)._resource_apply_sparse_duplicate_indices(
                grad, var, indices, **kwargs
            )
        else:
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = kwargs.get("apply_state", {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)

            return tf.raw_ops.ResourceScatterAdd(
                resource=var.handle,
                indices=indices,
                updates=-grad * coefficients["lr_t"],
            )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        momentum_var = self.get_slot(var, "momentum")
        return tf.raw_ops.ResourceSparseApplyKerasMomentum(
            var=var.handle,
            accum=momentum_var.handle,
            lr=coefficients["lr_t"],
            grad=grad,
            indices=indices,
            momentum=coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=self.nesterov,
        )

    def get_config(self):
        config = super(SGD, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "decay": self._initial_decay,
                "momentum": self._serialize_hyperparameter("momentum"),
                "nesterov": self.nesterov,
            }
        )
        return config


def test_image(img):
    # get list of fine and coarse labels
    (fine_labels, _) = get_labels()
    # load model
    model = load_model(filepath=os.getcwd() + "/ml/image_classifier.h5")
    # All predictions for every image!
    prediction = model.predict(img)
    guess_index = np.argmax(prediction)
    # Use state to get the index of correct answer
    return f"IClass guesses: {fine_labels[guess_index]}"
