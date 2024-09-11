# Copyright 2024 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras


@keras_cv_export(
    "keras_cv.models.faster_rcnn.RPNHead",
    package="keras_cv.models.faster_rcnn",
)
class RPNHead(keras.layers.Layer):
    """A Keras layer implementing the RPN architecture.

    Region Proposal Networks (RPN) was first suggested in
    [FasterRCNN](https://arxiv.org/abs/1506.01497).
    This is an end to end trainable layer which proposes regions
    for a detector (RCNN).

    Args:
        num_achors_per_location: (Optional) the number of anchors per location,
            defaults to 3.
        num_filters: (Optional) number convolution filters
        kernel_size: (Optional) kernel size of the convolution filters.
        conv_depth: (Optional) Number convolution layers before
            object and class heads.
    """

    def __init__(
        self,
        num_anchors_per_location=3,
        num_filters=256,
        kernel_size=3,
        conv_depth=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.convs = []
        for _ in range(conv_depth):
            self.convs.append(
                keras.layers.Conv2D(
                    num_filters,
                    kernel_size=kernel_size,
                    activation="relu",
                    kernel_initializer=keras.initializers.RandomNormal(
                        stddev=0.01
                    ),
                )
            )

        self.seq_conv = keras.Sequential(self.convs)

        self.objectness_logits = keras.layers.Conv2D(
            filters=num_anchors_per_location * 1,
            kernel_size=1,
            strides=1,
            padding="valid",
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        )
        self.anchor_deltas = keras.layers.Conv2D(
            filters=num_anchors_per_location * 4,
            kernel_size=1,
            strides=1,
            padding="valid",
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        )

        # === config ===
        self.num_anchors = num_anchors_per_location
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_depth = conv_depth

    def call(self, feature_map, training=False):
        rpn_boxes = {}
        rpn_scores = {}
        for level in feature_map:
            x = feature_map[level]
            x = self.seq_conv(x)
            rpn_scores[level] = self.objectness_logits(x)
            rpn_boxes[level] = self.anchor_deltas(x)
        return rpn_boxes, rpn_scores

    def get_config(self):
        config = super().get_config()
        config["num_anchors_per_location"] = self.num_anchors
        config["num_filters"] = self.num_filters
        config["kernel_size"] = self.kernel_size
        config["conv_depth"] = self.conv_depth
        return config

    def build(self, input_shape):
        for conv in self.convs:
            conv.build((None, None, None, self.num_filters))

        self.objectness_logits.build((None, None, None, self.num_filters))
        self.anchor_deltas.build((None, None, None, self.num_filters))
        self.built = True
