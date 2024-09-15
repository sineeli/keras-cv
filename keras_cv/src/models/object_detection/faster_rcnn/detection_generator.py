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

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.bounding_box.converters import _decode_deltas_to_boxes
from keras_cv.src.bounding_box.utils import _clip_boxes
from keras_cv.src.layers.object_detection.non_max_suppression import (
    non_max_suppression,
)


def generate_detections(
    boxes,
    scores,
    nms_confidence_threshold=0.5,
    nms_iou_threshold=0.5,
    max_num_detections=100,
):
    nmsed_boxes = []
    nmsed_classes = []
    nmsed_scores = []

    batch_size, _, _, _ = boxes.shape
    _, total_anchors, num_classes = scores.shape

    max_detections = min(max_num_detections, total_anchors)

    for i in range(num_classes):
        boxes_i = boxes[:, :, i, :]
        scores_i = scores[:, :, i]

        idx, _ = non_max_suppression(
            boxes_i,
            scores_i,
            max_output_size=max_detections,
            iou_threshold=nms_iou_threshold,
            score_threshold=nms_confidence_threshold,
        )

        nmsed_boxes_i = ops.take_along_axis(
            boxes_i, ops.expand_dims(idx, axis=-1), axis=1
        )
        nmsed_boxes_i = ops.reshape(nmsed_boxes_i, (-1, max_detections, 4))
        nmsed_scores_i = ops.take_along_axis(scores_i, idx, axis=1)

        nmsed_classes_i = ops.full(
            [batch_size, max_num_detections], fill_value=i
        )
        nmsed_boxes.append(nmsed_boxes_i)
        nmsed_scores.append(nmsed_scores_i)
        nmsed_classes.append(nmsed_classes_i)

    nmsed_boxes = ops.concatenate(nmsed_boxes, axis=1)
    nmsed_scores = ops.concatenate(nmsed_scores, axis=1)
    nmsed_classes = ops.concatenate(nmsed_classes, axis=1)

    nmsed_idx, valid_det = non_max_suppression(
        nmsed_boxes,
        nmsed_scores,
        max_output_size=max_num_detections,
        iou_threshold=nms_iou_threshold,
        score_threshold=nms_confidence_threshold,
    )
    boxes = ops.take_along_axis(
        nmsed_boxes, ops.expand_dims(nmsed_idx, axis=-1), axis=1
    )

    classes = ops.take_along_axis(nmsed_classes, nmsed_idx, axis=1)

    scores = ops.take_along_axis(nmsed_scores, nmsed_idx, axis=1)

    classes = ops.where(classes == -1, -1, classes + 1)

    return boxes, scores, classes, valid_det


@keras_cv_export(
    "keras_cv.models.faster_rcnn.DetectionGenerator",
    package="keras_cv.models.faster_rcnn",
)
class DetectionGenerator(keras.layers.Layer):
    def __init__(
        self,
        nms_iou_threshold=0.5,
        nms_confidence_threshold=0.5,
        max_num_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_confidence_threshold = nms_confidence_threshold
        self.max_num_detections = max_num_detections
        self.box_variance = box_variance
        self.built = True

    def call(self, raw_boxes, raw_scores, anchor_boxes, image_shape):
        box_scores = ops.softmax(raw_scores, axis=-1)

        # Removes the background class.
        box_scores_shape = ops.shape(box_scores)
        box_scores_shape_list = box_scores.shape
        batch_size = box_scores_shape[0]
        num_locations = box_scores_shape_list[1]
        num_classes = box_scores_shape_list[-1]

        box_scores = ops.slice(
            box_scores,
            start_indices=[0, 0, 1],
            shape=[
                box_scores.shape[0],
                box_scores.shape[1],
                box_scores.shape[2] - 1,
            ],
        )

        num_detections = num_locations * (num_classes - 1)

        raw_boxes = ops.reshape(
            raw_boxes, [batch_size, num_locations, num_classes, 4]
        )
        raw_boxes = ops.slice(
            raw_boxes,
            start_indices=[0, 0, 1, 0],
            shape=[
                raw_boxes.shape[0],
                raw_boxes.shape[1],
                raw_boxes.shape[2] - 1,
                raw_boxes.shape[3],
            ],
        )
        anchor_boxes = ops.tile(
            ops.expand_dims(anchor_boxes, axis=2), [1, 1, num_classes - 1, 1]
        )
        raw_boxes = ops.reshape(raw_boxes, [batch_size, num_detections, 4])
        anchor_boxes = ops.reshape(
            anchor_boxes, [batch_size, num_detections, 4]
        )

        # Box decoding.
        decoded_boxes = _decode_deltas_to_boxes(
            anchors=anchor_boxes,
            boxes_delta=raw_boxes,
            anchor_format="yxyx",
            box_format="yxyx",
            variance=self.box_variance,
            image_shape=image_shape,
        )

        decoded_boxes = _clip_boxes(decoded_boxes, "yxyx", image_shape)

        decoded_boxes = ops.reshape(
            decoded_boxes, [batch_size, num_locations, num_classes - 1, 4]
        )

        nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
            generate_detections(
                decoded_boxes,
                box_scores,
                nms_confidence_threshold=self.nms_confidence_threshold,
                nms_iou_threshold=self.nms_iou_threshold,
                max_num_detections=self.max_num_detections,
            )
        )
        bounding_boxes = {
            "boxes": nmsed_boxes,
            "confidence": nmsed_scores,
            "classes": nmsed_classes,
            "num_detections": valid_detections,
        }

        # this is required to comply with KerasCV bounding box format.
        return bounding_box.mask_invalid_detections(
            bounding_boxes, output_ragged=False
        )

    def get_config(self):
        config = super().get_config()
        config["nms_iou_threshold"] = self.nms_iou_threshold
        config["nms_confidence_threshold"] = self.nms_confidence_threshold
        config["max_num_detections"] = self.max_num_detections
        config["box_variance"] = self.box_variance

        return config
