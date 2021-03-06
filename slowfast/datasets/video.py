#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import os
import json
from shutil import rmtree

import numpy as np
import cv2
import torch
import torch.utils.data

from . import transform as transform
from . import utils as utils
import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


CAPTURED_CLASS_IDS = [0]
THRESHOLD = 0.5


@DATASET_REGISTRY.register()
class Video(torch.utils.data.Dataset):
    """
    Videos Dataset loader. Construct videos loader for features extraction.
    Given a list of videos being extracted
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, path_to_video, target_fps=30):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        self.cfg = cfg
        self.num_samples = cfg.DATA.NUM_FRAMES
        self.target_fps = target_fps

        assert os.path.exists(path_to_video), "video {} not found".format(
            path_to_video
        )

        if os.path.exists(cfg.DATA.PATH_TO_TMP_DIR):
            rmtree(cfg.DATA.PATH_TO_TMP_DIR)
        os.makedirs(cfg.DATA.PATH_TO_TMP_DIR)

        video = cv2.VideoCapture(path_to_video)
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = video.get(cv2.CAP_PROP_FPS)
        frames_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        target_sampling_rate = self.cfg.DATA.SAMPLING_RATE * fps / target_fps

        sampling_pts = torch.arange(0, frames_length, target_sampling_rate).tolist()

        frame_idx, new_frame_idx = -1, 0
        for sampling_idx in sampling_pts:
            sampling_idx = round(sampling_idx)
            for _ in range(sampling_idx - frame_idx):
                success, frame = video.read()

            if success:
                cv2.imwrite(os.path.join(cfg.DATA.PATH_TO_TMP_DIR, '%d.jpg' % new_frame_idx), frame)
                new_frame_idx += 1
            frame_idx = sampling_idx

        video.release()

        self.num_frames = new_frame_idx

        if cfg.DETECTION.ENABLE:
            assert os.path.isdir(cfg.DATA.PATH_TO_BBOX_DIR), 'Invalid DATA.PATH_TO_BBOX_DIR.'
            bbox_path = os.path.join(
                cfg.DATA.PATH_TO_BBOX_DIR,
                os.path.splitext(os.path.basename(path_to_video))[0] + '.json')

            with open(bbox_path, 'r') as f:
                bboxes_data = json.load(f)
                self.bboxes, self.pts = {}, []
                for frame_data in bboxes_data['video_bboxes']:
                    self.pts.append(frame_data['pts'])
                    frame_bboxes = []

                    for bbox in frame_data['frame_bboxes']:
                        if bbox['class_id'] in CAPTURED_CLASS_IDS and bbox['score'] > THRESHOLD:
                            box = bbox['box'].copy()
                            width, height = box[2] - box[0], box[3] - box[1]
                            assert width > 0 and height > 0, 'Width %d and height %d are not positive' % (width, height)
                            box[0] = max(0, box[0] - width / 5.)
                            box[1] = max(0, box[1] - height / 10.)
                            box[2] = min(video_width, box[2] + width / 5.)
                            box[3] = min(video_height, box[3] + height / 10.)

                            frame_bboxes.append(box)
                    frame_bboxes = sorted(
                        frame_bboxes,
                        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]),
                        reverse=True
                    )[:5]

                    self.bboxes[frame_data['pts']] = np.array(frame_bboxes)

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        # Perform color normalization.
        frame_index = index * self.num_samples
        segment_idx = torch.tensor([frame_index, frame_index + self.num_samples])

        frames = [
            cv2.cvtColor(
                cv2.imread(os.path.join(
                    self.cfg.DATA.PATH_TO_TMP_DIR,
                    '%d.jpg' % frame_idx)
                ),
                cv2.COLOR_BGR2RGB
            )
            for frame_idx in range(*segment_idx.tolist())
        ]
        frames = torch.tensor(np.stack(frames)).float()
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)

        if self.cfg.DETECTION.ENABLE:
            bboxes_pts = self.pts[index]
            begin_pts = frame_index / self.target_fps
            end_pts = (frame_index + self.num_samples) / self.target_fps
            assert bboxes_pts >= begin_pts and bboxes_pts < end_pts, 'Bbox {} lies outside the chunk scope [{}, {}].'.format(
                bboxes_pts, begin_pts, end_pts
            )
            bboxes = self.bboxes[bboxes_pts]
        else:
            bboxes = None

        shorter_side_size = self.cfg.DATA.TEST_CROP_SIZE
        frames, bboxes = transform.random_short_side_scale_jitter(frames, shorter_side_size, shorter_side_size, bboxes)

        # Two pathways. First: [C T/4 H W]. Second: [C T H W]. if T is not a multiple of 4, drop it.
        frames = utils.pack_pathway_output(self.cfg, frames)

        if self.cfg.DETECTION.ENABLE:
            return frames, bboxes, segment_idx, index
        else:
            return frames, segment_idx, index

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return math.floor(self.num_frames / self.num_samples)
