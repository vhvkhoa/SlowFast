#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import os
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

from . import transform as transform
from . import utils as utils
import slowfast.utils.logging as logging

import av

from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


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
        self.num_frames = cfg.DATA.NUM_FRAMES
        self.target_fps = target_fps

        logger.info("Constructing VideoDataset for video {}...".format(path_to_video))

        assert os.path.exists(path_to_video), "video {} not found".format(
            path_to_video
        )

        video_container = av.open(path_to_video)

        # Decode video. Meta info is used to perform selective decoding.
        video_stream = video_container.streams.video[0]

        fps = float(video_stream.average_rate)
        frames_length = video_stream.frames
        duration = video_stream.duration
        timebase_per_frame = duration / frames_length

        target_sampling_rate = self.cfg.DATA.SAMPLING_RATE * timebase_per_frame * fps / target_fps

        sampling_pts = torch.arange(0, duration + 1, target_sampling_rate)

        self.frames, idx = [], 0
        for frame in video_container.decode(video_stream):
            if idx < len(sampling_pts) and frame.pts >= sampling_pts[idx]:
                self.frames.append(frame.to_rgb().to_ndarray())
                idx += 1
        self.frames = torch.as_tensor(np.stack(self.frames))
        print(self.frames.size())
        print(math.ceil(len(self.frames) / self.num_frames))

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
        frame_index = index * self.num_frames
        frames = self.frames[frame_index: frame_index + self.num_frames]
        frames = frames.float()
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)

        shorter_side_size = self.cfg.DATA.TEST_CROP_SIZE
        frames, _ = transform.random_short_side_scale_jitter(frames, shorter_side_size, shorter_side_size)

        # Two pathways. First: [C T/4 H W]. Second: [C T H W]. Pad T to multiple of 4 if needed
        if frames.size(1) % 4 != 0:
            pad = tuple([0] * 2 * len(frames.size() - 2) + [0, 4 - frames.size(1) % 4])
            F.pad(frames[0], pad, mode='replicate')
        frames = utils.pack_pathway_output(self.cfg, frames)
        return index, frames

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return math.ceil(len(self.frames) / self.num_frames)
