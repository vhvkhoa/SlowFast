#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import numpy as np
import torch
import torch.utils.data

from . import transform as transform
from . import utils as utils
import slowfast.utils.logging as logging

import av

from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class VideoDataset(torch.utils.data.Dataset):
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

    def __init__(self, cfg, path_to_video, num_retries=10):
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

        logger.info("Constructing VideoDataset for video {}...".format(path_to_video))

        assert os.path.exists(path_to_video), "video {} not found".format(
            path_to_video
        )
        self._video_meta = {}

        video_container = av.open(self._path_to_video)

        # Decode video. Meta info is used to perform selective decoding.
        video_stream = video_container.streams.video[0]
        fps = float(video_stream.average_rate)
        target_fps = 30
        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        num_samples = video_stream.frames / (sampling_rate * fps / target_fps)

        self.frames = []
        for frame in video_container.decode(video_stream):
            self.frames.append(frame.to_rgb().to_ndarray())
        self.frames = torch.as_tensor(np.stack(self.frames))

        index = torch.linspace(0, len(self.frames), num_samples)
        index = torch.clamp(index, 0, self.frames.shape[0] - 1).long()
        self.frames = torch.index_select(self.frames, 0, index)

        # Perform color normalization.
        self.frames = self.frames.float()
        self.frames = self.frames / 255.0
        self.frames = self.frames - torch.tensor(self.cfg.DATA.MEAN)
        self.frames = self.frames / torch.tensor(self.cfg.DATA.STD)
        # T H W C -> C T H W.
        self.frames = self.frames.permute(3, 0, 1, 2)

        self.frames = utils.pack_pathway_output(self.cfg, self.frames)

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
        self.frames[index]

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
