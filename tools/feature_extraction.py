
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import os
import os.path as osp
import glob
import argparse
import json
from tqdm import tqdm
import numpy as np
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import model_builder
from slowfast.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


def perform_bbox_feature_extract(data_loader, model, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        data_loader (loader): video loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    all_features = {}

    with torch.no_grad():
        for inputs, bboxes, segment_indices, indices in data_loader:
            if len(bboxes) == 0:
                for i, index in enumerate(indices):
                    all_features[index.item()] = {
                        'features': [],
                        'segment': segment_indices[i].tolist()
                    }
                continue

            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            
            bboxes = bboxes.cuda(non_blocking=True)

            features = model(inputs, bboxes=bboxes)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                features = du.all_gather([features])

            for i, index in enumerate(indices):
                all_features[index.item()] = {
                    'features': features.cpu().tolist(),
                    'segment': segment_indices[i].tolist()
                }
    indices = sorted(all_features.keys(), key=lambda x: float(x))
    all_features = [all_features[idx] for idx in indices]

    return all_features


def perform_feature_extract(data_loader, model, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        data_loader (loader): video loader.
        model (model): the pretrained video model to test.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    all_features = {}

    with torch.no_grad():
        for inputs, segment_indices, indices in data_loader:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            
            features = model(inputs)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                features = du.all_gather([features])

            for i, index in enumerate(indices):
                all_features[index.item()] = {
                    'features': features.cpu().tolist(),
                    'segment': segment_indices[i].tolist()
                }
    indices = sorted(all_features.keys(), key=lambda x: float(x))
    all_features = [all_features[idx] for idx in indices]

    return all_features


def feature_extract(cfg, path_to_video_dir, path_to_feat_dir):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    if not os.path.isdir(path_to_feat_dir):
        os.makedirs(path_to_feat_dir)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Extract feature with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg, feature_extraction=True)
    if du.is_master_proc():
        misc.log_model_info(model)

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        # raise NotImplementedError("Unknown way to load checkpoint.")
        logger.info("Extracting features with random initialization. Only for debugging.")

    feature_extract_fn = perform_bbox_feature_extract if cfg.DETECTION.ENABLE else perform_feature_extract

    # Create video feature extraction loaders.
    path_to_videos = glob.glob(osp.join(path_to_video_dir, '*'))
    for video_idx, path_to_video in enumerate(tqdm(path_to_videos)):
        video_extraction_loader = loader.construct_loader(cfg, path_to_video)

        video_data = {
            'num_features': len(video_extraction_loader),
            'video_features': {}
        }

        if len(video_extraction_loader) > 0:
            video_features = feature_extract_fn(video_extraction_loader, model, cfg)
            assert all([feature != None for feature in video_features]), 'Missing some features !'
            video_data['video_features'] = video_features

        with open(osp.join(path_to_feat_dir, osp.splitext(osp.basename(path_to_video))[0] + '.json'), 'w') as f:
            json.dump(video_data, f)
