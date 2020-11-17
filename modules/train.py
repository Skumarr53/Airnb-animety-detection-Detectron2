import logging
import os
import argparse
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
        MetadataCatalog,
        build_detection_test_loader,
        build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import ( ### Changes from original, I don't need all the different evaluators
        COCOEvaluator,
        DatasetEvaluators,
        inference_on_dataset,
        print_csv_format
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
# Not sure what these do but they may help to track experiments
from detectron2.utils.events import (
        CommonMetricPrinter,
        EventStorage,
        JSONWriter,
        TensorboardXWriter
)
from pdb import set_trace


# Setup logger
logger = logging.getLogger("detectron2")

# Changed from original: Create evaluator for COCOEvaluator only
# Since we are only using bounding boxes to begin with, our evaluator can be simple COCO style
def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create a COCOEvaluator
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator = COCOEvaluator(dataset_name=dataset_name,
                              cfg=cfg,
                              distributed=False,
                              output_dir=output_folder)
    return evaluator


# Create testing function
def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        # Create the evaluator
        evaluator = get_evaluator(cfg,
                                  dataset_name,
                                  output_folder=os.path.join(
                                      cfg.OUTPUT_DIR, "inference",
                                      dataset_name))
        # Make inference on dataset
        results_i = inference_on_dataset(model, data_loader, evaluator)
        # Update results dictionary
        results[dataset_name] = results_i

        print("### Returning results_i...")
        #print(results_i)
        #print(f"### Average Precision: {results_i['AP']}")
        # Let's get some communication happening
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            ## wandb.log()? TODO/NOTE: This may be something Weights & Biases can track
            #print("### Calculating results...")
            print_csv_format(results_i)

        # Check to see length of results
        if len(results) == 1:
            results = list(results.values())[0]
        #print("### Returning results...")
        #print(results)

        # TODO : log results_i dict with different parameters
        print("### Saving results to Weights & Biases...")
        wandb.log(results_i)
        
        return results


# Create training function
def do_train(cfg, model, resume=False):
    # Set model to training mode
    model.train()
    # Create optimizer from config file (returns torch.nn.optimizer.Optimizer)
    optimizer = build_optimizer(cfg, model)
    # Create scheduler for learning rate (returns torch.optim.lr._LR_scheduler)
    scheduler = build_lr_scheduler(cfg, optimizer)
    print(f"Scheduler: {scheduler}")

    # Create checkpointer
    checkpointer = DetectionCheckpointer(model,
                                         save_dir=cfg.OUTPUT_DIR,
                                         optimizer=optimizer,
                                         scheduler=scheduler)

    # Create start iteration (refernces checkpointer) - https://detectron2.readthedocs.io/modules/checkpoint.html#detectron2.checkpoint.Checkpointer.resume_or_load
    start_iter = (
        # This can be 0
        checkpointer.resume_or_load(
            cfg.MODEL.
            WEIGHTS,  # Use predefined model weights (pretrained model)
            resume=resume).get("iteration", -1) + 1)
    # Set max number of iterations
    max_iter = cfg.SOLVER.MAX_ITER

    # Create periodiccheckpoint
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer=checkpointer,
        # How often to make checkpoints?
        period=cfg.SOLVER.CHECKPOINT_PERIOD,
        max_iter=max_iter)

    # Create writers (for saving checkpoints?)
    writers = ([
        # Print out common metrics such as iteration time, ETA, memory, all losses, learning rate
        CommonMetricPrinter(max_iter=max_iter),
        # Write scalars to a JSON file such as loss values, time and more
        JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
        # Write all scalars such as loss values to a TensorBoard file for easy visualization
        TensorboardXWriter(cfg.OUTPUT_DIR),
    ] if comm.is_main_process() else [])

    ### Original note from script: ###
    # compared to "train_net.py", we do not support accurate timing and precise BN
    # here, because they are not trivial to implement

    # Build a training data loader based off the training dataset name in the config
    data_loader = build_detection_train_loader(cfg)

    # Start logging
    logger.info("Starting training from iteration {}".format(start_iter))

    # Store events
    with EventStorage(start_iter) as storage:
        # Loop through zipped data loader and iteration
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step(
            )  # update stroage with step - https://detectron2.readthedocs.io/modules/utils.html#detectron2.utils.events.EventStorage.step

            # Create loss dictionary by trying to model data
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            # Are losses infinite? If so, something is wrong
            assert torch.isfinite(losses).all(), loss_dict

            # TODO - Not quite sure what's happening here
            loss_dict_reduced = {
                k: v.item()
                for k, v in comm.reduce_dict(loss_dict).items()
            }
            # Sum up losses
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # # TODO: wandb.log()? log the losses
            # wandb.log({
            #         "Total loss": losses_reduced
            # })

            # Update storage
            if comm.is_main_process():
                # Store informate in storage - https://detectron2.readthedocs.io/modules/utils.html#detectron2.utils.events.EventStorage.put_scalars
                storage.put_scalars(total_loss=losses_reduced,
                                    **loss_dict_reduced)

            # Start doing PyTorch things
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # Add learning rate to storage information
            storage.put_scalar("lr",
                               optimizer.param_groups[0]["lr"],
                               smoothing_hint=False)
            # This is required for your learning rate to change!!!! (not having this meant my learning rate was staying at 0)
            scheduler.step()

            # Perform evaluation?
            if (cfg.TEST.EVAL_PERIOD > 0
                    and iteration % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter):
                do_test(cfg, model)
                # TODO - compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            # Log different metrics with writers
            if iteration - start_iter > 5 and (iteration % 20 == 0
                                               or iteration == max_iter):
                for writer in writers:
                    writer.write()

            # Update the periodic_checkpointer
            periodic_checkpointer.step(iteration)


# Create setup function
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(
        args.config_file)  # This will take some kind of model.yaml file
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # this logs the config and arguments passed to the command line to the output file

    # Load config YAML as dict
    cfg_yaml = cfg.load_yaml_with_base(
        os.path.join(cfg.OUTPUT_DIR, "config.yaml"))

    # default_config = get_cfg()
    # default_config_loaded = default_config.load_yaml_with_base("output/config.yaml")
    # default_config_loaded

    # TODO: turn config into YAML and save to weights & biases
    # TODO: Init wandb and add configs
    # Setup a new weights & biases run every time we run the setup() function
    wandb.init(project="airbnb-object-detection", sync_tensorboard=True)

    #print("### Printing config_yaml file to go into Weights & Biases")
    #print(cfg_yaml)
    wandb.config.update(cfg_yaml)

    return cfg

# Create main function
def main(args):
    
    # Create the config file
    cfg = setup(args)

    # Build the model
    model = build_model(cfg)
    
    # Log what's going on
    logger.info("Model:\n{}".format(model))

    # TODO: Fix this (if it doesn't work)
    #wandb.watch(model, log="all")

    # Only do evaluation if the args say so
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    # Do distributed training? (depends on number of GPUs available)
    distributed = comm.get_world_size() > 1
    if distributed:
        # Put the model on multiple devices if available
        model = DistributedDataParallel(
                model, 
                device_ids=[comm.get_local_rank()], 
                broadcast_buffers=False
        )

    # Train the model
    do_train(cfg, model)
    # TODO - May want to evaluate in a different step?
    return do_test(cfg, model)