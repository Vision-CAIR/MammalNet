#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

from cProfile import label
import math
import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.models.contrastive import (
    # contrastive_forward,
    contrastive_parameter_surgery,
)

import torch.nn.functional as F


from slowfast.utils.meters_composition import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)
from itertools import product
import pickle as pkl
import time



pair2idx = None

def get_label(animal_label, behavior_label):
    global pair2idx

    if pair2idx is None:
        animals = []
        behaviors = []

        tag = True
        while tag:
            try:
                with open("/ibex/ai/project/c2133/benchmarks/mvit2/SlowFast/animalnet_trimmed_last/id_to_genus.pkl","rb") as f:
                    id_to_genus = pkl.load(f)
                    for key in id_to_genus.keys():
                        animals.append(key)
                    tag = False
            except:
                time.sleep(0.1)
                
        

        tag = True
        while tag:
            try:
                with open("/ibex/ai/project/c2133/benchmarks/mvit2/SlowFast/animalnet_trimmed_last/id_to_label.pkl","rb") as f:
                    id_to_behavior = pkl.load(f)
                    for key in id_to_behavior.keys():
                        behaviors.append(key)
                    
                    tag = False
            except:
                time.sleep(0.1)


        full_pairs = list(product(animals, behaviors))

        pair2idx = {pair : idx for idx, pair in enumerate(full_pairs)}
    
    batch_size = animal_label.shape[0]
    
    index_list = []

    for index in range(batch_size):
        index_list.append(pair2idx[(animal_label[index].item(),behavior_label[index].item())])
    
    # print("pair 2index",len(pair2idx))
    # print(animal_label.shape,behavior_label.shape,type(pair2idx))


    # print("label to index", pair2idx)

    # print("animal_label",type(animal_label))
    # print("index list", index_list)

    return torch.tensor(index_list).cuda()
    # assert False
    # pass

def compute_loss(y,gt):
    return F.cross_entropy(50* y, gt)


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    iters_noupdate = 0
    if (
        cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.TYPE == "moco"
    ):
        assert (
            cfg.CONTRASTIVE.QUEUE_LEN % (cfg.TRAIN.BATCH_SIZE * cfg.NUM_SHARDS)
            == 0
        )
        iters_noupdate = (
            cfg.CONTRASTIVE.QUEUE_LEN // cfg.TRAIN.BATCH_SIZE // cfg.NUM_SHARDS
        )
    if cfg.MODEL.FROZEN_BN:
        misc.frozen_bn_stats(model)
    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    for cur_iter, (inputs, labels,behavior_labels,index, time, meta) in enumerate(
        train_loader
    ):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            behavior_labels = behavior_labels.cuda()


            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            index = index.cuda()
            time = time.cuda()
        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels,behavior_labels = mixup_fn(inputs[0], labels,behavior_labels)
            inputs[0] = samples
        

        # print("labels shape",labels.shape)
        # print("behavior shape",behavior_labels.shape)


        # print("full pairs",full_pairs.shape,type(full_pairs))
        # assert False



        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # Explicitly declare reduction to mean.
            perform_backward = True
            optimizer.zero_grad()

            if cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                (
                    model,
                    preds,
                    partial_loss,
                    perform_backward,
                ) = contrastive_forward(
                    model, cfg, inputs, index, time, epoch_exact, scaler
                )
       
            elif cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])
            else:


                # behavior_prediction,animal_prediction = model(inputs) # previous
                
                # print("labels", labels)
                # assert False
                behavior_prediction,animal_prediction = model(inputs, labels, behavior_labels) # previous
                # behavior_prediction,animal_prediction = model(inputs,labels,behavior_labels)





        if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            labels = torch.zeros(
                preds.size(0), dtype=labels.dtype, device=labels.device
            )


        if cfg.MODEL.MODEL_NAME == "ContrastiveModel" and partial_loss:
            loss = partial_loss
        else:
            # Compute the loss.
          
            # loss = loss_fun(preds, labels)
            

            # loss = 0.5 * loss_fun(animal_prediction,labels) + 0.5 * loss_fun(behavior_prediction,behavior_labels)
            loss = 0.5 * compute_loss(animal_prediction, labels) + 0.5 * compute_loss(behavior_prediction, behavior_labels)
            # new_label = get_function(labels, behavior_labels)

            # pair2idx = model.pair2idx

            # compose_labels = get_label(labels,behavior_labels)

            # loss = loss_fun(animal_behavior_prediction,compose_labels)



        # check Nan Loss.
        misc.check_nan_losses(loss)

        if perform_backward:
            scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )

        # model = cancel_swav_gradients(model, cfg, epoch_exact)
        
        
        model, update_param = contrastive_parameter_surgery(
            model, cfg, epoch_exact, cur_iter
        )
        if cur_iter < iters_noupdate and cur_epoch == 0:  #  for e.g. MoCo
            logger.info(
                "Not updating parameters {}/{}".format(cur_iter, iters_noupdate)
            )
        else:
            # Update the parameters.
            scaler.step(optimizer)
        scaler.update()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.

                animal_loss = loss_fun(animal_prediction,labels)
                behavior_loss = loss_fun(behavior_prediction,behavior_labels)

                composition_loss = 0.5 * animal_loss + 0.5 * behavior_loss
                num_topks_correct_animal, num_topks_correct_behavior,num_topks_correct_composition = metrics.topks_correct_composition(animal_prediction, labels, behavior_prediction,behavior_labels, (1, 5))
                
                
                top1_err_animal, top5_err_animal = [
                    (1.0 - x / animal_prediction.size(0)) * 100.0 for x in num_topks_correct_animal
                ]

                top1_err_behavior, top5_err_behavior = [
                    (1.0 - x / animal_prediction.size(0)) * 100.0 for x in num_topks_correct_behavior
                ]

                top1_err_composition, top5_err_composition = [
                    (1.0 - x / animal_prediction.size(0)) * 100.0 for x in num_topks_correct_composition
                ]





                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    animal_loss, top1_err_animal, top5_err_animal = du.all_reduce(
                        [animal_loss.detach(), top1_err_animal, top5_err_animal]
                    )


                # Copy the stats from GPU to CPU (sync point).
                animal_loss, top1_err_animal, top5_err_animal = (
                    animal_loss.item(),
                    top1_err_animal.item(),
                    top5_err_animal.item(),
                )





                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    behavior_loss, top1_err_behavior, top5_err_behavior = du.all_reduce(
                        [behavior_loss.detach(), top1_err_behavior, top5_err_behavior]
                    )
                    

                # Copy the stats from GPU to CPU (sync point).
                behavior_loss, top1_err_behavior, top5_err_behavior = (
                    behavior_loss.item(),
                    top1_err_behavior.item(),
                    top5_err_behavior.item(),
                )


                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    composition_loss, top1_err_composition, top5_err_composition = du.all_reduce(
                        [composition_loss.detach(), top1_err_composition, top5_err_composition]
                    )
                    

                # Copy the stats from GPU to CPU (sync point).
                composition_loss, top1_err_composition, top5_err_composition = (
                    composition_loss.item(),
                    top1_err_composition.item(),
                    top5_err_composition.item(),
                )









            # Update and log stats.
            train_meter.update_stats(
                top1_err_animal,
                top5_err_animal,
                animal_loss,
                lr,
                batch_size
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                "animal",
            )



            # Update and log stats.
            train_meter.update_stats(
                top1_err_behavior,
                top5_err_behavior,
                behavior_loss,
                lr,
                batch_size
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                "behavior",
            )


            # Update and log stats.
            train_meter.update_stats(
                top1_err_composition,
                top5_err_composition,
                composition_loss,
                lr,
                batch_size
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                "composition",
            )


            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss_animal": animal_loss,
                        "Train/lr": lr,
                        "Train/Top1_err_animal": top1_err_animal,
                        "Train/Top5_err_animal": top5_err_animal,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )



            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss_behavior": behavior_loss,
                        "Train/lr": lr,
                        "Train/Top1_err_behavior": top1_err_behavior,
                        "Train/Top5_err_behavior": top5_err_behavior,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )


            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss_composition": composition_loss,
                        "Train/lr": lr,
                        "Train/Top1_err_composition": top1_err_composition,
                        "Train/Top5_err_composition": top5_err_composition,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )


        torch.cuda.synchronize()
        train_meter.iter_toc()  # do measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter,"animal")
        train_meter.log_iter_stats(cur_epoch, cur_iter,"behavior")
        train_meter.log_iter_stats(cur_epoch, cur_iter,"composition")
        torch.cuda.synchronize()
        train_meter.iter_tic()
    del inputs
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch,"animal")
    train_meter.log_epoch_stats(cur_epoch,"behavior")
    train_meter.log_epoch_stats(cur_epoch,"composition")

    train_meter.reset()


@torch.no_grad()
def eval_epoch(
    val_loader, model, val_meter, cur_epoch, cfg, train_loader, writer
):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, behavior_labels, index, time, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            behavior_labels = behavior_labels.cuda()

            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            index = index.cuda()
            time = time.cuda()
        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                if not cfg.CONTRASTIVE.KNN_ON:
                    return
                train_labels = (
                    model.module.train_labels
                    if hasattr(model, "module")
                    else model.train_labels
                )
                yd, yi = model(inputs, index, time)
                K = yi.shape[1]
                C = (
                    cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM
                )  # eg 400 for Kinetics400
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)
                retrieval_one_hot = torch.zeros((batch_size * K, C)).cuda()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
                probs = torch.mul(
                    retrieval_one_hot.view(batch_size, -1, C),
                    yd_transform.view(batch_size, -1, 1),
                )
                preds = torch.sum(probs, 1)
            else:
                behavior_prediction,animal_prediction = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct_animal, num_topks_correct_behavior,num_topks_correct_composition = metrics.topks_correct_composition(animal_prediction, labels, behavior_prediction,behavior_labels, (1, 5))
                
                top1_err_animal, top5_err_animal = [
                    (1.0 - x / animal_prediction.size(0)) * 100.0 for x in num_topks_correct_animal
                ]

                top1_err_behavior, top5_err_behavior = [
                    (1.0 - x / animal_prediction.size(0)) * 100.0 for x in num_topks_correct_behavior
                ]

                top1_err_composition, top5_err_composition = [
                    (1.0 - x / animal_prediction.size(0)) * 100.0 for x in num_topks_correct_composition
                ]





                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    top1_err_animal, top5_err_animal = du.all_reduce(
                        [top1_err_animal, top5_err_animal]
                    )


                # Copy the stats from GPU to CPU (sync point).
                top1_err_animal, top5_err_animal = (
                    top1_err_animal.item(),
                    top5_err_animal.item(),
                )





                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    top1_err_behavior, top5_err_behavior = du.all_reduce(
                        [ top1_err_behavior, top5_err_behavior]
                    )
                    

                # Copy the stats from GPU to CPU (sync point).
                top1_err_behavior, top5_err_behavior = (
                    top1_err_behavior.item(),
                    top5_err_behavior.item(),
                )






                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    top1_err_composition, top5_err_composition = du.all_reduce(
                        [top1_err_composition, top5_err_composition]
                    )
                    

                # Copy the stats from GPU to CPU (sync point).
                top1_err_composition, top5_err_composition = (
                    top1_err_composition.item(),
                    top5_err_composition.item(),
                )




                # val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err_animal,
                    top5_err_animal,
                    batch_size
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                    "animal",
                )

                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err_animal": top1_err_animal, "Val/Top5_err_animal": top5_err_animal},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )
                # val_meter.iter_toc()

                val_meter.update_stats(
                    top1_err_behavior,
                    top5_err_behavior,
                    batch_size
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                    "behavior",
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err_behavior": top1_err_behavior, "Val/Top5_err_behavior": top5_err_behavior},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

                # val_meter.iter_toc()
                val_meter.update_stats(
                    top1_err_composition,
                    top5_err_composition,
                    batch_size
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                    "composition",
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err_composition": top1_err_composition, "Val/Top5_err_composition": top5_err_composition},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )





      
            
            # print("preds, groud truth", preds, labels)
            val_meter.update_predictions(animal_prediction, labels,"animal")
            val_meter.update_predictions(behavior_prediction, behavior_labels,"behavior")

        val_meter.log_iter_stats(cur_epoch, cur_iter,"animal")
        val_meter.log_iter_stats(cur_epoch, cur_iter,"behavior")
        val_meter.log_iter_stats(cur_epoch, cur_iter,"composition")
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch,"animal")
    val_meter.log_epoch_stats(cur_epoch,"behavior")
    val_meter.log_epoch_stats(cur_epoch,"composition")
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:


            all_preds_animal = [pred.clone().detach() for pred in val_meter.all_preds_animal]
            all_labels_animal = [
                label.clone().detach() for label in val_meter.all_labels_animal
            ]
            if cfg.NUM_GPUS:
                all_preds_animal = [pred.cpu() for pred in all_preds_animal]
                all_labels_animal = [label.cpu() for label in all_labels_animal]
            writer.plot_eval(
                preds=all_preds_animal, labels=all_labels_animal, global_step=cur_epoch
            )



            all_preds_behavior = [pred.clone().detach() for pred in val_meter.all_preds_behavior]
            all_labels_behavior = [
                label.clone().detach() for label in val_meter.all_labels_behavior
            ]
            if cfg.NUM_GPUS:
                all_preds_behavior = [pred.cpu() for pred in all_preds_behavior]
                all_labels_behavior = [label.cpu() for label in all_labels_behavior]
            writer.plot_eval(
                preds=all_preds_behavior, labels=all_labels_behavior, global_step=cur_epoch
            )



            all_preds_composition = [pred.clone().detach() for pred in val_meter.all_preds_composition]
            all_labels_composition = [
                label.clone().detach() for label in val_meter.all_labels_composition
            ]
            if cfg.NUM_GPUS:
                all_preds_composition = [pred.cpu() for pred in all_preds_composition]
                all_labels_composition = [label.cpu() for label in all_labels_composition]
            writer.plot_eval(
                preds=all_preds_composition, labels=all_labels_composition, global_step=cur_epoch
            )
    val_meter.reset()


def contrastive_forward(model, cfg, inputs, index, time, epoch_exact, scaler):
    if cfg.CONTRASTIVE.SEQUENTIAL:
        perform_backward = False
        mdl = model.module if hasattr(model, "module") else model
        keys = (
            mdl.compute_key_feat(
                inputs,
                compute_predictor_keys=False,
                batched_inference=True if len(inputs) < 2 else False,
            )
            if cfg.CONTRASTIVE.TYPE == "moco" or cfg.CONTRASTIVE.TYPE == "byol"
            else [None] * len(inputs)
        )
        for k, vid in enumerate(inputs):
            other_keys = keys[:k] + keys[k + 1 :]
            time_cur = torch.cat(
                [
                    time[:, k : k + 1, :],
                    time[:, :k, :],
                    time[:, k + 1 :, :],
                ],
                1,
            )  # q, kpre, kpost
            vids = [vid]
            if (
                cfg.CONTRASTIVE.TYPE == "swav"
                or cfg.CONTRASTIVE.TYPE == "simclr"
            ):
                if k < len(inputs) - 1:
                    vids = inputs[k : k + 2]
                else:
                    break
            lgt_k, loss_k = model(
                vids, index, time_cur, epoch_exact, keys=other_keys
            )
            scaler.scale(loss_k).backward()
            if k == 0:
                preds, partial_loss = lgt_k, loss_k.detach()
            else:
                preds = torch.cat([preds, lgt_k], dim=0)
                partial_loss += loss_k.detach()
        partial_loss /= len(inputs) * 2.0  # to have same loss as symm model
        if cfg.CONTRASTIVE.TYPE == "moco":
            mdl._dequeue_and_enqueue(keys)
    else:
        perform_backward = True
        preds, partial_loss = model(inputs, index, time, epoch_exact, keys=None)
    return model, preds, partial_loss, perform_backward


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task=cfg.TASK)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
            start_epoch = checkpoint_epoch + 1
        elif "ssl_eval" in cfg.TASK:
            last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task="ssl")
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
                epoch_reset=True,
                clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            )
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = 0
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        print("load from given checkpoint files")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    if (
        cfg.TASK == "ssl"
        and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.KNN_ON
    ):
        if hasattr(model, "module"):
            model.module.init_knn_labels(train_loader)
        else:
            model.init_knn_labels(train_loader)

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        if cur_epoch > 0 and cfg.DATA.LOADER_CHUNK_SIZE > 0:
            num_chunks = math.ceil(
                cfg.DATA.LOADER_CHUNK_OVERALL_SIZE / cfg.DATA.LOADER_CHUNK_SIZE
            )
            skip_rows = (cur_epoch) % num_chunks * cfg.DATA.LOADER_CHUNK_SIZE
            logger.info(
                f"=================+++ num_chunks {num_chunks} skip_rows {skip_rows}"
            )
            cfg.DATA.SKIP_ROWS = skip_rows
            logger.info(f"|===========| skip_rows {skip_rows}")
            train_loader = loader.construct_loader(cfg, "train")
            loader.shuffle_dataset(train_loader, cur_epoch)

        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(
                        cfg.OUTPUT_DIR, task=cfg.TASK
                    )
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(cur_epoch)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
            writer,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = (
            cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                train_loader,
                writer,
            )
    if writer is not None:
        writer.close()
    result_string = "Top1 Acc: {:.2f} Top5 Acc: {:.2f} MEM: {:.2f}" "".format(
        100 - val_meter.min_top1_err_animal,
        100 - val_meter.min_top5_err_animal,
        misc.gpu_mem_usage(),
    )
    logger.info("training done: {}".format(result_string))


    result_string = "Top1 Acc: {:.2f} Top5 Acc: {:.2f} MEM: {:.2f}" "".format(
        100 - val_meter.min_top1_err_behavior,
        100 - val_meter.min_top5_err_behavior,
        misc.gpu_mem_usage(),
    )
    logger.info("training done: {}".format(result_string))

    result_string = "Top1 Acc: {:.2f} Top5 Acc: {:.2f} MEM: {:.2f}" "".format(
        100 - val_meter.min_top1_err_composition,
        100 - val_meter.min_top5_err_composition,
        misc.gpu_mem_usage(),
    )
    logger.info("training done: {}".format(result_string))

    return result_string
