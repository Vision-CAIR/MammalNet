#!/usr/bin/env python3
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader_composition
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters_composition import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
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
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels,behavior_labels, video_idx, time, meta) in enumerate(
        test_loader
    ):

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            behavior_labels = behavior_labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (
                model.module.train_labels
                if hasattr(model, "module")
                else model.train_labels
            )
            yd, yi = model(inputs, video_idx, time)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)
        else:
            # Perform the forward pass.
            behavior_prediction,animal_prediction = model(inputs)
        # Gather all the predictions across all the devices to perform ensemble.



        print("cur_iter",cur_iter)




        if cfg.NUM_GPUS > 1:
            all_animal_prediction, all_animal_labels = du.all_gather(
                [animal_prediction, labels]
            )
        if cfg.NUM_GPUS:
            all_animal_prediction = all_animal_prediction.cpu()
            all_animal_labels = all_animal_labels.cpu()
     
        

        if cfg.NUM_GPUS > 1:
            all_behavior_prediction, all_behavior_labels, video_idx = du.all_gather(
                [behavior_prediction, behavior_labels,video_idx]
            )
        if cfg.NUM_GPUS:
            all_behavior_prediction = all_behavior_prediction.cpu()
            all_behavior_labels = all_behavior_labels.cpu()
            video_idx = video_idx.cpu()


        test_meter.iter_toc()

     
        test_meter.update_stats(
            all_animal_prediction.detach(), all_animal_labels.detach(), video_idx.detach(), "animal"
        )

        test_meter.update_stats(
            all_behavior_prediction.detach(), all_behavior_labels.detach(), video_idx.detach(),"behavior"
        )

        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()







    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:

        all_preds_animal = test_meter.video_preds_animal.clone().detach()
        all_labels_animal = test_meter.video_labels_animal

        all_preds_behavior = test_meter.video_preds_behavior.clone().detach()
        all_labels_behavior = test_meter.video_labels_behavior



        if cfg.NUM_GPUS:
            all_preds_animal = all_preds_animal.cpu()
            all_labels_animal = all_labels_animal.cpu()

            all_preds_behavior = all_preds_behavior.cpu()
            all_labels_behavior = all_labels_behavior.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds_animal, labels=all_labels_animal)
            writer.plot_eval(preds=all_preds_behavior, labels=all_labels_behavior)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds_animal, all_labels_animal], f)
                    pickle.dump([all_preds_behavior, all_labels_behavior], f)
            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )




        # all_preds = test_meter.video_preds.clone().detach()
        # all_labels = test_meter.video_labels
        # if cfg.NUM_GPUS:
        #     all_preds = all_preds.cpu()
        #     all_labels = all_labels.cpu()
        # if writer is not None:
        #     writer.plot_eval(preds=all_preds, labels=all_labels)

        # if cfg.TEST.SAVE_RESULTS_PATH != "":
        #     save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

        #     if du.is_root_proc():
        #         with pathmgr.open(save_path, "wb") as f:
        #             pickle.dump([all_preds, all_labels], f)

        #     logger.info(
        #         "Successfully saved prediction results to {}".format(save_path)
        #     )






    # test_meter.finalize_metrics()
    test_meter.finalize_metrics()


    test_meter.finalize_metrics_per_class()
    # test_meter.finalize_metrics_long_tail_perexample()

    test_meter.finalize_metrics_long_tail()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
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

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg)
        out_str_prefix = "lin" if cfg.MODEL.DETACH_FINAL_FC else ""

        flops, params = 0.0, 0.0
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info(
                model, cfg, use_train_input=False
            )

        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)
        if (
            cfg.TASK == "ssl"
            and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            and cfg.CONTRASTIVE.KNN_ON
        ):
            train_loader = loader_composition.construct_loader(cfg, "train")
            if hasattr(model, "module"):
                model.module.init_knn_labels(train_loader)
            else:
                model.init_knn_labels(train_loader)

        cu.load_test_checkpoint(cfg, model)

        # Create video testing loaders.
        test_loader = loader_composition.construct_loader(cfg, "test")
        
        
        logger.info("Testing model for {} iterations".format(len(test_loader)))

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        else:
            assert (
                test_loader.dataset.num_videos
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
            )
            # Create meters for multi-view testing.
            # test_meter = TestMeter(
            #     test_loader.dataset.num_videos
            #     // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            #     cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            #     cfg.MODEL.NUM_CLASSES
            #     if not cfg.TASK == "ssl"
            #     else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
            #     len(test_loader),
            #     cfg.DATA.MULTI_LABEL,
            #     cfg.DATA.ENSEMBLE_METHOD,
            # )
            
            


            # print("number of videos ", test_loader.dataset.num_videos)
            # print("ensemble videos",cfg.TEST.NUM_ENSEMBLE_VIEWS)
            # print("crops",cfg.TEST.NUM_SPATIAL_CROPS)
            # print("result anchor",test_loader.dataset.num_videos
            #     // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS))
            
            
            # assert False
            test_meter = TestMeter(
                test_loader.dataset.num_videos
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),

                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,

                cfg.MODEL.NUM_CLASSES_ANIMAL,

                cfg.MODEL.NUM_CLASSES_BEHAVIOR,

                cfg.DATA.PATH_TO_DATA_DIR,
                cfg.OUTPUT_DIR,

                len(test_loader),

                cfg.DATA.MULTI_LABEL,

                cfg.DATA.ENSEMBLE_METHOD,
            )




        # Set up writer for logging to Tensorboard format.
        if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
        ):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None

        # # Perform multi-view test on the entire dataset.
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)

        test_meters.append(test_meter)
        if writer is not None:
            writer.close()









    result_string = (
        "_a{}{} Animal Head Top1 Acc: {} Head Top5 Acc: {}   Med Top1 Acc: {} Med Top5 Acc: {} Tail Top1 Acc: {} Tail Top5 Acc: {}  MEM: {:.2f} dataset: {}{}"
        "".format(
            out_str_prefix,
            cfg.TEST.DATASET[0],
            # test_meter.stats["perclass_animal_top1_acc"],
            test_meter.stats["perclass_head_animal_top1_acc"],
            test_meter.stats["perclass_head_animal_top5_acc"],
            test_meter.stats["perclass_med_animal_top1_acc"],
            test_meter.stats["perclass_med_animal_top5_acc"],
            test_meter.stats["perclass_tail_animal_top1_acc"],
            test_meter.stats["perclass_tail_animal_top5_acc"],
            misc.gpu_mem_usage(),
            cfg.TEST.DATASET[0],
            cfg.MODEL.NUM_CLASSES,
        )
    )
    logger.info("Per-class testing done: {}".format(result_string))
    # print(result_string)




    result_string = (
        "_a{}{} Behavior Head Top1 Acc: {} Head Top5 Acc: {}   Med Top1 Acc: {} Med Top5 Acc: {} Tail Top1 Acc: {} Tail Top5 Acc: {}  MEM: {:.2f} dataset: {}{}"
        "".format(
            out_str_prefix,
            cfg.TEST.DATASET[0],
            # test_meter.stats["perclass_animal_top1_acc"],
            test_meter.stats["perclass_head_behavior_top1_acc"],
            test_meter.stats["perclass_head_behavior_top5_acc"],
            test_meter.stats["perclass_med_behavior_top1_acc"],
            test_meter.stats["perclass_med_behavior_top5_acc"],
            test_meter.stats["perclass_tail_behavior_top1_acc"],
            test_meter.stats["perclass_tail_behavior_top5_acc"],
            misc.gpu_mem_usage(),
            cfg.TEST.DATASET[0],
            cfg.MODEL.NUM_CLASSES,
        )
    )
    logger.info("Per-class testing done: {}".format(result_string))
    # print(result_string)



    result_string = (
        "_a{}{} Composition Head Top1 Acc: {} Head Top5 Acc: {}   Med Top1 Acc: {} Med Top5 Acc: {} Tail Top1 Acc: {} Tail Top5 Acc: {}  MEM: {:.2f} dataset: {}{}"
        "".format(
            out_str_prefix,
            cfg.TEST.DATASET[0],
            # test_meter.stats["perclass_animal_top1_acc"],
            test_meter.stats["perclass_head_composition_top1_acc"],
            test_meter.stats["perclass_head_composition_top5_acc"],
            test_meter.stats["perclass_med_composition_top1_acc"],
            test_meter.stats["perclass_med_composition_top5_acc"],
            test_meter.stats["perclass_tail_composition_top1_acc"],
            test_meter.stats["perclass_tail_composition_top5_acc"],
            misc.gpu_mem_usage(),
            cfg.TEST.DATASET[0],
            cfg.MODEL.NUM_CLASSES,
        )
    )
    logger.info("Per-class testing done: {}".format(result_string))




    return result_string


    # result_string_views = "_p{:.2f}_f{:.2f}".format(params / 1e6, flops)

    # for view, test_meter in zip(cfg.TEST.NUM_TEMPORAL_CLIPS, test_meters):
    #     logger.info(
    #         "Finalized testing with {} temporal clips and {} spatial crops".format(
    #             view, cfg.TEST.NUM_SPATIAL_CROPS
    #         )
    #     )
    #     result_string_views += "_{}a{}" "".format(
    #         view, test_meter.stats["top1_acc"]
    #     )

    #     result_string = (
    #         "_p{:.2f}_f{:.2f}_{}a{} Top5 Acc: {} MEM: {:.2f} f: {:.4f}"
    #         "".format(
    #             params / 1e6,
    #             flops,
    #             view,
    #             test_meter.stats["top1_acc"],
    #             test_meter.stats["top5_acc"],
    #             misc.gpu_mem_usage(),
    #             flops,
    #         )
    #     )

    #     logger.info("{}".format(result_string))
    # logger.info("{}".format(result_string_views))
    # return result_string + " \n " + result_string_views
