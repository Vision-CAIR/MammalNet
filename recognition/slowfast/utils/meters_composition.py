#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
from socket import ALG_SET_PUBKEY
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import slowfast.datasets.ava_helper as ava_helper
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.utils.ava_eval_helper import (
    evaluate_ava,
    read_csv,
    read_exclusions,
    read_labelmap,
)

logger = logging.get_logger(__name__)
import pickle as pkl



def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    Args:
        full_groundtruth(dict): list of groundtruth.
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):
        for key in full_groundtruth[i].keys():
            if int(key.split(",")[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret


class AVAMeter(object):
    """
    Measure the AVA train, val, and test stats.
    """

    def __init__(self, overall_iters, cfg, mode):
        """
        overall_iters (int): the overall number of iterations of one epoch.
        cfg (CfgNode): configs.
        mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.lr = None
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.full_ava_test = cfg.AVA.FULL_TEST_ON_VAL
        self.mode = mode
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []
        self.overall_iters = overall_iters
        self.excluded_keys = read_exclusions(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
        )
        self.categories, self.class_whitelist = read_labelmap(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE)
        )
        gt_filename = os.path.join(
            cfg.AVA.ANNOTATION_DIR, cfg.AVA.GROUNDTRUTH_FILE
        )
        self.full_groundtruth = read_csv(gt_filename, self.class_whitelist)
        self.mini_groundtruth = get_ava_mini_groundtruth(self.full_groundtruth)

        _, self.video_idx_to_name = ava_helper.load_image_lists(
            cfg, mode == "train"
        )
        self.output_dir = cfg.OUTPUT_DIR

        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        self.stats = {}
        self.stats["top1_acc"] = 100.0
        self.stats["top5_acc"] = 100.0

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats.
        Args:
            cur_epoch (int): the current epoch.
            cur_iter (int): the current iteration.
        """

        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return

        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        if self.mode == "train":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}/{}".format(
                    cur_epoch + 1, self.cfg.SOLVER.MAX_EPOCH
                ),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
                "loss": self.loss.get_win_median(),
                "lr": self.lr,
            }
        elif self.mode == "val":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}/{}".format(
                    cur_epoch + 1, self.cfg.SOLVER.MAX_EPOCH
                ),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        elif self.mode == "test":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()

        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []

    def update_stats(self, preds, ori_boxes, metadata, loss=None, lr=None):
        """
        Update the current stats.
        Args:
            preds (tensor): prediction embedding.
            ori_boxes (tensor): original boxes (x1, y1, x2, y2).
            metadata (tensor): metadata of the AVA data.
            loss (float): loss value.
            lr (float): learning rate.
        """
        if self.mode in ["val", "test"]:
            self.all_preds.append(preds)
            self.all_ori_boxes.append(ori_boxes)
            self.all_metadata.append(metadata)
        if loss is not None:
            self.loss.add_value(loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, log=True):
        """
        Calculate and log the final AVA metrics.
        """
        all_preds = torch.cat(self.all_preds, dim=0)
        all_ori_boxes = torch.cat(self.all_ori_boxes, dim=0)
        all_metadata = torch.cat(self.all_metadata, dim=0)

        if self.mode == "test" or (self.full_ava_test and self.mode == "val"):
            groundtruth = self.full_groundtruth
        else:
            groundtruth = self.mini_groundtruth

        self.full_map = evaluate_ava(
            all_preds,
            all_ori_boxes,
            all_metadata.tolist(),
            self.excluded_keys,
            self.class_whitelist,
            self.categories,
            groundtruth=groundtruth,
            video_idx_to_name=self.video_idx_to_name,
        )
        if log:
            stats = {"mode": self.mode, "map": self.full_map}
            logging.log_json_stats(stats, self.output_dir)

        map_str = "{:.{prec}f}".format(self.full_map * 100.0, prec=2)

        self.min_top1_err = self.full_map
        self.stats["top1_acc"] = map_str
        self.stats["top5_acc"] = map_str

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.mode in ["val", "test"]:
            self.finalize_metrics(log=False)
            stats = {
                "_type": "{}_epoch".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "mode": self.mode,
                "map": self.full_map,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
                "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            }
            logging.log_json_stats(stats, self.output_dir)


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls_animal,
        num_cls_behavior,
        file_path,
        output_dir,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        self.data_file_path = file_path
        self.output_dir = output_dir


        # Initialize tensors.
        self.video_preds_animal = torch.zeros((num_videos, num_cls_animal))
        self.video_preds_behavior = torch.zeros((num_videos, num_cls_behavior))
        if multi_label:
            self.video_preds_animal -= 1e10
            self.video_preds_behavior -= 1e10


        self.video_labels_animal = (
            torch.zeros((num_videos, num_cls_animal))
            if multi_label
            else torch.zeros((num_videos)).long()
        )


        self.clip_count_animal = torch.zeros((num_videos)).long()
        self.topk_accs_animal = []
        self.stats_animal = {}


        self.video_labels_behavior = (
            torch.zeros((num_videos, num_cls_behavior))
            if multi_label
            else torch.zeros((num_videos)).long()
        )
        self.clip_count_behavior = torch.zeros((num_videos)).long()
        self.topk_accs_behavior = []
        self.stats_behavior = {}


        self.clip_count_composition = torch.zeros((num_videos)).long()
        self.topk_accs_composition = []
        self.stats_composition = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count_animal.zero_()
        self.clip_count_behavior.zero_()
        self.clip_count_composition.zero_()
        self.video_preds_animal.zero_()
        self.video_preds_behavior.zero_()
        
        if self.multi_label:
            self.video_preds_animal -= 1e10
            self.video_labels_behavior -= 1e10
        self.video_labels_animal.zero_()
        self.video_labels_behavior.zero_()

    def update_stats(self, preds, labels, clip_ids,data_type):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """

        for ind in range(preds.shape[0]):

            if data_type =="animal":
                vid_id = int(clip_ids[ind]) // self.num_clips

                if self.video_labels_animal[vid_id].sum() > 0:
                    assert torch.equal(
                        self.video_labels_animal[vid_id].type(torch.FloatTensor),
                        labels[ind].type(torch.FloatTensor),
                    )

                self.video_labels_animal[vid_id] = labels[ind]
                if self.ensemble_method == "sum":
                    self.video_preds_animal[vid_id] += preds[ind]
                elif self.ensemble_method == "max":
                    self.video_preds[vid_id] = torch.max(
                        self.video_preds_animal[vid_id], preds[ind]
                    )
                else:
                    raise NotImplementedError(
                        "Ensemble Method {} is not supported".format(
                            self.ensemble_method
                        )
                    )
                self.clip_count_animal[vid_id] += 1
            elif data_type =="behavior":
                vid_id = int(clip_ids[ind]) // self.num_clips
                if self.video_labels_behavior[vid_id].sum() > 0:
                    assert torch.equal(
                        self.video_labels_behavior[vid_id].type(torch.FloatTensor),
                        labels[ind].type(torch.FloatTensor),
                    )
                self.video_labels_behavior[vid_id] = labels[ind]
                if self.ensemble_method == "sum":
                    self.video_preds_behavior[vid_id] += preds[ind]
                elif self.ensemble_method == "max":
                    self.video_preds_behavior[vid_id] = torch.max(
                        self.video_preds_behavior[vid_id], preds[ind]
                    )
                else:
                    raise NotImplementedError(
                        "Ensemble Method {} is not supported".format(
                            self.ensemble_method
                        )
                    )
                self.clip_count_behavior[vid_id] += 1           

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):

        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """

        self.stats = {"split": "test_final"}


        if self.multi_label:
            mean_ap = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            map_str = "{:.{prec}f}".format(mean_ap * 100.0, prec=2)
            self.stats["map"] = map_str
            self.stats["top1_acc"] = map_str
            self.stats["top5_acc"] = map_str
        else:

            num_topks_correct_animal,num_topks_correct_behavior,num_topks_correct_composition= \
            metrics.topks_correct_composition(self.video_preds_animal, self.video_labels_animal, \
            self.video_preds_behavior, self.video_labels_behavior, ks)



            topks_animal = [
                (x / self.video_preds_animal.size(0)) * 100.0
                for x in num_topks_correct_animal
            ]
            assert len({len(ks), len(topks_animal)}) == 1
            for k, topk in zip(ks, topks_animal):
                self.stats["animal_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
            

            topks_behavior = [
                (x / self.video_preds_behavior.size(0)) * 100.0
                for x in num_topks_correct_behavior
            ]
            assert len({len(ks), len(topks_behavior)}) == 1
            for k, topk in zip(ks, topks_behavior):
                self.stats["behavior_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )


            topks_composition = [
                (x / self.video_preds_animal.size(0)) * 100.0
                for x in num_topks_correct_composition
            ]
            assert len({len(ks), len(topks_composition)}) == 1
            for k, topk in zip(ks, topks_composition):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )


        logging.log_json_stats(self.stats)



    def finalize_metrics_per_class(self, ks=(1, 5)):

        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
    

        self.stats = {"split": "test_final"}


        if self.multi_label:
            mean_ap = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            map_str = "{:.{prec}f}".format(mean_ap * 100.0, prec=2)
            self.stats["map"] = map_str
            self.stats["top1_acc"] = map_str
            self.stats["top5_acc"] = map_str
        else:

            

            unique_animal_labels = torch.unique(self.video_labels_animal)
            unique_behavior_labels = torch.unique(self.video_labels_behavior)

           

            animal_perclass_acc = torch.Tensor([0,0])
            behavior_perclass_acc = torch.Tensor([0,0])
            composition_perclass_acc = torch.Tensor([0,0])


            for animal in unique_animal_labels:
                label_indexs = (self.video_labels_animal==animal).nonzero(as_tuple=True)[0]
                
                sub_label = torch.index_select(self.video_labels_animal,0,label_indexs)
                sub_pred = torch.index_select(self.video_preds_animal,0, label_indexs)

                num_topks_correct_animal=metrics.topks_correct(sub_pred,sub_label,ks)

                topks_animal = [
                    (x / sub_pred.size(0)) * 100.0
                    for x in num_topks_correct_animal
                ]
                animal_perclass_acc+=torch.Tensor(topks_animal)
            animal_perclass_acc /= len(unique_animal_labels)
            animal_perclass_acc = animal_perclass_acc.tolist()
           
            
            
            assert len({len(ks), len(animal_perclass_acc)}) == 1
            for k, topk in zip(ks, animal_perclass_acc):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_animal_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )


            '''
            behavior per classs analysis
            '''


            for behavior in unique_behavior_labels:
                label_indexs = (self.video_labels_behavior==behavior).nonzero(as_tuple=True)[0]
                
                sub_label = torch.index_select(self.video_labels_behavior,0,label_indexs)
                sub_pred = torch.index_select(self.video_preds_behavior,0, label_indexs)

                num_topks_correct_behavior=metrics.topks_correct(sub_pred,sub_label,ks)

                topks_behavior = [
                    (x / sub_pred.size(0)) * 100.0
                    for x in num_topks_correct_behavior
                ]
                # print(topks_behavior)
                behavior_perclass_acc+=torch.Tensor(topks_behavior)
            behavior_perclass_acc /= len(unique_behavior_labels)
            behavior_perclass_acc = behavior_perclass_acc.tolist()


            
            
            assert len({len(ks), len(behavior_perclass_acc)}) == 1
            for k, topk in zip(ks, behavior_perclass_acc):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_behavior_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )



            '''
            compositional evaluation
            '''
            ### compositional evaluate results


            concat_labels = torch.cat([self.video_labels_animal.view(1,-1),self.video_labels_behavior.view(1,-1)],0).t()
            unique_composition_labels = concat_labels.unique(dim=0)


            for label in unique_composition_labels:
                composition_index = ((concat_labels==label).sum(1)==len(label)).nonzero(as_tuple = True)[0]

                sub_animal_pred = torch.index_select(self.video_preds_animal,0,composition_index)
                sub_animal_label = torch.index_select(self.video_labels_animal,0,composition_index)
                sub_behavior_pred = torch.index_select(self.video_preds_behavior,0,composition_index)
                sub_behavior_label = torch.index_select(self.video_labels_behavior,0,composition_index)



                num_topks_animal,num_toks_behavior,num_topks_correct_composition= \
                metrics.topks_correct_composition(sub_animal_pred, sub_animal_label, \
                sub_behavior_pred, sub_behavior_label, ks)
                

                topks_composition = [
                    (x / sub_animal_pred.size(0)) * 100.0
                    for x in num_topks_correct_composition
                ]

                composition_perclass_acc+=torch.Tensor(topks_composition)
            composition_perclass_acc /= len(unique_composition_labels)
            composition_perclass_acc = composition_perclass_acc.tolist()


            assert len({len(ks), len(composition_perclass_acc)}) == 1
            for k, topk in zip(ks, composition_perclass_acc):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
            

        logging.log_json_stats(self.stats)





    def classify_longtail_classes(self,file):
        animal_head_classes = []
        animal_med_classes = []
        animal_tail_classes = []

        behavior_head_classes = []
        behavior_med_classes = []
        behavior_tail_classes = []

        composition_head_classes = []
        composition_med_classes = []
        composition_tail_classes = []

        ah_bh = []
        am_bh = []
        af_bh = []
        ah_bm = []
        am_bm = []
        af_bm = []
        ah_bf = []
        am_bf = []
        af_bf = []

        animal_dict = dict()
        behavior_dict = dict()
        composition_dict = dict()
            
        # print(self.data_file_path)
        with open(self.data_file_path+file,"r") as f:
            for line in f.readlines():
                behavior = line.split()[1]
                animal = line.split()[2]
                try:
                    animal_dict[animal] +=1
                except:
                    animal_dict[animal] = 1
                
                try:
                    behavior_dict[behavior] +=1
                except:
                    behavior_dict[behavior] = 1
                
                animal_behavior = animal+"_"+behavior
                try:
                    composition_dict[animal_behavior] +=1
                except:
                    composition_dict[animal_behavior] =1
        

        for key in animal_dict.keys():
            animal_id = key
            frequency = animal_dict[key]
            if frequency > 300:
                animal_head_classes.append(animal_id)
            elif frequency>100:
                animal_med_classes.append(animal_id)
            else:
                animal_tail_classes.append(animal_id)


        for key in behavior_dict.keys():
            behavior_id = key
            frequency = behavior_dict[key]
            if frequency > 1500:
                behavior_head_classes.append(behavior_id)
            elif frequency>500:
                behavior_med_classes.append(behavior_id)
            else:
                behavior_tail_classes.append(behavior_id)


        # split on composition
        for key in composition_dict.keys():
            composition_id = key
            frequency = composition_dict[key]
            if frequency > 70:
                composition_head_classes.append(composition_id)
            elif frequency>20:
                composition_med_classes.append(composition_id)
            else:
                composition_tail_classes.append(composition_id)
        
        # split based on metrix evaluation

        for key in composition_dict.keys():
            animal,behavior = key.split("_")[0], key.split("_")[1]
            animal_frequency = animal_dict[animal]
            behavior_frequency = behavior_dict[behavior]

            if animal_frequency > 300 and behavior_frequency > 1500:
                ah_bh.append(key)
            elif animal_frequency <= 300 and animal_frequency >100 and behavior_frequency > 1500:
                am_bh.append(key)
            elif animal_frequency <= 100 and behavior_frequency > 1500:
                af_bh.append(key)
            elif animal_frequency > 300 and behavior_frequency > 500:
                ah_bm.append(key)
            elif animal_frequency <=300 and animal_frequency > 100 and behavior_frequency >500:
                am_bm.append(key)
            elif animal_frequency <=100 and behavior_frequency >500:
                af_bm.append(key)
            elif animal_frequency >300 and behavior_frequency <=500:
                ah_bf.append(key)
            elif animal_frequency >100 and animal_frequency <=300 and behavior_frequency <=500:
                am_bf.append(key)
            elif animal_frequency <=100 and behavior_frequency <=500:
                af_bf.append(key)
            


        return animal_head_classes, animal_med_classes, animal_tail_classes,\
            behavior_head_classes, behavior_med_classes, behavior_tail_classes, \
                composition_head_classes, composition_med_classes, composition_tail_classes, \
                    ah_bh, am_bh, af_bh,ah_bm, am_bm, af_bm, ah_bf, am_bf, af_bf



    def compute_frequency(self,file):

        animal_dict = dict()
        behavior_dict = dict()
        composition_dict = dict()
            
        with open(self.data_file_path+file,"r") as f:
            for line in f.readlines():
                behavior = line.split()[1]
                animal = line.split()[2]
                try:
                    animal_dict[animal] +=1
                except:
                    animal_dict[animal] = 1
                
                try:
                    behavior_dict[behavior] +=1
                except:
                    behavior_dict[behavior] = 1
                
                animal_behavior = animal+"_"+behavior
                try:
                    composition_dict[animal_behavior] +=1
                except:
                    composition_dict[animal_behavior] =1
        



        return animal_dict, behavior_dict, composition_dict

    def finalize_metrics_group_by_counterpart(self, ks=(1, 5)):

        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
    

        self.stats = {"split": "test_final"}


        if self.multi_label:
            mean_ap = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            map_str = "{:.{prec}f}".format(mean_ap * 100.0, prec=2)
            self.stats["map"] = map_str
            self.stats["top1_acc"] = map_str
            self.stats["top5_acc"] = map_str
        else:

            

            unique_animal_labels = torch.unique(self.video_labels_animal)
            unique_behavior_labels = torch.unique(self.video_labels_behavior)

           

            animal_perclass_acc = torch.Tensor([0,0])
            behavior_perclass_acc = torch.Tensor([0,0])
            composition_perclass_acc = torch.Tensor([0,0])



            animal_dict = {}
            behavior_dict ={}

            animal_per_behavior_accuracy =[]

            for animal in unique_animal_labels:
                label_indexs = (self.video_labels_animal==animal).nonzero(as_tuple=True)[0]
                
                sub_label = torch.index_select(self.video_labels_animal,0,label_indexs)
                sub_pred = torch.index_select(self.video_preds_animal,0, label_indexs)
                sub_behavior_gt = torch.index_select(self.video_labels_behavior,0,label_indexs)


                for index, behavior in enumerate(sub_behavior_gt):
                    
                    behavior = behavior.item()
                    if behavior not in behavior_dict.keys():
                        behavior_dict[behavior]= [[sub_pred[index].tolist()], [sub_label[index].tolist()]]
                    else:
                        behavior_dict[behavior][0].append(sub_pred[index].tolist())
                        behavior_dict[behavior][1].append(sub_label[index].tolist())
            

            
            for key in behavior_dict.keys():
                sub_pred = behavior_dict[key][0]
                sub_label = behavior_dict[key][1]

                sub_pred = torch.tensor(sub_pred)
                sub_label = torch.tensor(sub_label)
            

                num_topks_correct_animal=metrics.topks_correct(sub_pred,sub_label,ks)

                topks_animal = [
                    (x / sub_pred.size(0)).item() * 100.0
                    for x in num_topks_correct_animal
                ]

                animal_per_behavior_accuracy.append([key,topks_animal])

            print("animal per behavior accuracy",animal_per_behavior_accuracy)


            # animal_perclass_acc /= len(unique_animal_labels)
            # animal_perclass_acc = animal_perclass_acc.tolist()
           
            
            
            # assert len({len(ks), len(animal_perclass_acc)}) == 1
            # for k, topk in zip(ks, animal_perclass_acc):
            #     # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
            #     self.stats["perclass_animal_top{}_acc".format(k)] = "{:.{prec}f}".format(
            #         topk, prec=2
            #     )






            '''
            behavior per classs analysis
            '''

            
            behavior_per_animal_accuracy=[]
            for behavior in unique_behavior_labels:
                label_indexs = (self.video_labels_behavior==behavior).nonzero(as_tuple=True)[0]
                
                sub_label = torch.index_select(self.video_labels_behavior,0,label_indexs)
                sub_pred = torch.index_select(self.video_preds_behavior,0, label_indexs)

                sub_animal_gt = torch.index_select(self.video_labels_animal,0,label_indexs)


                for index, animal in enumerate(sub_animal_gt):
                    animal = animal.item()
                    if animal not in animal_dict.keys():
                        animal_dict[animal] = [[sub_pred[index].tolist()],[sub_label[index].tolist()]]
                    else:
                        animal_dict[animal][0].append(sub_pred[index].tolist())
                        animal_dict[animal][1].append(sub_label[index].tolist())
                    
                


                for key in animal_dict.keys():
                    
                    sub_pred = animal_dict[key][0]
                    sub_label = animal_dict[key][1]

                    sub_pred = torch.tensor(sub_pred)
                    sub_label = torch.tensor(sub_label)


                    num_topks_correct_behavior = metrics.topks_correct(sub_pred,sub_label,ks)

                    topks_behavior = [
                        (x / sub_pred.size(0)).item() * 100.0
                        for x in num_topks_correct_behavior
                    ]

                    behavior_per_animal_accuracy.append([key,topks_behavior])
                

            


            #     num_topks_correct_behavior=metrics.topks_correct(sub_pred,sub_label,ks)

            #     topks_behavior = [
            #         (x / sub_pred.size(0)) * 100.0
            #         for x in num_topks_correct_behavior
            #     ]
            #     # print(topks_behavior)
            #     behavior_perclass_acc+=torch.Tensor(topks_behavior)
            # behavior_perclass_acc /= len(unique_behavior_labels)
            # behavior_perclass_acc = behavior_perclass_acc.tolist()


            

        logging.log_json_stats(self.stats)


    def finalize_metrics_long_tail(self, ks=(1, 5)):

        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        # clip_check = self.clip_count == self.num_clips
        # if not all(clip_check):
        #     logger.warning(
        #         "clip count Ids={} = {} (should be {})".format(
        #             np.argwhere(~clip_check),
        #             self.clip_count[~clip_check],
        #             self.num_clips,
        #         )
        #     )

        self.stats = {"split": "test_final"}


        if self.multi_label:
            mean_ap = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            map_str = "{:.{prec}f}".format(mean_ap * 100.0, prec=2)
            self.stats["map"] = map_str
            self.stats["top1_acc"] = map_str
            self.stats["top5_acc"] = map_str
        else:

        
            unique_animal_labels = torch.unique(self.video_labels_animal)
            unique_behavior_labels = torch.unique(self.video_labels_behavior)

           
            animal_head, animal_med, animal_tail,behavior_head, behavior_med, \
            behavior_tail, composition_head, composition_med, composition_tail, \
            ah_bh, am_bh, af_bh,ah_bm, am_bm, af_bm, ah_bf, am_bf, af_bf = \
                 self.classify_longtail_classes("/train.csv")

                
            animal_frequency, behavior_frequency, composition_frequency = self.compute_frequency("/train.csv")



            animal_perclass_acc_head = torch.Tensor([0,0])
            animal_perclass_acc_med = torch.Tensor([0,0])
            animal_perclass_acc_tail = torch.Tensor([0,0])
            number_animal_head = number_animal_med = number_animal_tail = 0

            behavior_perclass_acc_head = torch.Tensor([0,0])
            behavior_perclass_acc_med = torch.Tensor([0,0])
            behavior_perclass_acc_tail = torch.Tensor([0,0])
            number_behavior_head = number_behavior_med = number_behavior_tail = 0


            composition_perclass_acc_head = torch.Tensor([0,0])
            composition_perclass_acc_med = torch.Tensor([0,0])
            composition_perclass_acc_tail = torch.Tensor([0,0])
            number_composition_head = number_composition_med = number_composition_tail = 0


            ah_bh_composition = torch.Tensor([0,0])
            ah_bm_composition = torch.Tensor([0,0])
            ah_bf_composition = torch.Tensor([0,0])
            am_bh_composition = torch.Tensor([0,0])
            am_bm_composition = torch.Tensor([0,0])
            am_bf_composition = torch.Tensor([0,0])
            af_bh_composition = torch.Tensor([0,0])
            af_bm_composition = torch.Tensor([0,0])
            af_bf_composition = torch.Tensor([0,0])
            number_ah_bh = number_ah_bm = number_ah_bf = number_am_bh = \
                number_am_bm = number_am_bf = number_af_bh = number_af_bm = number_af_bf = 0


            unique_animal_result = []

            with open(self.output_dir+"/animal_prediction.txt","w") as f:
                for animal in unique_animal_labels:
                    label_indexs = (self.video_labels_animal==animal).nonzero(as_tuple=True)[0]
                    
                    sub_label = torch.index_select(self.video_labels_animal,0,label_indexs)
                    sub_pred = torch.index_select(self.video_preds_animal,0, label_indexs)
                
                    num_topks_correct_animal,pred_index, gt_index=metrics.topks_correct(sub_pred,sub_label,ks,return_pred_index=True)


                    _,pred_index_top1, gt_index_top1 =metrics.topks_correct(sub_pred,sub_label,[1],return_pred_index=True)
                    for each_pred, each_label in zip(pred_index_top1, gt_index_top1):
 
                        if each_pred.shape[0] >1:
                            for each_sub_pred, each_sub_label in zip(each_pred, each_label):
                                f.write(str(each_sub_pred.item())+" "+str(each_sub_label.item())+"\n")
                        else:
                            f.write(str(each_pred.item())+" "+str(each_label.item())+"\n")
                        



                    topks_animal = [
                        (x / sub_pred.size(0)) * 100.0
                        for x in num_topks_correct_animal
                    ]

                    if str(torch.unique(sub_label).item()) in animal_head:
                        animal_perclass_acc_head+=torch.Tensor(topks_animal)
                        number_animal_head+=1
                    elif str(torch.unique(sub_label).item()) in animal_med:
                        animal_perclass_acc_med+=torch.Tensor(topks_animal)
                        number_animal_med+=1
                    elif str(torch.unique(sub_label).item()) in animal_tail:
                        animal_perclass_acc_tail+=torch.Tensor(topks_animal)
                        number_animal_tail+=1
                    else:
                        print(sub_label, "do not exist in the training dataset")
                    
                    unique_animal_result.append([animal.item(),topks_animal[0].item(),animal_frequency[str(animal.item())]])
                
            animal_perclass_acc_head /= number_animal_head
            animal_perclass_acc_med /= number_animal_med
            animal_perclass_acc_tail /= number_animal_tail

            animal_perclass_acc_head = animal_perclass_acc_head.tolist()
            animal_perclass_acc_med = animal_perclass_acc_med.tolist()
            animal_perclass_acc_tail = animal_perclass_acc_tail.tolist()




            assert len({len(ks), len(animal_perclass_acc_head)}) == 1
            for k, topk in zip(ks, animal_perclass_acc_head):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_head_animal_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )

            assert len({len(ks), len(animal_perclass_acc_med)}) == 1
            for k, topk in zip(ks, animal_perclass_acc_med):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_med_animal_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )

            assert len({len(ks), len(animal_perclass_acc_tail)}) == 1
            for k, topk in zip(ks, animal_perclass_acc_tail):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_tail_animal_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )



            '''
            behavior per classs analysis
            '''

            unique_behavior_result=[]

            with open(self.output_dir+"/behavior_prediction.txt","w") as f:
                for behavior in unique_behavior_labels:
                    label_indexs = (self.video_labels_behavior==behavior).nonzero(as_tuple=True)[0]
                    sub_label = torch.index_select(self.video_labels_behavior,0,label_indexs)
                    sub_pred = torch.index_select(self.video_preds_behavior,0, label_indexs)

                    num_topks_correct_behavior,pred_index, gt_index = metrics.topks_correct(sub_pred,sub_label,ks,return_pred_index=True)



                    _,pred_index_top1, gt_index_top1 =metrics.topks_correct(sub_pred,sub_label,[1],return_pred_index=True)
                    for each_pred, each_label in zip(pred_index_top1, gt_index_top1):
 
                        if each_pred.shape[0] >1:
                            for each_sub_pred, each_sub_label in zip(each_pred, each_label):
                                f.write(str(each_sub_pred.item())+" "+str(each_sub_label.item())+"\n")
                        else:
                            f.write(str(each_pred.item())+" "+str(each_label.item())+"\n")


                    topks_behavior = [
                        (x / sub_pred.size(0)) * 100.0
                        for x in num_topks_correct_behavior
                    ]

                    # print("sub label",torch.unique(sub_label).item())
                    if str(torch.unique(sub_label).item()) in behavior_head:
                        behavior_perclass_acc_head+=torch.Tensor(topks_behavior)
                        number_behavior_head+=1
                    elif str(torch.unique(sub_label).item()) in behavior_med:
                        behavior_perclass_acc_med+=torch.Tensor(topks_behavior)
                        number_behavior_med+=1
                    elif str(torch.unique(sub_label).item()) in behavior_tail:
                        behavior_perclass_acc_tail+=torch.Tensor(topks_behavior)
                        number_behavior_tail+=1
                    else:
                        print("behavior ",str(torch.unique(sub_label).item(),"do not exist in training"))

                    unique_behavior_result.append([behavior.item(),topks_behavior[0].item(),behavior_frequency[str(behavior.item())]])



            # print(number_behavior_head,number_behavior_med, number_behavior_tail)
            behavior_perclass_acc_head /= number_behavior_head
            behavior_perclass_acc_med /= number_behavior_med
            behavior_perclass_acc_tail /= number_behavior_tail

            behavior_perclass_acc_head = behavior_perclass_acc_head.tolist()
            behavior_perclass_acc_med = behavior_perclass_acc_med.tolist()
            behavior_perclass_acc_tail = behavior_perclass_acc_tail.tolist()




            assert len({len(ks), len(behavior_perclass_acc_head)}) == 1
            for k, topk in zip(ks, behavior_perclass_acc_head):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_head_behavior_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )

            assert len({len(ks), len(behavior_perclass_acc_med)}) == 1
            for k, topk in zip(ks, behavior_perclass_acc_med):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_med_behavior_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )

            assert len({len(ks), len(behavior_perclass_acc_tail)}) == 1
            for k, topk in zip(ks, behavior_perclass_acc_tail):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_tail_behavior_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )













            '''
            compositional evaluation
            '''
            ### compositional evaluate results


            concat_labels = torch.cat([self.video_labels_animal.view(1,-1),self.video_labels_behavior.view(1,-1)],0).t()
            unique_composition_labels = concat_labels.unique(dim=0)

            unique_composition_result = []


            for label in unique_composition_labels:
                composition_index = ((concat_labels==label).sum(1)==len(label)).nonzero(as_tuple = True)[0]

                sub_animal_pred = torch.index_select(self.video_preds_animal,0,composition_index)
                sub_animal_label = torch.index_select(self.video_labels_animal,0,composition_index)
                sub_behavior_pred = torch.index_select(self.video_preds_behavior,0,composition_index)
                sub_behavior_label = torch.index_select(self.video_labels_behavior,0,composition_index)



                num_topks_animal,num_toks_behavior,num_topks_correct_composition= \
                metrics.topks_correct_composition(sub_animal_pred, sub_animal_label, \
                sub_behavior_pred, sub_behavior_label, ks)
                

                topks_composition = [
                    (x / sub_animal_pred.size(0)) * 100.0
                    for x in num_topks_correct_composition
                ]


                animal_behavior = str(torch.unique(sub_animal_label).item())+"_"+str(torch.unique(sub_behavior_label).item())

                if animal_behavior in composition_head:
                    composition_perclass_acc_head+=torch.Tensor(topks_composition)
                    number_composition_head+=1
                elif animal_behavior in composition_med:
                    composition_perclass_acc_med+=torch.Tensor(topks_composition)
                    number_composition_med+=1
                elif animal_behavior in composition_tail:
                    composition_perclass_acc_tail+=torch.Tensor(topks_composition)
                    number_composition_tail+=1
                else:
                    print("composition not in head med tail",animal_behavior)

                label = str(int(label[0].item()))+"_"+str(int(label[1].item()))
                if label not in composition_frequency:
                    unique_composition_result.append([label,topks_composition[0].item(),0])
                else:
                    unique_composition_result.append([label,topks_composition[0].item(),composition_frequency[label]])





            composition_perclass_acc_head /= number_composition_head
            composition_perclass_acc_med /= number_composition_med
            composition_perclass_acc_tail /= number_composition_tail

            composition_perclass_acc_head = composition_perclass_acc_head.tolist()
            composition_perclass_acc_med = composition_perclass_acc_med.tolist()
            composition_perclass_acc_tail = composition_perclass_acc_tail.tolist()


            assert len({len(ks), len(composition_perclass_acc_head)}) == 1
            for k, topk in zip(ks, composition_perclass_acc_head):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_head_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )

            assert len({len(ks), len(composition_perclass_acc_med)}) == 1
            for k, topk in zip(ks, composition_perclass_acc_med):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_med_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )

            assert len({len(ks), len(composition_perclass_acc_tail)}) == 1
            for k, topk in zip(ks, composition_perclass_acc_tail):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["perclass_tail_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
            





























            '''
            compositional metrix evaluation
            '''
            ### compositional evaluate results


            concat_labels = torch.cat([self.video_labels_animal.view(1,-1),self.video_labels_behavior.view(1,-1)],0).t()
            unique_composition_labels = concat_labels.unique(dim=0)


            # print("concat label", concat_labels)
            # print("unique composition labels", unique_composition_labels)
            # print("video labels shape", concat_labels.shape)
            # print("unique composition shape",unique_composition_labels.shape)

            for label in unique_composition_labels:
                # print(label)
                composition_index = ((concat_labels==label).sum(1)==len(label)).nonzero(as_tuple = True)[0]

                sub_animal_pred = torch.index_select(self.video_preds_animal,0,composition_index)
                sub_animal_label = torch.index_select(self.video_labels_animal,0,composition_index)
                sub_behavior_pred = torch.index_select(self.video_preds_behavior,0,composition_index)
                sub_behavior_label = torch.index_select(self.video_labels_behavior,0,composition_index)

                # print("sub aniaml label",sub_animal_label,sub_behavior_label)


                num_topks_animal,num_toks_behavior,num_topks_correct_composition= \
                metrics.topks_correct_composition(sub_animal_pred, sub_animal_label, \
                sub_behavior_pred, sub_behavior_label, ks)
                
                # print("topks correct composition", num_topks_animal, num_toks_behavior, num_topks_correct_composition)

                topks_composition = [
                    (x / sub_animal_pred.size(0)) * 100.0
                    for x in num_topks_correct_composition
                ]

                # composition_perclass_acc+=torch.Tensor(topks_composition)

                animal_behavior = str(torch.unique(sub_animal_label).item())+"_"+str(torch.unique(sub_behavior_label).item())

                if animal_behavior in ah_bh:
                    ah_bh_composition+=torch.Tensor(topks_composition)
                    number_ah_bh+=1
                elif animal_behavior in am_bh:
                    am_bh_composition+=torch.Tensor(topks_composition)
                    number_am_bh+=1
                elif animal_behavior in af_bh:
                    af_bh_composition+=torch.Tensor(topks_composition)
                    number_af_bh+=1
                elif animal_behavior in ah_bm:
                    ah_bm_composition+=torch.Tensor(topks_composition)
                    number_ah_bm+=1
                elif animal_behavior in am_bm:
                    am_bm_composition += torch.Tensor(topks_composition)
                    number_am_bm +=1
                elif animal_behavior in af_bm :
                    af_bm_composition += torch.Tensor(topks_composition)
                    number_af_bm +=1
                elif animal_behavior in ah_bf:
                    ah_bf_composition += torch.Tensor(topks_composition)
                    number_ah_bf +=1
                elif animal_behavior in am_bf:
                    am_bf_composition += torch.Tensor(topks_composition)
                    number_am_bf +=1
                elif animal_behavior in af_bf:
                    af_bf_composition += torch.Tensor(topks_composition)
                    number_af_bf +=1



            print(number_ah_bh,number_am_bh, number_af_bh,number_ah_bm,number_am_bm,number_af_bm,number_ah_bf,number_am_bf,number_af_bf)
            ah_bh_composition /= number_ah_bh
            am_bh_composition /= number_am_bh
            af_bh_composition /= number_af_bh
            ah_bm_composition /= number_ah_bm
            am_bm_composition /= number_am_bm
            af_bm_composition /= number_af_bm
            ah_bf_composition /= number_ah_bf
            am_bf_composition /= number_am_bf
            af_bf_composition /= number_af_bf


            ah_bh_composition = ah_bh_composition.tolist()
            am_bh_composition = am_bh_composition.tolist()
            af_bh_composition = af_bh_composition.tolist()
            ah_bm_composition = ah_bm_composition.tolist()
            am_bm_composition = am_bm_composition.tolist()
            af_bm_composition = af_bm_composition.tolist()
            ah_bf_composition = ah_bf_composition.tolist()
            am_bf_composition = am_bf_composition.tolist()
            af_bf_composition = af_bf_composition.tolist()

            # print("animal per class",animal_perclass_acc_head,animal_perclass_acc_med,animal_perclass_acc_tail)
            # assert False


            # assert len({len(ks), len(ah_bh_composition)}) == 1
            # for k, topk in zip(ks, ah_bh_composition):
            #     # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
            #     self.stats["ah_bh_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
            #         topk, prec=2
            #     )

            # assert len({len(ks), len(am_bh_composition)}) == 1
            # for k, topk in zip(ks, am_bh_composition):
            #     # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
            #     self.stats["am_bh_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
            #         topk, prec=2
            #     )

            # assert len({len(ks), len(af_bh_composition)}) == 1
            # for k, topk in zip(ks, af_bh_composition):
            #     # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
            #     self.stats["af_bh_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
            #         topk, prec=2
            #     )

            # assert len({len(ks), len(ah_bm_composition)}) == 1
            # for k, topk in zip(ks, ah_bm_composition):
            #     # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
            #     self.stats["ah_bm_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
            #         topk, prec=2
            #     )

            # assert len({len(ks), len(am_bm_composition)}) == 1
            # for k, topk in zip(ks, am_bm_composition):
            #     # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
            #     self.stats["am_bm_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
            #         topk, prec=2
            #     )

            # assert len({len(ks), len(af_bm_composition)}) == 1
            # for k, topk in zip(ks, af_bm_composition):
            #     # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
            #     self.stats["af_bm_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
            #         topk, prec=2
            #     )

            # assert len({len(ks), len(ah_bf_composition)}) == 1
            # for k, topk in zip(ks, ah_bf_composition):
            #     # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
            #     self.stats["ah_bf_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
            #         topk, prec=2
            #     )

            # assert len({len(ks), len(am_bf_composition)}) == 1
            # for k, topk in zip(ks, am_bf_composition):
            #     # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
            #     self.stats["am_bf_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
            #         topk, prec=2
            #     )

            # assert len({len(ks), len(af_bf_composition)}) == 1
            # for k, topk in zip(ks, af_bf_composition):
            #     # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
            #     self.stats["af_bf_composition_top{}_acc".format(k)] = "{:.{prec}f}".format(
            #         topk, prec=2
            #     )



        logging.log_json_stats(self.stats)


    def finalize_metrics_long_tail_perexample(self, ks=(1, 5)):

        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        # clip_check = self.clip_count == self.num_clips
        # if not all(clip_check):
        #     logger.warning(
        #         "clip count Ids={} = {} (should be {})".format(
        #             np.argwhere(~clip_check),
        #             self.clip_count[~clip_check],
        #             self.num_clips,
        #         )
        #     )

        self.stats = {"split": "test_final"}


        if self.multi_label:
            mean_ap = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            map_str = "{:.{prec}f}".format(mean_ap * 100.0, prec=2)
            self.stats["map"] = map_str
            self.stats["top1_acc"] = map_str
            self.stats["top5_acc"] = map_str
        else:

            
            # assert False

            unique_animal_labels = torch.unique(self.video_labels_animal)
            unique_behavior_labels = torch.unique(self.video_labels_behavior)


            # print("unique number animal labels", len(unique_animal_labels))
            # print("unique number behavior labels", len(unique_behavior_labels))
           
            animal_head, animal_med, animal_tail,behavior_head, behavior_med, \
            behavior_tail, composition_head, composition_med, composition_tail, \
            ah_bh, am_bh, af_bh,ah_bm, am_bm, af_bm, ah_bf, am_bf, af_bf = \
                 self.classify_longtail_classes("/train.csv")

                
            animal_frequency, behavior_frequency, composition_frequency = self.compute_frequency("/train.csv")



            animal_perclass_acc_head = torch.Tensor([0,0])
            animal_perclass_acc_med = torch.Tensor([0,0])
            animal_perclass_acc_tail = torch.Tensor([0,0])
            number_animal_head = number_animal_med = number_animal_tail = 0

            behavior_perclass_acc_head = torch.Tensor([0,0])
            behavior_perclass_acc_med = torch.Tensor([0,0])
            behavior_perclass_acc_tail = torch.Tensor([0,0])
            number_behavior_head = number_behavior_med = number_behavior_tail = 0


            composition_perclass_acc_head = torch.Tensor([0,0])
            composition_perclass_acc_med = torch.Tensor([0,0])
            composition_perclass_acc_tail = torch.Tensor([0,0])
            number_composition_head = number_composition_med = number_composition_tail = 0


            unique_animal_result = []
            
            
            
            
            
            animal_head_accumulation_pred=None
            animal_head_accumulation_gt = None
            animal_med_accumulation_pred = None
            animal_med_accumulation_gt = None
            animal_tail_accumulation_pred = None
            animal_tail_accumulation_gt = None
            

            for animal in unique_animal_labels:
                label_indexs = (self.video_labels_animal==animal).nonzero(as_tuple=True)[0]
                
                sub_label = torch.index_select(self.video_labels_animal,0,label_indexs)
                sub_pred = torch.index_select(self.video_preds_animal,0, label_indexs)


                if str(torch.unique(sub_label).item()) in animal_head:
                    if animal_head_accumulation_pred == None:
                        animal_head_accumulation_pred = sub_pred
                    else:
                        animal_head_accumulation_pred = torch.cat([animal_head_accumulation_pred, sub_pred],dim=0)
                    
                    
                    if animal_head_accumulation_gt == None:
                        animal_head_accumulation_gt = sub_label
                    else:
                        animal_head_accumulation_gt = torch.cat([animal_head_accumulation_gt, sub_label],dim=0)
                    
                    
                elif str(torch.unique(sub_label).item()) in animal_med:
                    if animal_med_accumulation_pred == None:
                        animal_med_accumulation_pred = sub_pred
                    else:
                        animal_med_accumulation_pred = torch.cat([animal_med_accumulation_pred, sub_pred],dim=0)
                    
                    
                    if animal_med_accumulation_gt == None:
                        animal_med_accumulation_gt = sub_label
                    else:
                        animal_med_accumulation_gt = torch.cat([animal_med_accumulation_gt, sub_label],dim=0)                   
                    
                elif str(torch.unique(sub_label).item()) in animal_tail:
                    if animal_tail_accumulation_pred == None:
                        animal_tail_accumulation_pred = sub_pred
                    else:
                        animal_tail_accumulation_pred = torch.cat([animal_tail_accumulation_pred, sub_pred],dim=0)
                    
                    
                    if animal_tail_accumulation_gt == None:
                        animal_tail_accumulation_gt = sub_label
                    else:
                        animal_tail_accumulation_gt = torch.cat([animal_tail_accumulation_gt, sub_label],dim=0)      
                else:
                    print(sub_label, "do not exist in the training dataset")


                
            num_topks_correct_animal_head,pred_index, gt_index=metrics.topks_correct(animal_head_accumulation_pred,animal_head_accumulation_gt,ks,return_pred_index=True)
                
            topks_animal_head = [
                (x / animal_head_accumulation_pred.size(0)) * 100.0
                for x in num_topks_correct_animal_head
            ]
            
            assert len({len(ks), len(topks_animal_head)}) == 1
            for k, topk in zip(ks, topks_animal_head):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["animal_head_perexample_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
                
                
                
            num_topks_correct_animal_med,pred_index, gt_index=metrics.topks_correct(animal_med_accumulation_pred,animal_med_accumulation_gt,ks,return_pred_index=True)
                
            topks_animal_med = [
                (x / animal_med_accumulation_pred.size(0)) * 100.0
                for x in num_topks_correct_animal_med
            ]
            
            assert len({len(ks), len(topks_animal_med)}) == 1
            for k, topk in zip(ks, topks_animal_med):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["animal_med_perexample_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
          


            num_topks_correct_animal_tail,pred_index, gt_index=metrics.topks_correct(animal_tail_accumulation_pred,animal_tail_accumulation_gt,ks,return_pred_index=True)
                
            topks_animal_tail = [
                (x / animal_tail_accumulation_pred.size(0)) * 100.0
                for x in num_topks_correct_animal_tail
            ]
            
            assert len({len(ks), len(topks_animal_tail)}) == 1
            for k, topk in zip(ks, topks_animal_tail):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["animal_tail_perexample_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )









            '''
            behavior per classs analysis
            '''

            unique_behavior_result=[]

            behavior_head_accumulation_pred=None
            behavior_head_accumulation_gt = None
            behavior_med_accumulation_pred = None
            behavior_med_accumulation_gt = None
            behavior_tail_accumulation_pred = None
            behavior_tail_accumulation_gt = None
            
            
            for behavior in unique_behavior_labels:
                label_indexs = (self.video_labels_behavior==behavior).nonzero(as_tuple=True)[0]
                sub_label = torch.index_select(self.video_labels_behavior,0,label_indexs)
                sub_pred = torch.index_select(self.video_preds_behavior,0, label_indexs)
                

                # print("sub label",torch.unique(sub_label).item())
                if str(torch.unique(sub_label).item()) in behavior_head:
                    if behavior_head_accumulation_pred == None:
                        behavior_head_accumulation_pred = sub_pred
                    else:
                        behavior_head_accumulation_pred = torch.cat([behavior_head_accumulation_pred, sub_pred],dim=0)
                    
                    
                    if behavior_head_accumulation_gt == None:
                        behavior_head_accumulation_gt = sub_label
                    else:
                        behavior_head_accumulation_gt = torch.cat([behavior_head_accumulation_gt, sub_label],dim=0)
                    
                elif str(torch.unique(sub_label).item()) in behavior_med:
                    if behavior_med_accumulation_pred == None:
                        behavior_med_accumulation_pred = sub_pred
                    else:
                        behavior_med_accumulation_pred = torch.cat([behavior_med_accumulation_pred, sub_pred],dim=0)
                    
                    
                    if behavior_med_accumulation_gt == None:
                        behavior_med_accumulation_gt = sub_label
                    else:
                        behavior_med_accumulation_gt = torch.cat([behavior_med_accumulation_gt, sub_label],dim=0)
                        
                elif str(torch.unique(sub_label).item()) in behavior_tail:
                
                    if behavior_tail_accumulation_pred == None:
                        behavior_tail_accumulation_pred = sub_pred
                    else:
                        behavior_tail_accumulation_pred = torch.cat([behavior_tail_accumulation_pred, sub_pred],dim=0)
                    
                    
                    if behavior_tail_accumulation_gt == None:
                        behavior_tail_accumulation_gt = sub_label
                    else:
                        behavior_tail_accumulation_gt = torch.cat([behavior_tail_accumulation_gt, sub_label],dim=0)
                        
                        
                else:
                    print("behavior ",str(torch.unique(sub_label).item(),"do not exist in training"))
                    
                    



                
            num_topks_correct_behavior_head,pred_index, gt_index=metrics.topks_correct(behavior_head_accumulation_pred,behavior_head_accumulation_gt,ks,return_pred_index=True)
                
            topks_behavior_head = [
                (x / behavior_head_accumulation_pred.size(0)) * 100.0
                for x in num_topks_correct_behavior_head
            ]
            
            assert len({len(ks), len(topks_behavior_head)}) == 1
            for k, topk in zip(ks, topks_behavior_head):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["behavior_head_perexample_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
                
                
                
            num_topks_correct_behavior_med,pred_index, gt_index=metrics.topks_correct(behavior_med_accumulation_pred,behavior_med_accumulation_gt,ks,return_pred_index=True)
                
            topks_behavior_med = [
                (x / behavior_med_accumulation_pred.size(0)) * 100.0
                for x in num_topks_correct_behavior_med
            ]
            
            assert len({len(ks), len(topks_behavior_med)}) == 1
            for k, topk in zip(ks, topks_behavior_med):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["behavior_med_perexample_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
          


            num_topks_correct_behavior_tail,pred_index, gt_index=metrics.topks_correct(behavior_tail_accumulation_pred,behavior_tail_accumulation_gt,ks,return_pred_index=True)
                
            topks_behavior_tail = [
                (x / behavior_tail_accumulation_pred.size(0)) * 100.0
                for x in num_topks_correct_behavior_tail
            ]
            
            assert len({len(ks), len(topks_behavior_tail)}) == 1
            for k, topk in zip(ks, topks_behavior_tail):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["behavior_tail_perexample_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )











            '''
            compositional evaluation
            '''
            ### compositional evaluate results


            concat_labels = torch.cat([self.video_labels_animal.view(1,-1),self.video_labels_behavior.view(1,-1)],0).t()
            unique_composition_labels = concat_labels.unique(dim=0)


            # print("concat label", concat_labels)
            # print("unique composition labels", unique_composition_labels)
            # print("video labels shape", concat_labels.shape)
            # print("unique composition shape",unique_composition_labels.shape)

            unique_composition_result = []



            composition_head_accumulation_pred_animal=None
            composition_head_accumulation_gt_animal = None
            composition_med_accumulation_pred_animal = None
            composition_med_accumulation_gt_animal = None
            composition_tail_accumulation_pred_animal = None
            composition_tail_accumulation_gt_animal = None


            composition_head_accumulation_pred_behavior=None
            composition_head_accumulation_gt_behavior = None
            composition_med_accumulation_pred_behavior = None
            composition_med_accumulation_gt_behavior = None
            composition_tail_accumulation_pred_behavior = None
            composition_tail_accumulation_gt_behavior = None

            for label in unique_composition_labels:
                composition_index = ((concat_labels==label).sum(1)==len(label)).nonzero(as_tuple = True)[0]

                sub_animal_pred = torch.index_select(self.video_preds_animal,0,composition_index)
                sub_animal_label = torch.index_select(self.video_labels_animal,0,composition_index)
                sub_behavior_pred = torch.index_select(self.video_preds_behavior,0,composition_index)
                sub_behavior_label = torch.index_select(self.video_labels_behavior,0,composition_index)
                
                animal_behavior = str(torch.unique(sub_animal_label).item())+"_"+str(torch.unique(sub_behavior_label).item())




                if animal_behavior in composition_head:
                    if composition_head_accumulation_pred_animal == None or composition_head_accumulation_pred_behavior == None:
                        composition_head_accumulation_pred_animal = sub_animal_pred
                        composition_head_accumulation_pred_behavior = sub_behavior_pred
                    else:
                        composition_head_accumulation_pred_animal = torch.cat([composition_head_accumulation_pred_animal, sub_animal_pred],dim=0)                    
                        composition_head_accumulation_pred_behavior = torch.cat([composition_head_accumulation_pred_behavior, sub_behavior_pred],dim=0)
                    
                    if composition_head_accumulation_gt_animal == None or composition_head_accumulation_gt_behavior==None:
                        composition_head_accumulation_gt_animal = sub_animal_label
                        composition_head_accumulation_gt_behavior = sub_behavior_label
                        
                    else:
                        composition_head_accumulation_gt_animal = torch.cat([composition_head_accumulation_gt_animal, sub_animal_label],dim=0)
                        composition_head_accumulation_gt_behavior = torch.cat([composition_head_accumulation_gt_behavior, sub_behavior_label], dim = 0)
                    
                    
                    
                    
                elif animal_behavior in composition_med:
                    if composition_med_accumulation_pred_behavior == None or composition_med_accumulation_pred_animal ==None :
                        composition_med_accumulation_pred_animal = sub_animal_pred
                        composition_med_accumulation_pred_behavior = sub_behavior_pred
                    else:
                        composition_med_accumulation_pred_animal = torch.cat([composition_med_accumulation_pred_animal, sub_animal_pred],dim=0)                    
                        composition_med_accumulation_pred_behavior = torch.cat([composition_med_accumulation_pred_behavior, sub_behavior_pred],dim=0)
                    
                    if composition_med_accumulation_gt_behavior == None or composition_med_accumulation_gt_animal == None:
                        composition_med_accumulation_gt_animal = sub_animal_label
                        composition_med_accumulation_gt_behavior = sub_behavior_label
                        
                    else:
                        composition_med_accumulation_gt_animal = torch.cat([composition_med_accumulation_gt_animal, sub_animal_label],dim=0)
                        composition_med_accumulation_gt_behavior = torch.cat([composition_med_accumulation_gt_behavior, sub_behavior_label], dim = 0)
                    
               
                elif animal_behavior in composition_tail:

                    if composition_tail_accumulation_pred_animal == None or composition_tail_accumulation_pred_behavior ==None:
                        composition_tail_accumulation_pred_animal = sub_animal_pred
                        composition_tail_accumulation_pred_behavior = sub_behavior_pred
                    else:
                        composition_tail_accumulation_pred_animal = torch.cat([composition_tail_accumulation_pred_animal, sub_animal_pred],dim=0)                    
                        composition_tail_accumulation_pred_behavior = torch.cat([composition_tail_accumulation_pred_behavior, sub_behavior_pred],dim=0)
                    
                    if composition_tail_accumulation_gt_animal == None or composition_tail_accumulation_gt_behavior==None:
                        composition_tail_accumulation_gt_animal = sub_animal_label
                        composition_tail_accumulation_gt_behavior = sub_behavior_label
                        
                    else:
                        composition_tail_accumulation_gt_animal = torch.cat([composition_tail_accumulation_gt_animal, sub_animal_label],dim=0)
                        composition_tail_accumulation_gt_behavior = torch.cat([composition_tail_accumulation_gt_behavior, sub_behavior_label], dim = 0)
                        
                        
                else:
                    print("composition not in head med tail",animal_behavior)



            num_topks_animal,num_toks_behavior,num_topks_correct_composition_head= \
            metrics.topks_correct_composition(composition_head_accumulation_pred_animal, composition_head_accumulation_gt_animal, \
            composition_head_accumulation_pred_behavior, composition_head_accumulation_gt_behavior, ks)
            
            

            topks_composition_head = [
                (x / composition_head_accumulation_pred_animal.size(0)) * 100.0
                for x in num_topks_correct_composition_head
            ]



            assert len({len(ks), len(topks_composition_head)}) == 1
            for k, topk in zip(ks, topks_composition_head):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["composition_head_perexample_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
                
                
                

            num_topks_animal,num_toks_behavior,num_topks_correct_composition_med= \
            metrics.topks_correct_composition(composition_med_accumulation_pred_animal, composition_med_accumulation_gt_animal, \
            composition_med_accumulation_pred_behavior, composition_med_accumulation_gt_behavior, ks)
            
            

            topks_composition_med = [
                (x / composition_med_accumulation_pred_animal.size(0)) * 100.0
                for x in num_topks_correct_composition_med
            ]



            assert len({len(ks), len(topks_composition_med)}) == 1
            for k, topk in zip(ks, topks_composition_med):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["composition_med_perexample_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )




            num_topks_animal,num_toks_behavior,num_topks_correct_composition_tail= \
            metrics.topks_correct_composition(composition_tail_accumulation_pred_animal, composition_tail_accumulation_gt_animal, \
            composition_tail_accumulation_pred_behavior, composition_tail_accumulation_gt_behavior, ks)
            
            

            topks_composition_tail = [
                (x / composition_tail_accumulation_pred_animal.size(0)) * 100.0
                for x in num_topks_correct_composition_tail
            ]



            assert len({len(ks), len(topks_composition_tail)}) == 1
            for k, topk in zip(ks, topks_composition_tail):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["composition_tail_perexample_top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )      
                



        logging.log_json_stats(self.stats)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()

        self.lr = None
        # Current minibatch errors (smoothed over a window).

        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR

        self.loss_animal = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total_animal = 0.0
        self.mb_top1_err_animal = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err_animal = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis_animal = 0
        self.num_top5_mis_animal = 0
        self.num_samples_animal = 0

        self.loss_behavior = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total_behavior = 0.0
        self.mb_top1_err_behavior = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err_behavior = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis_behavior = 0
        self.num_top5_mis_behavior = 0
        self.num_samples_behavior = 0



        self.loss_composition = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total_composition = 0.0
        self.mb_top1_err_composition = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err_composition = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis_composition = 0
        self.num_top5_mis_composition = 0
        self.num_samples_composition = 0


    def reset(self):
        """
        Reset the Meter.
        """

        self.lr = None
        self.num_samples = 0


        self.loss_animal.reset()
        self.loss_total_animal = 0.0
        self.mb_top1_err_animal.reset()
        self.mb_top5_err_animal.reset()
        self.num_top1_mis_animal = 0
        self.num_top5_mis_animal = 0
        self.num_samples_animal =0


        self.loss_behavior.reset()
        self.loss_total_behavior = 0.0  
        self.mb_top1_err_behavior.reset()
        self.mb_top5_err_behavior.reset()
        self.num_top1_mis_behavior = 0
        self.num_top5_mis_behavior = 0
        self.num_samples_behavior = 0



        self.loss_composition.reset()
        self.loss_total_composition = 0.0
        self.mb_top1_err_composition.reset()
        self.mb_top5_err_composition.reset()
        self.num_top1_mis_composition = 0
        self.num_top5_mis_composition = 0
        self.num_samples_composition = 0
        

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size,data_type):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """

        self.lr = lr
        if data_type =="animal":
            self.loss_animal.add_value(loss)
            
            self.loss_total_animal += loss * mb_size
            self.num_samples_animal += mb_size

            if not self._cfg.DATA.MULTI_LABEL:
                # Current minibatch stats
                self.mb_top1_err_animal.add_value(top1_err)
                self.mb_top5_err_animal.add_value(top5_err)
                # Aggregate stats
                self.num_top1_mis_animal += top1_err * mb_size
                self.num_top5_mis_animal += top5_err * mb_size
        elif data_type == "behavior":
            self.loss_behavior.add_value(loss)
            self.loss_total_behavior += loss * mb_size
            self.num_samples_behavior += mb_size

            if not self._cfg.DATA.MULTI_LABEL:
                # Current minibatch stats
                self.mb_top1_err_behavior.add_value(top1_err)
                self.mb_top5_err_behavior.add_value(top5_err)
                # Aggregate stats
                self.num_top1_mis_behavior += top1_err * mb_size
                self.num_top5_mis_behavior += top5_err * mb_size

        else:
            self.loss_composition.add_value(loss)
            self.loss_total_composition += loss * mb_size
            self.num_samples_composition += mb_size

            if not self._cfg.DATA.MULTI_LABEL:
                # Current minibatch stats
                self.mb_top1_err_composition.add_value(top1_err)
                self.mb_top5_err_composition.add_value(top5_err)
                # Aggregate stats
                self.num_top1_mis_composition += top1_err * mb_size
                self.num_top5_mis_composition += top5_err * mb_size



    def log_iter_stats(self, cur_epoch, cur_iter, data_type):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        if data_type =="animal":
            stats = {
                "_type": "train_iter_{}".format(
                    "ssl" if self._cfg.TASK == "ssl" else ""
                ),
                "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
                "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "eta": eta,
                "animal_prediction_loss": self.loss_animal.get_win_median(),
                "lr": self.lr,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            }
            if not self._cfg.DATA.MULTI_LABEL:
                stats["top1_err_animal"] = self.mb_top1_err_animal.get_win_median()
                stats["top5_err_animal"] = self.mb_top5_err_animal.get_win_median()
            logging.log_json_stats(stats)
        elif data_type =="behavior":
            stats = {
                "_type": "train_iter_{}".format(
                    "ssl" if self._cfg.TASK == "ssl" else ""
                ),
                "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
                "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "eta": eta,
                "loss_behavior": self.loss_behavior.get_win_median(),
                "lr": self.lr,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            }
            if not self._cfg.DATA.MULTI_LABEL:
                stats["top1_err_behavior"] = self.mb_top1_err_behavior.get_win_median()
                stats["top5_err_behavior"] = self.mb_top5_err_behavior.get_win_median()
            logging.log_json_stats(stats)

        elif data_type =="composition":
            stats = {
                "_type": "train_iter_{}".format(
                    "ssl" if self._cfg.TASK == "ssl" else ""
                ),
                "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
                "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "eta": eta,
                "loss_composition": self.loss_composition.get_win_median(),
                "lr": self.lr,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            }
            if not self._cfg.DATA.MULTI_LABEL:
                stats["top1_err_composition"] = self.mb_top1_err_composition.get_win_median()
                stats["top5_err_composition"] = self.mb_top5_err_composition.get_win_median()
            logging.log_json_stats(stats)

        
    def log_epoch_stats(self, cur_epoch, data_type):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        if data_type =="animal":
            stats = {
                "_type": "train_epoch{}".format(
                    "_ssl" if self._cfg.TASK == "ssl" else ""
                ),
                "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "eta": eta,
                "lr": self.lr,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
                "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            }
            if not self._cfg.DATA.MULTI_LABEL:
                top1_err = self.num_top1_mis_animal / self.num_samples_animal
                top5_err = self.num_top5_mis_animal / self.num_samples_animal
                avg_loss = self.loss_total_animal / self.num_samples_animal
                stats["top1_err_animal"] = top1_err
                stats["top5_err_animal"] = top5_err
                stats["loss_animal"] = avg_loss
            logging.log_json_stats(stats, self.output_dir)
        
        elif data_type =="behavior":
            stats = {
            "_type": "train_epoch{}".format(
                "_ssl" if self._cfg.TASK == "ssl" else ""
            ),
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            }
            if not self._cfg.DATA.MULTI_LABEL:
                top1_err = self.num_top1_mis_behavior / self.num_samples_behavior
                top5_err = self.num_top5_mis_behavior / self.num_samples_behavior
                avg_loss = self.loss_total_behavior / self.num_samples_behavior
                stats["top1_err_behavior"] = top1_err
                stats["top5_err_behavior"] = top5_err
                stats["loss_behavior"] = avg_loss
            logging.log_json_stats(stats, self.output_dir)
        elif data_type =="composition":
            stats = {
            "_type": "train_epoch{}".format(
                "_ssl" if self._cfg.TASK == "ssl" else ""
            ),
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            }
            if not self._cfg.DATA.MULTI_LABEL:
                top1_err = self.num_top1_mis_composition / self.num_samples_composition
                top5_err = self.num_top5_mis_composition / self.num_samples_composition
                avg_loss = self.loss_total_composition / self.num_samples_composition
                stats["top1_err_composition"] = top1_err
                stats["top5_err_composition"] = top5_err
                stats["loss_composition"] = avg_loss
            logging.log_json_stats(stats, self.output_dir)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.output_dir = cfg.OUTPUT_DIR
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()


        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err_animal = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err_animal = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err_animal = 100.0
        self.min_top5_err_animal = 100.0
        # Number of misclassified examples.
        self.num_top1_mis_animal = 0
        self.num_top5_mis_animal = 0
        self.num_samples_animal = 0
        self.all_preds_animal = []
        self.all_labels_animal = []




        self.mb_top1_err_behavior = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err_behavior = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err_behavior = 100.0
        self.min_top5_err_behavior = 100.0
        # Number of misclassified examples.
        self.num_top1_mis_behavior = 0
        self.num_top5_mis_behavior = 0
        self.num_samples_behavior = 0
        self.all_preds_behavior = []
        self.all_labels_behavior = []



        self.mb_top1_err_composition = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err_composition = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err_composition = 100.0
        self.min_top5_err_composition = 100.0
        # Number of misclassified examples.
        self.num_top1_mis_composition = 0
        self.num_top5_mis_composition = 0
        self.num_samples_composition = 0
        self.all_preds_composition = []
        self.all_labels_composition = []
        






    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.data_timer.reset()
        self.net_timer.reset()

        self.mb_top1_err_animal.reset()
        self.mb_top5_err_animal.reset()
        self.num_top1_mis_animal = 0
        self.num_top5_mis_animal = 0
        self.num_samples_animal = 0
        self.all_preds_animal = []
        self.all_labels_animal = []




        self.mb_top1_err_behavior.reset()
        self.mb_top5_err_behavior.reset()
        self.num_top1_mis_behavior = 0
        self.num_top5_mis_behavior = 0
        self.num_samples_behavior = 0
        self.all_preds_behavior = []
        self.all_labels_behavior = []


        self.mb_top1_err_composition.reset()
        self.mb_top5_err_composition.reset()
        self.num_top1_mis_composition = 0
        self.num_top5_mis_composition = 0
        self.num_samples_composition = 0
        self.all_preds_composition = []
        self.all_labels_composition = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, mb_size,data_type):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """

        if data_type =="animal":
            self.mb_top1_err_animal.add_value(top1_err)
            self.mb_top5_err_animal.add_value(top5_err)
            self.num_top1_mis_animal += top1_err * mb_size
            self.num_top5_mis_animal += top5_err * mb_size
            self.num_samples_animal += mb_size
        elif data_type =="behavior":
            self.mb_top1_err_behavior.add_value(top1_err)
            self.mb_top5_err_behavior.add_value(top5_err)
            self.num_top1_mis_behavior += top1_err * mb_size
            self.num_top5_mis_behavior += top5_err * mb_size
            self.num_samples_behavior += mb_size
        elif data_type =="composition":
            self.mb_top1_err_composition.add_value(top1_err)
            self.mb_top5_err_composition.add_value(top5_err)
            self.num_top1_mis_composition += top1_err * mb_size
            self.num_top5_mis_composition += top5_err * mb_size
            self.num_samples_composition += mb_size

    def update_predictions(self, preds, labels,data_type):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.

        if data_type =="animal":
            self.all_preds_animal.append(preds)
            self.all_labels_animal.append(labels)
        elif data_type =="behavior":
            self.all_preds_behavior.append(preds)
            self.all_labels_behavior.append(labels)
        elif data_type =="composition":
            self.all_preds_composition.append(preds)
            self.all_labels_composition.append(labels)           

    def log_iter_stats(self, cur_epoch, cur_iter, data_type):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter{}".format(
                "_ssl" if self._cfg.TASK == "ssl" else ""
            ),
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        
        if not self._cfg.DATA.MULTI_LABEL:
            if data_type =="animal":
                stats["top1_err_animal"] = self.mb_top1_err_animal.get_win_median()
                stats["top5_err_animal"] = self.mb_top5_err_animal.get_win_median()
            elif data_type =="behavior":
                stats["top1_err_behavior"] = self.mb_top1_err_behavior.get_win_median()
                stats["top5_err_behavior"] = self.mb_top5_err_behavior.get_win_median()
            elif data_type =="composition":
                stats["top1_err_composition"] = self.mb_top1_err_composition.get_win_median()
                stats["top5_err_composition"] = self.mb_top5_err_composition.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch, data_type):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch{}".format(
                "_ssl" if self._cfg.TASK == "ssl" else ""
            ),
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            stats["map"] = get_map(
                torch.cat(self.all_preds).cpu().numpy(),
                torch.cat(self.all_labels).cpu().numpy(),
            )
        else:
            if data_type =="animal":
                top1_err = self.num_top1_mis_animal / self.num_samples_animal
                top5_err = self.num_top5_mis_animal / self.num_samples_animal
                self.min_top1_err = min(self.min_top1_err_animal, top1_err)
                self.min_top5_err = min(self.min_top5_err_animal, top5_err)

                stats["top1_err_animal"] = top1_err
                stats["top5_err_animal"] = top5_err
                stats["min_top1_err_animal"] = self.min_top1_err
                stats["min_top5_err_animal"] = self.min_top5_err
            elif data_type =="behavior":
                top1_err = self.num_top1_mis_behavior / self.num_samples_behavior
                top5_err = self.num_top5_mis_behavior / self.num_samples_behavior
                self.min_top1_err = min(self.min_top1_err_behavior, top1_err)
                self.min_top5_err = min(self.min_top5_err_behavior, top5_err)

                stats["top1_err_behavior"] = top1_err
                stats["top5_err_behavior"] = top5_err
                stats["min_top1_err_behavior"] = self.min_top1_err
                stats["min_top5_err_behavior"] = self.min_top5_err
            elif data_type =="composition":
                top1_err = self.num_top1_mis_composition / self.num_samples_composition
                top5_err = self.num_top5_mis_composition / self.num_samples_composition
                self.min_top1_err = min(self.min_top1_err_composition, top1_err)
                self.min_top5_err = min(self.min_top5_err_composition, top5_err)

                stats["top1_err_composition"] = top1_err
                stats["top5_err_composition"] = top5_err
                stats["min_top1_err_composition"] = self.min_top1_err
                stats["min_top5_err_composition"] = self.min_top5_err

        logging.log_json_stats(stats, self.output_dir)


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)
