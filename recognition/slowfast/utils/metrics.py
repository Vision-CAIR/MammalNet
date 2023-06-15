#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch


def topks_correct(preds, labels, ks,return_pred_index=False):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.


    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]

    if return_pred_index:
        return topks_correct, top_max_k_inds, rep_max_k_labels
    else:
        return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]







def topks_correct_composition(animal_preds, animal_labels, behavior_preds, behavior_labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert animal_preds.size(0) == animal_labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        animal_preds, max(ks), dim=1, largest=True, sorted=True
    )


    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = animal_labels.view(1, -1).expand_as(top_max_k_inds)

    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct_animal = top_max_k_inds.eq(rep_max_k_labels)
 
    # Compute the number of topk correct predictions for each k.
    topks_correct_animal = [top_max_k_correct_animal[:k, :].float().sum() for k in ks]

    assert behavior_preds.size(0) == behavior_labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        behavior_preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = behavior_labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct_behavior = top_max_k_inds.eq(rep_max_k_labels)


    # Compute the number of topk correct predictions for each k.
    topks_correct_behavior = [top_max_k_correct_behavior[:k, :].float().sum() for k in ks]

    top_max_k_coposition = top_max_k_correct_behavior.float() + top_max_k_correct_animal.float()

    topks_correct_composition = [(top_max_k_coposition[:k,:].sum(0)>1).sum() for k in ks]
    

    return topks_correct_animal, topks_correct_behavior, topks_correct_composition




def topks_correct_composition_per_class_acc(animal_preds, animal_labels, behavior_preds, behavior_labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert animal_preds.size(0) == animal_labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        animal_preds, max(ks), dim=1, largest=True, sorted=True
    )

    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = animal_labels.view(1, -1).expand_as(top_max_k_inds)


    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct_animal = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct_animal = [top_max_k_correct_animal[:k, :].float().sum() for k in ks]




    assert behavior_preds.size(0) == behavior_labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        behavior_preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = behavior_labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct_behavior = top_max_k_inds.eq(rep_max_k_labels)


    # Compute the number of topk correct predictions for each k.
    topks_correct_behavior = [top_max_k_correct_behavior[:k, :].float().sum() for k in ks]


    top_max_k_coposition = top_max_k_correct_behavior.float() + top_max_k_correct_animal.float()

    topks_correct_composition = [(top_max_k_coposition[:k,:].sum(0)>1).sum() for k in ks]

    return topks_correct_animal, topks_correct_behavior, topks_correct_composition






def topk_errors_composition(animal_preds, animal_labels, behavior_preds, behavior_labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct_animal, num_topks_correct_behavior,num_topks_correct_composition = topks_correct_composition(animal_preds, animal_labels, behavior_preds, behavior_labels, ks)
    
    topks_error_animal = [(1.0 - x / animal_preds.size(0)) * 100.0 for x in num_topks_correct_animal]
    topks_error_behavior = [(1.0 - x / behavior_preds.size(0)) * 100.0 for x in num_topks_correct_behavior]
    topks_error_composition = [(1.0 - x / animal_preds.size(0)) * 100.0 for x in num_topks_correct_composition]



    return topks_error_animal, topks_error_behavior, topks_error_composition





def topk_accuracies_composition(animal_preds, animal_labels, behavior_preds, behavior_labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct_animal, num_topks_correct_behavior, num_topks_correct_composition = topks_correct_composition(animal_preds,animal_labels,behavior_preds, behavior_labels, ks)
    
    topks_accuracy_animal = [(x / animal_preds.size(0)) * 100.0 for x in num_topks_correct_animal]
    topks_accuracy_behavior = [(x / animal_preds.size(0)) * 100.0 for x in num_topks_correct_behavior]
    topks_accuracy_composition = [(x / animal_preds.size(0)) * 100.0 for x in num_topks_correct_composition]
    return topks_accuracy_animal, topks_accuracy_behavior, topks_accuracy_composition
