"""
Metrics to measure classification performance
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score, confusion_matrix
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


def calculate_equal_opportunity_difference(predictions, labels, sensitive_attribute):
    # Convert predictions and labels to numpy arrays if they are tensors
    predictions = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    sensitive_attribute = sensitive_attribute.numpy() if isinstance(sensitive_attribute, torch.Tensor) else sensitive_attribute
    
    # Get indices for the privileged and unprivileged groups
    privileged_indices = sensitive_attribute == 1
    unprivileged_indices = sensitive_attribute == 0
    
    # Calculate true positive rates for both groups
    tpr_privileged = sum(predictions[privileged_indices] * labels[privileged_indices]) / sum(labels[privileged_indices])
    tpr_unprivileged = sum(predictions[unprivileged_indices] * labels[unprivileged_indices]) / sum(labels[unprivileged_indices])
    
    # Calculate Equal Opportunity Difference
    eod = abs(tpr_privileged - tpr_unprivileged)

    
    return eod



def mutinfo(_Cset_yhat,_Cset_s):
    demoErr=mutual_info_score(_Cset_s,_Cset_yhat)
    return abs(demoErr)


def get_logits_labels(model, data_loader, device):
    """
    Utility function to get logits and labels.
    """
    model.eval()
    logits = []
    labels = []
    sensitive = []
    with torch.no_grad():
        for data, label,s in data_loader:
            data = data.to(device)
            label = label.to(device)

            logit = model(data)
            logits.append(logit)
            labels.append(label)
            sensitive.append(s)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    sensitives = torch.cat(sensitive, dim=0)
    return logits, labels, sensitives


def test_classification_net_softmax(softmax_prob, labels, sensitives, is_val):
    """
    This function reports classification accuracy and confusion matrix given softmax vectors and
    labels from a model.
    """
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    confidence_vals, predictions = torch.max(softmax_prob, dim=1)
    labels_list.extend(labels.cpu().numpy())
    predictions_list.extend(predictions.cpu().numpy())
    confidence_vals_list.extend(confidence_vals.cpu().numpy())
    accuracy = accuracy_score(labels_list, predictions_list)
    mut_info_fairness = mutinfo(predictions_list, sensitives)



    dpd_fairness = demographic_parity_difference(labels_list, predictions_list, sensitive_features=sensitives)
    eod_fairness = equalized_odds_difference(labels_list, predictions_list, sensitive_features=sensitives)




    return (
        confusion_matrix(labels_list, predictions_list),
        accuracy,
        mut_info_fairness,
        dpd_fairness,
        eod_fairness,
        labels_list,
        predictions_list,
        confidence_vals_list,
    )


def test_classification_net_logits(logits, labels, sensitives, is_val):
    """
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    """
    softmax_prob = F.softmax(logits, dim=1)
    return test_classification_net_softmax(softmax_prob, labels, sensitives, is_val)


def test_classification_net(model, data_loader, device, is_val):
    """
    This function reports classification accuracy and confusion matrix over a dataset.
    """



    logits, labels, sensitives = get_logits_labels(model, data_loader, device)
    return test_classification_net_logits(logits, labels, sensitives, is_val)

from torch.utils.data import DataLoader, TensorDataset

def test_classification_net_decoupled(model0, model1, data_loader, device, is_val):
    data = []
    labels = []
    s_values = []

    for batch_data, batch_labels, batch_s in data_loader:
        data.extend(batch_data)
        labels.extend(batch_labels)
        s_values.extend(batch_s)

    # Convert lists to tensors
    data = torch.stack(data)
    labels = torch.stack(labels)
    s_values = torch.stack(s_values)

    # Separate data based on the value of s
    data_s0 = data[s_values != 1]
    labels_s0 = labels[s_values != 1]
    s_s0 = s_values[s_values != 1]
    data_s1 = data[s_values == 1]
    labels_s1 = labels[s_values == 1]
    s_s1 = s_values[s_values == 1]

    # Create separate TensorDatasets for s == 0 and s == 1
    dataset_s0 = TensorDataset(data_s0, labels_s0, s_s0)
    dataset_s1 = TensorDataset(data_s1, labels_s1, s_s1)

    # Define batch size (same as the original dataloader)
    batch_size = data_loader.batch_size

    # Create separate DataLoaders for s == 0 and s == 1
    dataloader_s0 = DataLoader(dataset_s0, batch_size=batch_size, shuffle=True)
    dataloader_s1 = DataLoader(dataset_s1, batch_size=batch_size, shuffle=True)

    logits0, labels0, sensitives0 = get_logits_labels(model0, dataloader_s0, device)
    logits1, labels1, sensitives1 = get_logits_labels(model1, dataloader_s1, device)
    

    logits = torch.cat((logits0, logits1), dim=0)
    labels = torch.cat((labels0, labels1), dim=0)
    sensitives = torch.cat((sensitives0, sensitives1), dim =0)
    #import pdb; pdb.set_trace()
    return test_classification_net_logits(logits, labels, sensitives, is_val)


