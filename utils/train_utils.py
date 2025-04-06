"""
This module contains methods for training models.
"""
# import sys
# import os
# parent = os.path.abspath("..")
# print(parent)
# sys.path.append(parent+"/DDU")
# print(sys.path)
import torch
from torch.nn import functional as F
from torch import nn
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from utils.gmm_utils import gmm_forward
#from metrics.classification_metrics_fair2.py import ddp

loss_function_dict = {"cross_entropy": F.cross_entropy}




def ddp(z,yhat):

    length = len(z)
    countz =0;

    for item in z:
        if item==1:

            countz +=1;
    p1 = (countz*1.0)/length;
    if length ==0 or p1 ==0 or p1 == 1 or p1 == 1.0:

        return torch.tensor(0.0001)

    sum = 0;
    for i in range(len(z)):
        cur_z = z[i]
        cur_y_hat = yhat[i]

        sum += ((cur_z + 1) / 2 - p1) * cur_y_hat
    sum = sum * 1/(p1*(1 - p1));
    output = sum / length
    output = torch.abs(output)
    if torch.isnan(output).any():
        import pdb; pdb.set_trace()

    return output



class MeanLoss(nn.Module):
    def __init__(self, device, fair_criteria):
        super(MeanLoss, self).__init__()
        self.device = device
        assert fair_criteria in ['EqOdd', 'EqOpp']
        self.fair_criteria = fair_criteria

    def forward(self, outputs, labels, group):
        result = torch.FloatTensor([0.0]).contiguous().to(self.device)
        outputs = nn.LogSigmoid()(outputs).unsqueeze(1)
        if self.fair_criteria == 'EqOdd':
            unique_labels = [0, 1]
        else:
            unique_labels = [0]
        for the_label in unique_labels:
            if (labels == the_label).sum() > 0:
                result = result + self.compute_mean_gap_group(outputs[labels == the_label], group[labels == the_label])
            else:
                print("Skipping regularization due to no samples")
        #import pdb; pdb.set_trace()
        return result

    def compute_mean_gap_group(self, outputs, group):
        result = torch.FloatTensor([0.0]).contiguous().to(self.device)
        # unique_groups = group.unique()
        # for i, the_group in enumerate(unique_groups):
        #     result = result + self.compute_mean_gap(outputs[group == the_group], outputs)
        result = result + self.compute_mean_gap(outputs[group == -1], outputs[group == 1])
        #import pdb; pdb.set_trace()
        return result

    def compute_mean_gap(self, x, y):
        return (x.mean() - y.mean()) ** 2




def fair_train_single_epoch3_ff(
    epoch, model, train_loader, optimizer, device, loss_function="cross_entropy", loss_mean=False, mu=1, slack=0
):
    """
    Util method for training a model for a single epoch.
    """
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0
    #mu = 2
    #slack = 0
    
    logits = []
    labels = []
    sensitives = []

    for batch_idx, (data, labels, s) in enumerate(train_loader):
        data = data.to(device)
        # print(data.shape)
        # exit()
        labels = labels.to(device)
        s = s.to(device)
        optimizer.zero_grad()

        logits = model(data)
        CE_loss = loss_function_dict[loss_function](logits, labels)

        if loss_mean:
            loss = loss / len(data)

        softmax_prob = F.softmax(logits, dim=1)

        ddp_score = ddp(s, logits)
        #import pdb; pdb.set_trace()
        loss = CE_loss + mu * (torch.mean(ddp_score) - slack)
        if torch.isnan(loss).any():
            import pdb; pdb.set_trace()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader) * len(data),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / num_samples))
    return train_loss / num_samples



def test_single_epoch(epoch, model, test_val_loader, device, loss_function="cross_entropy"):
    """
    Util method for testing a model for a single epoch.
    """
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for data, labels, _, _, _ in test_val_loader:
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += loss_function_dict[loss_function](logits, labels).item()
            num_samples += len(data)

    print("======> Test set loss: {:.4f}".format(loss / num_samples))
    return loss / num_samples

def get_test_logits(model, test_val_loader, device, loss_function="cross_entropy"):

    model.eval()
    loss = 0
    num_samples = 0
    all_logits = []
    with torch.no_grad():
        for data, labels, _, _, _ in test_val_loader:
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            for l in logits:
                all_logits.append(l)
            #loss += loss_function_dict[loss_function](logits, labels).item()
            #num_samples += len(data)
    return all_logits

def shannon_entropy(label_probabilities):
    # Ensure label_probabilities is a PyTorch tensor
    label_probabilities = torch.tensor(label_probabilities)

    # Calculate entropy using the softmax function and negative log
    entropy = -torch.sum(label_probabilities * torch.log2(label_probabilities))

    return entropy.item()

def get_top_k_uncertain_indices(logits, k):
    # Apply softmax separately to each set of label probabilities
    probabilities = [F.softmax(l, dim=-1) for l in logits]

    # Calculate Shannon entropy for each set of label probabilities
    entropies = torch.tensor([shannon_entropy(prob) for prob in probabilities])

    # Sort indices based on entropy in ascending order
    sorted_indices = torch.argsort(entropies, descending=True)

    # Get the top k most uncertain indices
    top_k_indices = sorted_indices[:k]
    
    #sorted_tensor, indices = torch.sort(entropies, descending=True)
    #print(sorted_tensor, indices, top_k_indices)
    #exit()
    return top_k_indices.tolist()


