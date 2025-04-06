import torch
from torch import nn
from tqdm import tqdm

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(
    net, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device,
):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    sensitives = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label, sensitive in tqdm(loader):
            data = data.to(device)
            label = label.to(device)
            sensitive = sensitive.to(device)

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
                out = net.module.feature
            else:
                out = net(data)
                out = net.feature

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            sensitives[start:end].copy_(sensitive, non_blocking=True)
            start = end

    return embeddings, labels, sensitives


def gmm_forward(net, gaussians_model, data_B_X):

    if isinstance(net, nn.DataParallel):
        features_B_Z = net.module(data_B_X)
        features_B_Z = net.module.feature
    else:
        features_B_Z = net(data_B_X)
        features_B_Z = net.feature

    log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])

    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, num_sensitives, storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes*num_sensitives), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label, _,in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            logit_B_C = gmm_forward(net, gaussians_model, data)

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N


def gmm_get_logits(gmm, embeddings):

    log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
    return log_probs_B_Y


def gmm_fit(embeddings, labels, num_classes, sensitives):
    #import pdb; pdb.set_trace()
    #print(torch.unique(sensitives))
    #import pdb; pdb.set_trace()
    with torch.no_grad():
        mean_values = [] 
        class_and_sensitive_wise_mean_dict = {}
        for c in range(num_classes):
            for s in torch.unique(sensitives):
                #print(c, s.item())
                label_condition = labels == c
                sensitive_condition = sensitives == s
                combined_condition = label_condition & sensitive_condition
                #print(label_condition, sensitive_condition, combined_condition)
                mean_features = torch.mean(embeddings[combined_condition], dim=0)
                #print(mean_features)
                mean_values.append(mean_features)
                class_and_sensitive_wise_mean_dict[(c, s.item())] = mean_features
        
        class_and_sensitive_wise_mean_features = torch.stack(mean_values)
        #print(class_and_sensitive_wise_mean_dict.keys())
        cov_values = []
        for c in range(num_classes):
            for s in torch.unique(sensitives):
                label_condition = labels == c
                sensitive_condition = sensitives == s
                combined_condition = label_condition & sensitive_condition

                cov_features = centered_cov_torch(embeddings[combined_condition] - class_and_sensitive_wise_mean_dict[(c, s.item())])
                cov_values.append(cov_features)
        
        class_and_sensitive_wise_cov_features = torch.stack(cov_values)
                


        #classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])



        # classwise_cov_features = torch.stack(
        #     [centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) for c in range(num_classes)]
        # )

    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    class_and_sensitive_wise_cov_features.shape[1], device=class_and_sensitive_wise_cov_features.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=class_and_sensitive_wise_mean_features, covariance_matrix=(class_and_sensitive_wise_cov_features + jitter),
                )
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "The parameter covariance_matrix has invalid values" in str(e):
                    continue
                else:
                    continue
            break

    return gmm, jitter_eps
