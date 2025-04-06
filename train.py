
import torch



# Import data utilities
import torch.utils.data as data
import data.active_learning.active_learningFF as active_learning

import datasets
import torch.distributions.bernoulli as bernoulli
from torch.nn import functional as F
from torch.utils.data import random_split
# Import network architectures
from net.resnet import resnet18

# Import train and test utils
from utils.train_utils import fair_train_single_epoch3_ff

# Importing uncertainty metrics
from metrics.uncertainty_confidence import logsumexp
from metrics.classification_metrics_custom import test_classification_net, get_logits_labels


# Importing args
from utils.args import al_args

# Importing GMM utilities
from utils.gmm_utils_ff import get_embeddings, gmm_evaluate, gmm_fit
import os

# Mapping model name to model function
models = {"resnet18": resnet18}






def log_class_probs(data_loader):
    num_classes = 2
    class_n = len(data_loader.dataset)
    class_count = torch.zeros(num_classes)
    for data, label,_ in data_loader:
        class_count += torch.Tensor([torch.sum(label == c) for c in range(num_classes)])

    class_prob = class_count / class_n
    log_class_prob = torch.log(class_prob)
    return log_class_prob

def log_class_sensitive_probs(data_loader, num_classes, num_sensitives):
    num_components = num_classes * num_sensitives
    class_n = len(data_loader.dataset)
    class_count = torch.zeros(num_components)

    
    for data, label,sensitive in data_loader:
        component_counts_in_batch = []
        for c in range(num_classes):
            for s in [0, 1]:
                label_condition = label == c
                sensitive_condition = sensitive == s
                combined_condition = label_condition & sensitive_condition
                component_counts_in_batch.append(torch.sum(combined_condition))
        class_count += torch.Tensor(component_counts_in_batch)


    class_prob = class_count / class_n
    log_class_prob = torch.log(class_prob)

    return log_class_prob


def get_probabilities(logits):
    return logsumexp(logits)

def select_candidates(scores, budget):
    # Sort scores in ascending order
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_scores = scores[sorted_indices]

    sorted_indices = sorted_indices.tolist()
    sorted_scores = sorted_scores.tolist()

    # Initialize arrays
    candidate_indices = []
    candidate_scores = []

    # Use scores as probabilities for Bernoulli trials
    for i in range(len(scores)):
        trial = bernoulli.Bernoulli(sorted_scores[i]).sample().item()
        if trial == 1:
            candidate_scores.append(sorted_scores[i])
            candidate_indices.append(sorted_indices[i])

        if len(candidate_indices) >= budget:
            break

    return candidate_scores, candidate_indices

def normalize_scores(scores):
    """
    Normalize a tensor of scores to the range [0, 1] using min-max normalization.

    Parameters:
    - scores (torch.Tensor): Input tensor of scores.

    Returns:
    - torch.Tensor: Normalized scores in the range [0, 1].
    """
    min_value = scores.min()
    max_value = scores.max()

    normalized_scores = (scores - min_value) / (max_value - min_value)

    return normalized_scores



def compute_logsumexp_density(logits, log_class_probs):
    return logsumexp(logits+log_class_probs)

from metrics.uncertainty_confidence import entropy, logsumexp


if __name__ == "__main__":


    args = al_args().parse_args()

    MU = args.MU
    slack = args.slack
    LAM = args.LAM
    highq = True



    print(args)

    
    pretext = '{}_highq_slack_{}_RUN1_mu{}'.format(str(highq), slack, MU)

    cuda = torch.cuda.is_available()


    torch.manual_seed(args.seed)
    device = torch.device("cuda" if cuda else "cpu")

    model_fn = models[args.model_name]

    # Load pretrained network for checking ambiguous samples


    # Creating the datasets
    num_classes = 2
    num_sensitives = 2
    dataset = vars(datasets)[args.dataset_name]
    overall_acc = []
    overall_fair_mutinfo = []
    overall_fair_dpd = []
    overall_fair_eod = []

    torch.manual_seed(args.seed)


    final_dataset = []

    for task_i, env_i in enumerate(dataset):
        total_length = len(env_i)
        print("Total Length of domain {}: {}".format(task_i, total_length))

        t1_size = total_length // 3
        t2_size = total_length // 3
        t3_size = total_length - t1_size - t2_size

        t1, t2, t3 = random_split(env_i, [t1_size, t2_size, t3_size])

        print("New dataset sizes: ",len(t1), len(t2), len(t3))

        final_dataset.append(t1)
        final_dataset.append(t2)
        final_dataset.append(t3)

    #import pdb; pdb.set_trace()

    for task_i, env_i in enumerate(final_dataset):
        
        num_acquired = 0
        current_dataset = env_i

        new_dataset = current_dataset



        kwargs = {"num_workers": 0, "pin_memory": False} if cuda else {}

        # Run experiment
        num_runs = 1
        test_accs = {}
        ambiguous_dict = {}
        ambiguous_entropies_dict = {}

        for i in range(num_runs):
            test_accs[i] = []
            ambiguous_dict[i] = []
            ambiguous_entropies_dict[i] = {}

        
        if task_i == 0:
            active_learning_data = active_learning.ActiveLearningData(new_dataset, None)

        else:
            active_learning_data = active_learning.ActiveLearningData(new_dataset, prev_dataset)

        # Acquiring the first training dataset from the total pool. This is random acquisition
        if task_i == 0: 
            initial_sample_indices = active_learning.get_sensitive_balanced_sample_indices(
            new_dataset, num_classes=num_classes, num_sensitives=num_sensitives, n_per_digit=int(args.num_initial_samples / (num_classes*num_sensitives)),
            )
            active_learning_data.acquire(initial_sample_indices)

        # Train loader for the current acquired training set
        sampler = active_learning.RandomFixedLengthSampler(
            dataset=active_learning_data.training_dataset, target_length=5056
        )
        train_loader = torch.utils.data.DataLoader(
            active_learning_data.training_dataset, sampler=sampler, batch_size=args.train_batch_size, **kwargs,
        )

        small_train_loader = torch.utils.data.DataLoader(
            active_learning_data.training_dataset, shuffle=True, batch_size=args.train_batch_size, **kwargs,
        )

        # Pool loader for the current acquired training set
        pool_loader = torch.utils.data.DataLoader(
            active_learning_data.pool_dataset, batch_size=args.scoring_batch_size, shuffle=False, **kwargs,
        )

        if task_i == 0:
            #if first task, we acquired some initial samples so can't use the whole train set at the begininng to test
            test_loader = torch.utils.data.DataLoader(active_learning_data.pool_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        else:
            test_loader = torch.utils.data.DataLoader(new_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        # Run active learning iterations
        active_learning_iteration = 0
        task_accs = []
        task_fairs_mutinfo = []
        task_fairs_dpd = []
        task_fairs_eod= []


        acc = []
        fairmutinfo = []
        fairdpd = []
        faireod = []

        while True:


            print("Task ", task_i, "====================================")
            print("Active Learning Iteration: " + str(active_learning_iteration) + " ================================>")

            lr = args.lr
            weight_decay = args.weight_decay

            model = model_fn(spectral_normalization=args.sn, mod=args.mod, mnist=False).to(device=device)

            optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
            model.train()

            # Train
            print("Length of train dataset: " + str(len(small_train_loader.dataset)))
            if (active_learning_iteration== 0): print(model)
            best_model = None
            #best_val_accuracy = 0
            for epoch in range(args.epochs):
                fair_train_single_epoch3_ff(epoch, model, small_train_loader, optimizer, device, mu=MU, slack=slack)



            # Fit the GMM on the trained model
            model.eval()
            embeddings, labels, sensitives = get_embeddings(
                model, small_train_loader, num_dim=512, dtype=torch.double, device="cuda", storage_device="cuda",
            )
            gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes, sensitives=sensitives)
            print("Training ended")




            print("Testing the model")
            (conf_matrix, accuracy, mutinfo, dpd, eod, labels_list, predictions, confidences,) = test_classification_net(
                model, test_loader, device=device, is_val=False
            )
            percentage_correct = 100.0 * accuracy
            print(conf_matrix)
            #test_accs[run].append(percentage_correct)

            acc.append(percentage_correct)
            fairmutinfo.append(mutinfo)
            fairdpd.append(dpd)
            faireod.append(eod)


            print(num_acquired)

            if num_acquired >= args.max_training_samples:
                break

            print("Test set: Accuracy: ({:.2f}%)".format(percentage_correct))

            print(active_learning_iteration, args.max_training_samples)

            # Acquisition phase
            N = len(active_learning_data.pool_dataset)

            print("Acquisition ========================================")

            
            model.eval()

            log_class_prob = log_class_sensitive_probs(small_train_loader, num_classes=num_classes, num_sensitives=num_sensitives)
            import pdb; pdb.set_trace()
            logits, labels = gmm_evaluate(
                model,
                gaussians_model,
                pool_loader,
                device=device,
                num_classes=num_classes,
                num_sensitives=num_sensitives,
                storage_device="cpu",
            )



            qz1ya = logits

            qz1y0a0 = qz1ya[:,0]
            qz1y0a1 = qz1ya[:,1]
            qz1y1a0 = qz1ya[:,2]
            qz1y1a1 = qz1ya[:,3]
            q1= torch.abs(qz1y0a0 - qz1y0a1)
            q2 = torch.abs(qz1y1a0 - qz1y1a1)


            tlogits, tlabels, tsensitives = get_logits_labels(model, pool_loader, device)

            softmax_prob = F.softmax(tlogits, dim=1)

            probabilities = softmax_prob.cpu()

            total_q = probabilities[:,0]*q1+ probabilities[:, 1]*q2

            uncertainty = compute_logsumexp_density(logits, log_class_prob)


            mean_uncertainty = uncertainty.mean(dim=0)
            std_uncertainty = uncertainty.std(dim=0)

            mean_total_q = total_q.mean(dim=0)
            std_total_q = total_q.std(dim=0)

            standardized_uncertainty = (uncertainty-mean_uncertainty)/(std_uncertainty)
            standardized_total_q = (total_q - mean_total_q)/(std_total_q)
            

            if highq:
                sign = -1
            else:
                sign = 1

            objective = standardized_uncertainty + (LAM*sign*standardized_total_q)

            probability_scores = 1 - normalize_scores(objective)



            

            (candidate_scores_q, candidate_indices,) = active_learning.get_top_k_scorers(
                objective, args.acquisition_batch_size, uncertainty=False,
            )
            
            num_acquired += len(candidate_indices)





            active_learning_data.acquire(candidate_indices)

            active_learning_iteration += 1


        prev_dataset = small_train_loader.dataset

        task_accs.append(acc[1:])
        task_fairs_mutinfo.append(fairmutinfo[1:])
        task_fairs_eod.append(faireod[1:])
        task_fairs_dpd.append(fairdpd[1:])

        
        
        
        overall_acc.append(acc[0])
        overall_fair_mutinfo.append(fairmutinfo[0])
        overall_fair_eod.append(faireod[0])
        overall_fair_dpd.append(fairdpd[0])





    results_folder =  "task-scores"
    results_dest = os.path.join(os.getcwd(),results_folder)
    final_results_dest = os.path.join(results_dest, pretext+"LAM{}ABS{}.txt".format(LAM, args.acquisition_batch_size))
    
    with open(final_results_dest, "w") as file:
        file.write(str(overall_acc))
        file.write("\n")
        file.write(str(overall_fair_mutinfo))
        file.write("\n")
        file.write(str(overall_fair_eod))
        file.write("\n")
        file.write(str(overall_fair_dpd))
        file.write("\n")




