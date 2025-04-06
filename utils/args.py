

import argparse




def al_args():
    model_name = "resnet18"
    trained_model_name = "resnet18_sn"
    saved_model_path = "./"
    saved_model_name = "resnet18_sn_3.0_50.model"
    dataset_root = "./"
    dataset_name ="CelebA"
    threshold = 1.0
    subsample = 1000
    al_acquisition = "softmax"


    MU = 0.7
    slack = 0.1
    LAM = 100

    lr = 0.1
    weight_decay = 5e-4

    num_initial_samples = 100
    max_training_samples = 200
    acquisition_batch_size = 50
    epochs = 20

    train_batch_size = 64
    test_batch_size = 512
    scoring_batch_size = 128

    parser = argparse.ArgumentParser(description="Active Learning Experiments")
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")
    parser.add_argument(
        "--model", type=str, default=model_name, dest="model_name", help="Model to train",
    )
    # Arguments for MU, slack, and LAM with defaults
    parser.add_argument("--MU", type=float, default=0.7, dest="MU", help="Mu parameter for fairness")
    parser.add_argument("--slack", type=float, default=0.1, dest="slack", help="Slack parameter for fairness constraint")
    parser.add_argument("--LAM", type=float, default=1.0, dest="LAM", help="Lambda parameter for objective function")

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=dataset_root,
        dest="dataset_root",
        help="path of a dataset (useful for ambiguous mnist)",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        default=trained_model_name,
        dest="trained_model_name",
        help="Trained model to check entropy of acquired samples",
    )

    parser.add_argument(
        "-tsn", action="store_true", dest="tsn", help="whether to use spectral normalisation",
    )
    parser.set_defaults(tsn=False)
    parser.add_argument(
        "--tcoeff", type=float, default=sn_coeff, dest="tcoeff", help="Coeff parameter for spectral normalisation",
    )
    parser.add_argument(
        "-tmod", action="store_true", dest="tmod", help="whether to use architectural modifications during training",
    )
    parser.set_defaults(tmod=False)

    parser.add_argument(
        "--saved-model-path",
        type=str,
        default=saved_model_path,
        dest="saved_model_path",
        help="Path of pretrained model",
    )
    parser.add_argument(
        "--saved-model-name",
        type=str,
        default=saved_model_name,
        dest="saved_model_name",
        help="File name of pretrained model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=threshold,
        dest="threshold",
        help="Entropy threshold to decide if a sample is ambiguous or not",
    )

    parser.add_argument(
        "-sn", action="store_true", dest="sn", help="whether to use spectral normalisation during training",
    )
    parser.set_defaults(sn=False)
    parser.add_argument(
        "--coeff", type=float, default=sn_coeff, dest="coeff", help="Coeff parameter for spectral normalisation",
    )
    parser.add_argument(
        "-mod", action="store_true", dest="mod", help="whether to use architectural modifications during training",
    )
    parser.set_defaults(mod=False)


    parser.add_argument("-mi", action="store_true", dest="mi", help="Use MI as acquisition function")
    parser.set_defaults(mi=False)

    parser.add_argument(
        "--num-initial-samples",
        type=int,
        default=num_initial_samples,
        dest="num_initial_samples",
        help="Initial number of samples in the training set",
    )
    parser.add_argument(
        "--max-training-samples",
        type=int,
        default=max_training_samples,
        dest="max_training_samples",
        help="Maximum training set size",
    )
    parser.add_argument(
        "--acquisition-batch-size",
        type=int,
        default=acquisition_batch_size,
        dest="acquisition_batch_size",
        help="Number of samples to acquire in each acquisition step",
    )

    parser.add_argument(
        "--epochs", type=int, default=epochs, dest="epochs", help="Number of epochs to train after each acquisition",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=train_batch_size, dest="train_batch_size", help="Training batch size",
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=test_batch_size, dest="test_batch_size", help="Test batch size",
    )
    parser.add_argument(
        "--scoring_batch_size",
        type=int,
        default=scoring_batch_size,
        dest="scoring_batch_size",
        help="Batch size for scoring the pool dataset",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=dataset_name,
        choices=["RotatedColoredMNIST", "FairFace", "CelebA", "FFHQ", "NYSF"],
        dest="dataset_name",
        help="Dataset to use for training",
    )

    return parser
