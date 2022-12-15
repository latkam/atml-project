import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm

import resnet
from trainer import accuracy, AverageMeter

GPU_NUMBER = 0
device = torch.device(f"cuda:{GPU_NUMBER}" if torch.cuda.is_available() else "cpu")

PATH_CHECKPOINTS = "./checkpoints"
PATH_LOGS = "./logs"
PATH_PLOTS = "./plots"
PATH_DATA = "./data"
MODEL_NAME = "model.th"


experiments_hr = {
    "full": "All Params Trainable",
    #"nobn": "Everything except BatchNorm",
    "bn": "Only BatchNorm",
    #"random": "2 random conv features per channel"
}


def gammas_below_threshold(models, experiments, thresholds):
    with tqdm(range(len(models) * len(experiments) * len(thresholds)), desc="Gammas below threshold") as pbar:
        fractions = {}
        for experiment in experiments:
            for threshold in thresholds:
                fractions[f"{experiment}-{threshold}"] = []
                for model in models:
                    gammas = []
                    params = torch.load(os.path.join(PATH_CHECKPOINTS, f"{model}-{experiment}", MODEL_NAME))["state_dict"]
                    for name, param in params.items():
                        if "bn" in name and "weight" in name:
                            gammas.extend(param.view(-1,).tolist())
                    gammas_below_threshold_cnt = (np.array(np.abs(gammas)) < threshold).sum()
                    fraction = gammas_below_threshold_cnt / len(gammas) * 100
                    fractions[f"{experiment}-{threshold}"].append(fraction)
                    pbar.update()
        fig = plt.figure()
        xticks = list(range(len(models)))
        for name, values in fractions.items():
            experiment = experiments_hr[name.split("-")[0]]
            threshold = name.split("-")[1]
            plt.plot(xticks, values, label=fr"{experiment}, $t$ = {threshold}")
        plt.title(r"Fraction of $\gamma$ parameters for which $\vert \gamma \vert$ is smaller than various thresholds")
        plt.legend()
        plt.xticks(ticks=xticks, labels=models)
        plt.xlabel("Models")
        plt.ylabel(r"Fraction of $\vert \gamma \vert < t$ [%]")
        plt.savefig(os.path.join(PATH_PLOTS, "gammas_below_thresholds.png"))


def accuracy_changes(models, experiments, thresholds):
    with tqdm(range(len(models) * len(experiments)), desc="Changes in accuracy - baselines") as pbar:
        baselines = {}
        accuracies = {}
        
        for experiment in experiments:
            baselines[experiment] = []
            for threshold in thresholds:
                accuracies[f"{experiment}-{threshold}"] = []
        
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        val_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root=PATH_DATA, train=False, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=1, pin_memory=True)
        
        for experiment in experiments:
            for model_name in models:
                model = resnet.__dict__[model_name]()
                model.load_state_dict(torch.load(os.path.join(PATH_CHECKPOINTS, f"{model_name}-{experiment}", MODEL_NAME))["state_dict"])
                model.to(device)

                acc_avg = AverageMeter()
                with torch.no_grad():
                    for i, (input, target) in enumerate(val_loader):
                        target = target.to(device)
                        input_var = input.to(device)

                        output = model(input_var)
                        acc = accuracy(output.data, target)[0]
                        acc_avg.update(acc.item(), input.size(0))
                baselines[experiment].append(acc_avg.avg)
                pbar.update()
    
    with tqdm(range(len(models) * len(experiments) * len(thresholds)), desc="Changes in accuracy - thresholds") as pbar:
        for model_name in models:                    
            for experiment in experiments:
                for threshold in thresholds:
                    model = resnet.__dict__[model_name]()
                    model.load_state_dict(torch.load(os.path.join(PATH_CHECKPOINTS, f"{model_name}-{experiment}", MODEL_NAME))["state_dict"])
                    zero_gammas(model, threshold)
                    model.to(device)

                    acc_avg = AverageMeter()
                    with torch.no_grad():
                        for i, (input, target) in enumerate(val_loader):
                            target = target.to(device)
                            input_var = input.to(device)

                            output = model(input_var)
                            acc = accuracy(output.data, target)[0]
                            acc_avg.update(acc.item(), input.size(0))
                    accuracies[f"{experiment}-{threshold}"].append(acc_avg.avg)
                    pbar.update()

    fig = plt.figure()
    xticks = list(range(len(models)))
    for name, values in accuracies.items():
        experiment = name.split("-")[0]
        experiment_hr = experiments_hr[experiment]
        threshold = name.split("-")[1]
        print("baselines", baselines)
        print("values", values)
        differences = [value - baselines[experiment][i] for i, value in enumerate(values)]
        print("differences", differences)
        plt.plot(xticks, differences, label=fr"{experiment_hr}, $t$ = {threshold}")
    plt.title(r"Accuracy change when setting $\gamma$ to zero if $\vert \gamma \vert < t$")
    plt.legend()
    plt.xticks(ticks=xticks, labels=models)
    plt.xlabel("Models")
    plt.ylabel(r"Difference in accuracy [%]")
    plt.savefig(os.path.join(PATH_PLOTS, "accuracy_changes.png"))
                            

def zero_gammas(model, threshold):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "bn" in name and "weight" in name:
                param.data[param.data < threshold] = torch.zeros_like(param.data[param.data < threshold])



def main():
    os.makedirs(PATH_PLOTS, exist_ok=True)
    
    models = [
        "resnet14",
        "resnet32",
        "resnet56",
        "resnet110",
        "resnet218",
        "resnet434",
        #"resnet866"
    ]
    experiments = [
        "full",
        #"nobn",
        "bn",
        #"random"
    ]
    
    thresholds = [
        0.01,
        0.05,
        0.1
    ]
    
    gammas_below_threshold(models, experiments, thresholds)
    accuracy_changes(models, experiments, thresholds)
    

if __name__ == "__main__":
    main()