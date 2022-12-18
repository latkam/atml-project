import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

PATH_CHECKPOINTS = "./checkpoints"
PATH_LOGS = "./logs"
PATH_PLOTS = "./plots"
MODEL_NAME = "model.th"
NUM_EPOCHS = 160

experiments_hr = {
    "full": "All Params Trainable",
    "nobn": "Everything except BatchNorm",
    "bn": "Only BatchNorm",
    "random": "2 random conv features per channel"
}

experiments_hist_colours = {
    "full": "b",
    "nobn": "g",
    "bn": "r"
}


def plots_accuracy(models, experiments):
    accuracies = {experiment: [] for experiment in experiments}
    xticks = list(range(len(models)))
    with tqdm(range(len(models) * len(experiments)), desc="Accuracy plots") as pbar:
        for model in models:
            for experiment in experiments:
                with open(os.path.join(PATH_LOGS, f"log-{model}-{experiment}.txt"), "r") as file:
                    txt = file.readlines()[-5]
                    acc = float(re.findall("\d+\.\d+", txt)[-1])
                    accuracies[experiment].append(acc)
                pbar.update()
    fig = plt.figure()
    for experiment, accs in accuracies.items():
        plt.plot(xticks, accs, label=experiments_hr[experiment])
    plt.title("Accuracies")
    plt.legend()
    plt.xticks(ticks=xticks, labels=models)
    plt.xlabel("Models")
    plt.ylabel("Accuracy [%]")
    plt.savefig(os.path.join(PATH_PLOTS, "accuracies.png"))


def plot_accuracy_evolution(models, experiments):
    values = {}
    with tqdm(range(len(models) * len(experiments)), desc="Accuracy evolution") as pbar:
        for model in models:
            for experiment in experiments:
                values[experiment] = []
                with open(os.path.join(PATH_LOGS, f"log-{model}-{experiment}.txt"), "r") as file:
                    lines = file.readlines()
                    lines = [line for line in lines if "* Prec@1" in line]
                    for line in lines:
                        found = re.findall("\d+\.\d+", line)
                        if len(found) > 0:
                            values[experiment].append(float(found[-1]))
                        else:
                            values[experiment].append(None)
                pbar.update()
            fig = plt.figure()
            xticks = list(range(NUM_EPOCHS))
            for experiment, accs in values.items():
                experiment_hr = experiments_hr[experiment]
                plt.plot(xticks, accs, label=experiment_hr)
            plt.title(f"Accuracies during training - {model}")
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy [%]")
            plt.savefig(os.path.join(PATH_PLOTS, f"accuracies-evolution-{model}.png"))
                    


def plots_gamma_distro(models, experiments):
    with tqdm(range(len(models)), desc="Gamma distributions plots") as pbar:
        for model in models:
            fig = plt.figure()
            for experiment in experiments:
                gammas = []
                params = torch.load(os.path.join(PATH_CHECKPOINTS, f"{model}-{experiment}", MODEL_NAME))["state_dict"]
                for name, param  in params.items():
                    if "bn" in name and "weight" in name:
                        gammas.extend(param.view(-1,).tolist())
                counts, bins = np.histogram(gammas, bins=50, range=(-1.5, 2.5), density=True)
                plt.stairs(counts, bins, color=experiments_hist_colours[experiment], fill=True, alpha=0.5, label=experiments_hr[experiment])
            plt.title(r"Distribution of $\gamma$ - " + model + " for CIFAR10")
            plt.legend()
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.savefig(os.path.join(PATH_PLOTS, f"gamma-{model}.png"))
            pbar.update()
 
def resources_tables(models, experiments):
    def process_gpu_time(lines):
        gpu_time = float(re.findall("\d+\.\d+", lines[-4])[-1])
        return round(gpu_time / 3600, 3)
    
    def process_gpu_memory(lines):
        gpu_memory = int(re.findall("\d+", lines[-3])[-1])
        return round(gpu_memory / 1e3, 3)
    
    def process_total_time(lines):
        total_time = float(re.findall("\d+\.\d+", lines[-2])[-1])
        return round(total_time / 3600, 3)
    
    def process_ram(lines):
        ram = int(re.findall("\d+", lines[-1])[-1])
        return round(ram / 1e9, 3)
    
    res = {
        "gpu_time": "experiment [h];",
        "gpu_memory": "experiment [GB];",
        "total_time": "experiment [h];",
        "ram": "experiment [GB];"
    }
    
    sums = {
        "gpu_time": 0,
        "total_time": 0
    }
    
    for model in models:
        for res_name in res.keys():
            res[res_name] += f"{model};"
    for res_name in res.keys():
        res[res_name] += "\n"
    with tqdm(range(len(experiments)), desc="Resources tables") as pbar:
        for experiment in experiments:
            for res_name in res.keys():
                res[res_name] += f"{experiment};"
            for model in models:
                with open(os.path.join(PATH_LOGS, f"log-{model}-{experiment}.txt"), "r") as file:
                    lines = file.readlines()
                    res["gpu_time"] += f"{process_gpu_time(lines)};"
                    res["gpu_memory"] += f"{process_gpu_memory(lines)};"
                    res["total_time"] += f"{process_total_time(lines)};"
                    res["ram"] += f"{process_ram(lines)};"

                    sums["gpu_time"] += process_gpu_time(lines)
                    sums["total_time"] += process_total_time(lines)
            for res_name in res.keys():
                res[res_name] += "\n"
            pbar.update()
    for res_name, res_val in res.items():
        with open(os.path.join(PATH_PLOTS, f"{res_name}.csv"), "w") as file:
            file.write(res_val)
    with open(os.path.join(PATH_PLOTS, "total.txt"), "w") as file:
        lines = []
        for sums_name, sums_val in sums.items():
            lines.append(f"total {sums_name}: {sums_val}")
        file.writelines(line + "\n" for line in lines)
        

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
        "nobn",
        "bn",
        "random"
    ]
    experiments_reduced = ["full", "bn"]
    plots_accuracy(models, experiments)
    plot_accuracy_evolution(models, experiments)
    plots_gamma_distro(models, experiments_reduced)
    resources_tables(models, experiments)

if __name__ == "__main__":
    main()