import matplotlib.pyplot as plt
import numpy as np
import time
import os
from pathlib import Path

def plot_times(time_stamps, codec = None, title: str = "Messung", save_folder: str = "plots"):
    times_sorted = []
    for i in range(len(codec)):
        sorted = []
        for t in time_stamps:
            sorted.append((t[codec[i][1]] - t[codec[i][2]])*1000)
        sorted = np.array(sorted)
        times_sorted.append(sorted)
    #for t in times_sorted:
    #    print(len(t))
    labels = [c[0] for c in codec]
    
    plt.figure(figsize=(12,6))
    plt.boxplot(times_sorted, showfliers=True, tick_labels=labels, vert=True, patch_artist=True, showmeans=True, meanline=True)
    plt.title(title)
    plt.ylabel("Zeit pro Frame [ms]")
    plt.xlabel("Verteilung der Prozess-Zeiten")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.rcParams["savefig.directory"] = ""
    #plt.savefig(f"{save_folder}/{title}.png")
    plt.show()

    plt.figure(figsize=(12,6))
    plt.boxplot(times_sorted, showfliers=False, tick_labels=labels, vert=True, patch_artist=True, showmeans=True, meanline=True)
    plt.title(title)
    plt.ylabel("Zeit pro Frame [ms]")
    plt.xlabel("Verteilung der Prozess-Zeiten")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.rcParams["savefig.directory"] = ""
    #plt.savefig(f"{save_folder}/{title}.png")
    plt.show()

    
    plt.figure(figsize=(12,6))
    plt.plot(times_sorted[7], marker='.', linestyle='-')
    plt.xlabel("Frame-Index")
    plt.ylabel("Zeit pro Frame [ms]")
    plt.title("Ball-Detection Zeiten")
    plt.show()

        

