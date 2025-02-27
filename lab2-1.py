import time
import tracemalloc
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def generate_datasets():
    size = 10000
    datasets = {
        "Random Large": [random.sample(range(1, 1000000), size)],
        "Nearly Sorted": [sorted(random.sample(range(1, 1000000), size))],
        "Small": [[random.randint(1, 100) for _ in range(20)]],
        "Integer Limited Range": [[random.randint(1, 100) for _ in range(size)]],
        "Floating Point": [[random.uniform(1.0, 1000.0) for _ in range(size)]],
    }
    return datasets, size

def analyze_quicksort():
    datasets, size = generate_datasets()
    results = []
    
    for dataset_name, dataset_list in datasets.items():
        for data in dataset_list:
            arr = data[:]
            tracemalloc.start()
            start_time = time.perf_counter()
            quicksort(arr)
            end_time = time.perf_counter()
            memory_used = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            
            results.append([dataset_name, size, end_time - start_time, memory_used])
    
    df = pd.DataFrame(results, columns=["Dataset Type", "Size", "Time (s)", "Memory (bytes)"])
    print(df)
    return df

def plot_results(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df["Dataset Type"], df["Time (s)"], color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.xlabel("Dataset Type")
    plt.ylabel("Time (seconds)")
    plt.title("QuickSort Performance Analysis")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Run analysis
df_results = analyze_quicksort()
plot_results(df_results)
