import time
import tracemalloc
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    sorted_arr = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sorted_arr.append(left[i])
            i += 1
        else:
            sorted_arr.append(right[j])
            j += 1
    sorted_arr.extend(left[i:])
    sorted_arr.extend(right[j:])
    return sorted_arr

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

def analyze_merge_sort():
    datasets, size = generate_datasets()
    results = []
    
    for dataset_name, dataset_list in datasets.items():
        for data in dataset_list:
            arr = data[:]
            tracemalloc.start()
            start_time = time.perf_counter()
            merge_sort(arr)
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
    plt.title("MergeSort Performance Analysis")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Run analysis
df_results = analyze_merge_sort()
plot_results(df_results)
