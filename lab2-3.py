import time
import tracemalloc
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

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

def analyze_heap_sort():
    datasets, size = generate_datasets()
    results = []
    
    for dataset_name, dataset_list in datasets.items():
        for data in dataset_list:
            arr = data[:]
            tracemalloc.start()
            start_time = time.perf_counter()
            heap_sort(arr)
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
    plt.title("HeapSort Performance Analysis")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Run analysis
df_results = analyze_heap_sort()
plot_results(df_results)
