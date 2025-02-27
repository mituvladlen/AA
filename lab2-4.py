import time
import tracemalloc
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def block_sort(arr):
    max_val = max(arr)
    size = len(arr)
    block_size = int(np.sqrt(size))
    blocks = [[] for _ in range(int(max_val // block_size) + 1)]
    
    for num in arr:
        blocks[int(num // block_size)].append(num)
        
    
    for block in blocks:
        block.sort()
    
    sorted_arr = []
    for block in blocks:
        sorted_arr.extend(block)
    
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

def analyze_block_sort():
    datasets, size = generate_datasets()
    results = []
    
    for dataset_name, dataset_list in datasets.items():
        for data in dataset_list:
            arr = data[:]
            tracemalloc.start()
            start_time = time.perf_counter()
            block_sort(arr)
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
    plt.title("BlockSort Performance Analysis")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Run analysis
df_results = analyze_block_sort()
plot_results(df_results)