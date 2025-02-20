import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from decimal import Decimal, getcontext

sys.setrecursionlimit(20000)  # Increase recursion limit

@lru_cache(None)
def fib_recursive(n):
    if n > 50:  # Recursive method is too slow for large n
        raise ValueError("Recursive method is too slow for n > 30")
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

def fib_dynamic(n):
    if n <= 1:
        return n
    dp = [0, 1] + [0] * (n - 1)
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

def fib_matrix(n):
    def multiply_matrices(A, B):
        return np.dot(A, B).astype(object)
    
    def matrix_exponentiation(F, n):
        result = np.identity(2, dtype=object)
        while n:
            if n % 2:
                result = multiply_matrices(result, F)
            F = multiply_matrices(F, F)
            n //= 2
        return result
    
    if n == 0:
        return 0
    F = np.array([[1, 1], [1, 0]], dtype=object)
    result = matrix_exponentiation(F, n - 1)
    return result[0][0]

def fib_binet(n):
    getcontext().prec = n + 100
    sqrt_5 = Decimal(5).sqrt()
    phi = (Decimal(1) + sqrt_5) / Decimal(2)
    return round((phi**n - (-1 / phi)**n) / sqrt_5)

def fib_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def fib_generator(n):
    def fibonacci_gen():
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b
    
    gen = fibonacci_gen()
    for _ in range(n + 1):
        result = next(gen)
    return result

def analyze_fibonacci_for_graph(methods, n_values):
    results = {name: [] for name in methods}
    
    for n in n_values:
        for name, func in methods.items():
            try:
                start_time = time.time()
                func(n)
                end_time = time.time()
                results[name].append(end_time - start_time)
            except Exception:
                results[name].append(None)  # Skip execution time for failed cases

    return results

def analyze_fibonacci(methods, n):
    results = {}
    for name, func in methods.items():
        try:
            start_time = time.time()
            result = func(n)
            end_time = time.time()
            results[name] = (result, end_time - start_time)
        except Exception as e:
            results[name] = (None, None)  # Store None instead of a string to avoid format error
    return results

methods = {
    "Recursive": fib_recursive,
    "Dynamic Programming": fib_dynamic,
    "Matrix Power": fib_matrix,
    "Binet Formula": fib_binet,
    "Iterative": fib_iterative,
    "Generator": fib_generator,
}

small_n_values = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45]
large_n_values = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849]

choice = input("Choose dataset (small or large): ").strip().lower()
if choice == "small":
    n_values = small_n_values
elif choice == "large":
    n_values = large_n_values
else:
    print("Invalid choice. Defaulting to small values.")
    n_values = small_n_values

results = analyze_fibonacci(methods, n_values[0])

for name, (value, time_taken) in results.items():
    if time_taken is None:
        print(f"{name}: Fibonacci({n_values[0]}) = ERROR (Method failed)")
    else:
        print(f"{name}: Fibonacci({n_values[0]}) = {str(value)[:50]}..., Time: {time_taken:.6f} seconds")

# Generate performance graph
graph_results = analyze_fibonacci_for_graph(methods, n_values)

plt.figure(figsize=(10, 6))

for name, times in graph_results.items():
    plt.plot(n_values, [t if t is not None else float('nan') for t in times], label=name, marker='o')

plt.xlabel("n (Fibonacci Number Index)")
plt.ylabel("Time (seconds)")
plt.title("Fibonacci Computation Time Comparison")
plt.legend()
plt.yscale("log")  # Use log scale for better visualization of large differences
plt.grid()
plt.show()
