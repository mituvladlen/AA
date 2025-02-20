import time
import math
import sys
from functools import lru_cache
import numpy as np
from decimal import Decimal, getcontext

sys.setrecursionlimit(20000)  # Adjust recursion limit to prevent RecursionError
getcontext().prec = 100  # Increase precision for Binet formula

@lru_cache(None)
def fib_recursive(n):
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
        return np.dot(A, B)
    
    def matrix_exponentiation(F, n):
        result = np.identity(2, dtype=int)
        while n:
            if n % 2:
                result = multiply_matrices(result, F)
            F = multiply_matrices(F, F)
            n //= 2
        return result
    
    if n == 0:
        return 0
    F = np.array([[1, 1], [1, 0]], dtype=int)
    result = matrix_exponentiation(F, n - 1)
    return result[0][0]

def fib_binet(n):
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

def analyze_fibonacci(methods, n):
    results = {}
    for name, func in methods.items():
        try:
            start_time = time.time()
            result = func(n)
            end_time = time.time()
            results[name] = (result, end_time - start_time)
        except ValueError as e:
            results[name] = (None, None)  # Skip execution time for invalid cases
    return results

methods = {
    "Recursive": fib_recursive,
    "Dynamic Programming": fib_dynamic,
    "Matrix Power": fib_matrix,
    "Binet Formula": fib_binet,
    "Iterative": fib_iterative,
    "Generator": fib_generator,
}

n = int(input("Enter a value for Fibonacci calculation: "))
results = analyze_fibonacci(methods, n)

for name, (value, time_taken) in results.items():
    print(f"{name}: Fibonacci({n}) = {value}, Time: {time_taken:.6f} seconds")
