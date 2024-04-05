import numpy as np
import sys


def LU(A, n):
    L = np.eye(n, dtype=float)
    U = A.copy()
    swaps = []
    
    for k in range(n):
        # Поиск максимального элемента в текущем столбце
        max_index = np.argmax(abs(U[k:, k])) + k
        if k != max_index:
            U[[k, max_index]] = U[[max_index, k]]
            L[[k, max_index], :k] = L[[max_index, k], :k]  # Обменять только части L
            swaps.append((k, max_index))
        
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            
    return L, U, swaps

def solve(L, U, b, swaps, n):
    # Применяем перестановки к вектору b
    for i, j in swaps:
        b[i], b[j] = b[j], b[i]
    
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        
    return x

def find_det(U, swaps, n):
    det = np.prod(np.diag(U)) * (-1)**len(swaps)
    return det

def invert(L, U, swaps, n):
    A_inv = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        A_inv[:, i] = solve(L, U, e, swaps, n)
    return A_inv

def main():
    n = int(input())
    A = np.array([list(map(float, input().split())) for _ in range(n)])
    b = np.array(list(map(float, input().split())), dtype=float)
    
    L, U, swaps = LU(A, n)
    x = solve(L, U, b, swaps, n)
    det = find_det(U, swaps, n)
    A_inv = invert(L, U, swaps, n)
    
    file_name = sys.argv[1]
    output_file_name = file_name.replace(".txt", "_answer.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(f"L:\n{L}\n")
        output_file.write(f"U:\n{U}\n")
        output_file.write(f"Solution: {x}\n")
        output_file.write(f"Det: {det}\n")
        output_file.write(f"Invert matrix:\n{A_inv}\n")
        
    print("Результаты записаны в файл:", output_file_name)

main()
