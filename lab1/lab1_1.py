import numpy as np
import sys


def LU(A, n):
    L = np.zeros((n, n))
    U = A
    
    for i in range(n):
        for j in range(i, n):
            if U[i][i] != 0:
                L[j][i] = U[j][i] / U[i][i]

    for k in range(1, n):
        for i in range(k - 1, n):
            for j in range(i, n):
                if U[i][i] != 0:
                    L[j][i] = U[j][i] / U[i][i]
                
        for i in range(k, n):
            for j in range(k - 1, n):
                U[i][j] = U[i][j] - L[i][k-1] * U[k-1][j]

    return L, U

def solve(L, U, b, n):
    x = np.ndarray(n)
    y = np.ndarray(n)

    for i in range(n):
        y_num = b[i]
        for j in range(i):
            y_num -= y[j] * L[i, j]
        y[i] = y_num

    for i in range(n - 1, -1, -1):
        if U[i][i] == 0:
            continue
        x_num = y[i]
        for j in range(n-1, i, -1):
            x_num -= x[j] * U[i][j]
        x[i] = x_num / U[i][i]
        
    return x

def find_det(U, n):
    det = 1

    for i in range(n):
        det *= U[i][i]

    return det

def invert(L, U, n):
    A_invert = np.ndarray((n, n))

    for i in range(n):
        b = np.zeros(n)
        b[i] = 1.
        x = solve(L, U, b, n)
        
        for j in range(n):
            A_invert[j, i] = x[j]

    return A_invert


def main():
    n = int(input())
    A = np.array([list(map(int, input().split())) for _ in range(n)], dtype=float)
    b = np.array(list(map(int, input().split())))
    
    L, U = LU(A, n)
    x = solve(L, U, b, n)
    det = find_det(A, n)
    A_invert = invert(L, U, n)

    file_name = sys.argv[1]

    output_file_name = file_name.replace(".txt", "_answer.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(f"L:\n {L}\n")
        output_file.write(f"U:\n {U}\n")
        output_file.write(f"L * U:\n {np.dot(L, U)}\n")
        output_file.write(f"Solution: {x}\n")
        output_file.write(f"Det: {det}\n")
        output_file.write(f"Invert matrix:\n {A_invert}\n")

    print("Результаты записаны в файл:", output_file_name)

main()