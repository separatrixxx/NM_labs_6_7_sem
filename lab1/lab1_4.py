import numpy as np
import sys
import math


def transpouse(mat):
    matrix = []
    for i in range(len(mat[0])):
        matrix.append(list())
        for j in range(len(mat)):
            matrix[i].append(mat[j][i])

    return matrix

def yakobi(A, epsilon, n, iterations):
    A_current = np.array(A)
    U_total = np.eye(n)
    max_element_change = sys.maxsize

    while max_element_change > epsilon:
        max_element_change = 0
        largest_i, largest_j = 0, 0

        for i in range(n):
            for j in range(i):
                if abs(A_current[i][j]) > max_element_change:
                    max_element_change, largest_i, largest_j = abs(A_current[i][j]), i, j

        if max_element_change > epsilon:
            phi = math.atan((2 * A_current[largest_i][largest_j]) / 
                            (A_current[largest_i][largest_i] - A_current[largest_j][largest_j])) / 2
            U = np.eye(n)
            U[largest_i, largest_i] = U[largest_j, largest_j] = math.cos(phi)
            U[largest_i, largest_j], U[largest_j, largest_i] = -math.sin(phi), math.sin(phi)

            A_current = np.round(np.dot(np.dot(U.T, A_current), U), decimals=3)
            U_total = np.dot(U_total, U)
        
        iterations += 1

    eigenvalues = np.diagonal(A_current)
    eigenvectors = [U_total[:, i] for i in range(n)]

    return eigenvalues, eigenvectors, iterations

# проверка !!!

def main():
    e = float(input())
    n = int(input())
    A = np.array([list(map(float, input().split())) for _ in range(n)])

    h, X_x, iter = yakobi(A, e, n, 0)

    file_name = sys.argv[1]

    output_file_name = file_name.replace(".txt", "_answer.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(f"Eigenvalues: {h}\n")
        output_file.write(f"Eigenvectors:\n")
        for i in range(len(X_x)):
            output_file.write(f"x{i + 1} = {X_x[i]}\n")
        output_file.write(f"Number of iterations: {iter}\n")
        output_file.write(f"Dependence of iterations on e: {iter / e}\n")

    print("Результаты записаны в файл:", output_file_name)

main()