import numpy as np
import sys


def chebyshev_norm(vec):
    norm = 0

    for i in range(len(vec)):
        norm = max(norm, abs(vec[i]))

    return norm

def line_rate_norm(matrix):
    norm = 0

    for i in range(len(matrix)):
        lineSum = 0

        for j in range(len(matrix[i])):
            lineSum += abs(matrix[i][j])

        norm = max(norm, lineSum)

    return norm

def iterations(A, b, e, n):
    A1 = np.zeros((n, n))
    b1 = np.ndarray(n)
    x = np.ndarray(n)

    for i in range(n):
        b1[i] = b[i] / A[i][i]
        for j in range(n):
            if (i != j):
                A1[i][j] -= A[i][j] / A[i][i]

    e1 = 1
    x = b1
    num = 0

    while e1 > e:
        x1 = np.ndarray(n)

        for i in range(n):
            x1[i] = b1[i] + np.dot(A1, x)[i]

        norm_A = line_rate_norm(A1)

        e1 = (norm_A / (1 - norm_A)) * chebyshev_norm(x1 - x)
        x = x1
        num += 1

    return (x, num)

def zeydel(A, b, e, n):
    A1 = np.zeros((n, n))
    b1 = np.ndarray(n)
    x0 = np.ndarray(n)
    x = np.ndarray(n)

    for i in range(n):
        b1[i] = b[i] / A[i][i]
        for j in range(n):
            if (i != j):
                A1[i][j] -= A[i][j] / A[i][i]

    num = 0

    while True:
        num += 1
        x0 = x.copy()

        for i in range(n):
            x[i] = b1[i] + np.dot(A1[i, :], x)

        if chebyshev_norm(x - x0) < e:
            break

    return (x, num)

def main():
    e = float(input())
    n = int(input())
    A = np.array([list(map(int, input().split())) for _ in range(n)])
    b = np.array(list(map(int, input().split())))

    (x1, num1) = iterations(A, b, e, n)
    (x2, num2) = zeydel(A, b, e, n)

    file_name = sys.argv[1]

    output_file_name = file_name.replace(".txt", "_answer.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(f"Solution by iteration method: {x1}\n")
        output_file.write(f"Number of iterations: {num1}\n")
        output_file.write(f"Seidel's solution: {x2}\n")
        output_file.write(f"Number of iterations: {num2}\n")

    print("Результаты записаны в файл:", output_file_name)

main()