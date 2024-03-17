import numpy as np
import sys


def solve(A, b, n):
    x = np.ndarray(n)
    v = np.ndarray(n)
    u = np.ndarray(n)

    v[0] = A[0][1] / (-A[0][0]) 
    u[0] = ( - b[0]) / (-A[0][0])

    for i in range(1, n - 1):
        v[i] = A[i][i + 1] / ( -A[i][i] - A[i][i - 1] * v[i - 1] )
        u[i] = (A[i][i - 1] * u[i - 1] - b[i]) / (-A[i][i] - A[i][i - 1] * v[i - 1] )

    v[n - 1] = 0
    u[n - 1] = (A[n - 1][n - 2] * u[n - 2] - b[n - 1]) / (-A[n - 1][n - 1] - A[n - 1][n - 2] * v[n - 2])

    x[n - 1] = u[n - 1]

    for i in range(n - 1, 0, -1):
        x[i - 1] = v[i - 1] * x[i] + u[i - 1]

    return x

def main():
    n = int(input())
    A = np.array([list(map(int, input().split())) for _ in range(n)])
    b = np.array(list(map(int, input().split())))

    file_name = sys.argv[1]

    x = solve(A, b, n)

    output_file_name = file_name.replace(".txt", "_answer.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(f"Solution:\n {x}\n")

    print("Результаты записаны в файл:", output_file_name)

main()