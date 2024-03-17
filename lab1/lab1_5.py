import numpy as np
import sys


def householder_transformation(a):
    v = a.copy()
    v[0] = np.sign(a[0]) * np.linalg.norm(a) + a[0]
    v = v / np.linalg.norm(v)
    H = np.eye(len(a)) - 2 * np.outer(v, v)
    return H

def qr_decomposition(A, n):
    Q = np.eye(n)
    R = A.copy()

    for i in range(n - 1):
        H = np.eye(n)
        H[i:, i:] = householder_transformation(R[i:, i])
        R = np.dot(H, R)
        Q = np.dot(Q, H.T)

    return Q, R

def qr_eigenvalues(A, e, n):
    Ak = A.copy()

    iter = 0

    e_check = 10
    
    while e_check > e:
        iter += 1

        Q, R = qr_decomposition(Ak, n)
        A1 = np.dot(R, Q)
        
        for m in range(n - 1):
            e_check = np.sqrt(np.sum(A1[m+1:, m] ** 2))
        
        Ak = A1

    return np.diag(Ak), iter

def main():
    e = float(input())
    n = int(input())
    A = np.array([list(map(float, input().split())) for _ in range(n)])

    Q, R = qr_decomposition(A, n)
    h, iter = qr_eigenvalues(A, e, n)

    file_name = sys.argv[1]

    output_file_name = file_name.replace(".txt", "_answer.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(f"Eigenvalues: {h}\n")
        output_file.write(f"Q:\n {np.round(Q, 2)}\n")
        output_file.write(f"R:\n {np.round(R, 2)}\n")
        output_file.write(f"Q * R:\n {np.round(np.dot(Q, R), 2)}\n")
        output_file.write(f"Number of iterations: {iter}\n")

    print("Результаты записаны в файл:", output_file_name)

main()