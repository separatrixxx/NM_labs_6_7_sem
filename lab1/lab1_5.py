import numpy as np
import sys


def householder_transformation(a):
    v = a.copy()
    
    v[0] = np.sign(a[0].real) * np.linalg.norm(a) + a[0]
    v = v / np.linalg.norm(v)
    H = np.eye(len(a), dtype=complex) - 2 * np.outer(v, v)
    
    return H

def qr_decomposition(A, n):
    Q = np.eye(n, dtype=complex)
    R = A.astype(complex)

    for i in range(n - 1):
        H = np.eye(n, dtype=complex)
        H[i:, i:] = householder_transformation(R[i:, i])
        R = np.dot(H, R)
        Q = np.dot(Q, H.T)

    return Q, R


def complex_solve(a11, a12, a21, a22, eps):
    a = 1
    b = -a11 - a22
    c = a11 * a22 - a12 * a21
    d = b ** 2 - 4 * a * c

    if d > eps:
        return None, None
    
    d_c = complex(0, np.sqrt(-d))
    x1 = (-b + d_c) / (2 * a)
    x2 = (-b - d_c) / (2 * a)

    return x1, x2

def qr_eigenvalues(A, e, n):
    Ak = np.array(A, dtype=complex)
    eig_vals = np.zeros(n, dtype=complex)
    iter = 0

    while True:
        iter += 1
        Q, R = qr_decomposition(Ak, n)
        Ak = np.dot(R, Q)

        conv = True
        i = 0

        while i < n:
            if i < n - 1 and abs(Ak[i + 1, i]) > e:
                eig1, eig2 = complex_solve(Ak[i, i], Ak[i, i +  1], Ak[i + 1, i], Ak[i + 1, i + 1], e)

                if eig1 is not None and eig2 is not None:
                    eig_vals[i], eig_vals[i+1] = eig1, eig2
                    i += 1
                else:
                    conv = False
            else:
                eig_vals[i] = Ak[i, i]

            i += 1

        if conv:
            break

    return eig_vals, iter

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
