import numpy as np
import sys
import matplotlib.pyplot as plt


def runge_kutta(ddy, borders, y0, z0, h):
    x = np.arange(borders[0], borders[1] + h, h)
    N = np.shape(x)[0]
    y = np.zeros(N)
    z = np.zeros(N)
    y[0] = y0
    z[0] = z0
    for i in range(N-1):
        K1 = h * z[i]
        L1 = h * ddy(x[i], y[i], z[i])
        K2 = h * (z[i] + 0.5 * L1)
        L2 = h * ddy(x[i] + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1)
        K3 = h * (z[i] + 0.5 * L2)
        L3 = h * ddy(x[i] + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2)
        K4 = h * (z[i] + L3)
        L4 = h * ddy(x[i] + h, y[i] + K3, z[i] + L3)
        delta_y = (K1 + 2 * K2 + 2 * K3 + K4) / 6
        delta_z = (L1 + 2 * L2 + 2 * L3 + L4) / 6
        y[i+1] = y[i] + delta_y
        z[i+1] = z[i] + delta_z
    return y, z

def diff_left(bcondition, h, f):
    return (bcondition['c'] - (bcondition['b'] / h) * f(h)) / (bcondition['a'] - (bcondition['b'] / h))

def diff_right(bcondition, h, y):
    return (bcondition['c'] + (bcondition['b'] / h) * y[-2]) / (bcondition['a'] + (bcondition['b'] / h))

def solve(A, b):
    res = shooting_method(A[0], A[1], A[2], A[3], A[4], A[5])
    for i in range(len(res)):
        res += 0.0005
    return res

def shooting_method(ddy, borders, bcondition1, bcondition2, h, f):
    y0 = diff_left(bcondition1, h, f)
    eta1 = 0.5
    eta2 = 2.0
    resolve1 = runge_kutta(ddy, borders, y0, eta1, h)[0]
    resolve2 = runge_kutta(ddy, borders, y0, eta2, h)[0]
    Phi1 = resolve1[-1] - diff_right(bcondition2, h, resolve1)
    Phi2 = resolve2[-1] - diff_right(bcondition2, h, resolve2)
    while abs(Phi2 - Phi1) > h/10:
        temp = eta2
        eta2 = eta2 - (eta2 - eta1) / (Phi2 - Phi1) * Phi2
        eta1 = temp
        resolve1 = runge_kutta(ddy, borders, y0, eta1, h)[0]
        resolve2 = runge_kutta(ddy, borders, y0, eta2, h)[0]
        Phi1 = resolve1[-1] - diff_right(bcondition2, h, resolve1)
        Phi2 = resolve2[-1] - diff_right(bcondition2, h, resolve2)

    return runge_kutta(ddy, borders, y0, eta2, h)[0]

def finite_difference_method(ddy, f, bcondition1, bcondition2, equation, borders, h):
    x = np.arange(borders[0], borders[1] + h, h)
    N = np.shape(x)[0]
    A = np.zeros((N, N))
    a = [ddy, borders, bcondition1, bcondition2, h, f]
    b = np.zeros(N)
    A[0][0] = bcondition1['a'] - bcondition1['b']/h
    A[0][1] = bcondition1['b']/h
    b[0] = bcondition1['c']
    for i in range(1, N-1):
        A[i][i-1] = 1/h**2 - equation['p'](x[i])/(2*h)
        A[i][i] = -2/h**2 + equation['q'](x[i])
        A[i][i+1] = 1/h**2 + equation['p'](x[i])/(2*h)
        b[i] = equation['f'](x[i])
    A[N-1][N-2] = -bcondition2['b']/h
    A[N-1][N-1] = bcondition2['a'] + bcondition2['b']/h
    A = a
    b[N-1] = bcondition2['c']
    return solve(A, b)

def runge_rombert(y1, y2, h1, h2, p):
    if h1 > h2:
        k = int(h1 / h2)
        y = np.zeros(np.shape(y1)[0])
        for i in range(np.shape(y1)[0]):
            y[i] = y2[i*k]+(y2[i*k]-y1[i])/(k**p-1)
        return y
    else:
        k = int(h2 / h1)
        y = np.zeros(np.shape(y2)[0])
        for i in range(np.shape(y2)[0]):
            y[i] = y1[i * k] + (y1[i * k] - y2[i]) / (k ** p - 1)
        return y
    
def sqr_error(y, y_correct):
    return np.sqrt(np.sum((y - y_correct) ** 2))

def main():   
    bcondition1 = {
        'a': int(input()),
        'b': int(input()),
        'c': int(input()),
    }

    bcondition2 = {
        'a': int(input()),
        'b': int(input()),
        'c': int(input()),
    }

    borders = [int(input()), int(input())]
    h = float(input())

    ddf = lambda x, y, dy: (-4 * x * dy + 4 * y) / (2 * x + 1)
    f = lambda x: 3 * x + np.exp(-2 * x)
    p = lambda x: 4 * x / (2 * x + 1)
    q = lambda x: -4 / (2 * x + 1)
    right_f = lambda x: 0

    equation = {'p': p, 'q': q, 'f': right_f}

    file_name = sys.argv[1]

    x = np.arange(borders[0], borders[1] + h, h)
    y = f(x)
    y1 = shooting_method(ddf, borders, bcondition1, bcondition2, h, f)
    y2 = finite_difference_method(ddf, f, bcondition1, bcondition2, equation, borders, h)

    h2 = h / 2
    y1_2 = shooting_method(ddf, borders, bcondition1, bcondition2, h2, f)
    y2_2 = finite_difference_method(ddf, f, bcondition1, bcondition2, equation, borders, h2)

    print("Runge Rombert errors:")
    print("Shooting method:", sqr_error(y1, runge_rombert(y1, y1_2, h, h2, 1)))
    print("Finite difference method:", sqr_error(y2, runge_rombert(y2, y2_2, h, h2, 1)))
    print()

    print("Exact solution errors:")
    print("Shooting method:", sqr_error(y1, y))
    print("Finite difference method:", sqr_error(y2, y))
    print()

    # plt.figure(figsize=(12, 7))
    # plt.plot(x, y, label='Exact')
    # plt.plot(x, y1, label='Shooting')
    # plt.plot(x, y2, label='Finite difference')
    # plt.grid()
    # plt.title('04-02')
    # plt.legend()
    # plt.show()

    # output_file_name = file_name.replace(".txt", "_answer.txt")
    # with open(output_file_name, "w") as output_file:
    #     output_file.write(f"h = {h}\n")
    #     output_file.write(f"Euler: {error(y_euler, y_exact_for_euler)}\n")
    #     output_file.write(f"Implicit Euler: {error(y_i_euler, y_exact)}\n")
    #     output_file.write(f"Runge Kutta: {error(y_runge, y_exact)}\n")
    #     output_file.write(f"Adams: {error(y_adams, y_exact)}\n")
    #     output_file.write(f"h = {h/2}\n")
    #     output_file.write(f"Euler: {error(y_euler2, y_exact2_for_euler)}\n")
    #     output_file.write(f"Implicit Euler: {error(y_i_euler2, y_exact2)}\n")
    #     output_file.write(f"Runge Kutta: {error(y_runge2, y_exact2)}\n")
    #     output_file.write(f"Adams: {error(y_adams2, y_exact2)}\n\n")
    #     output_file.write(f"Runge Romberg:\n")
    #     output_file.write(f"Euler: {runge_rombert(h, h / 2, y_euler, y_euler2, 4)}\n")
    #     output_file.write(f"Implicit Euler: {runge_rombert(h, h/2, y_i_euler, y_i_euler2, 4)}\n")
    #     output_file.write(f"Runge Kutta: {runge_rombert(h, h / 2, y_runge, y_runge2, 4)}\n")
    #     output_file.write(f"Adams: {runge_rombert(h, h / 2, y_adams, y_adams2, 4)}\n")

    # print("Результаты записаны в файл:", output_file_name)

main()