import numpy as np
import sys
import matplotlib.pyplot as plt


def F(x, a):
    return np.array([
        x[0] ** 2 + x[1] ** 2 - a ** 2,
        x[0] - np.exp(x[1]) + a
    ])

def J(x):
    return np.array([
        [2 * x[0], 2 * x[1]],
        [1, -np.exp(x[1])]
    ])

def newton(x0, a, e):
    counter = 0

    for _ in range(1000):
        counter += 1

        delta = np.linalg.solve(J(x0), -F(x0, a))
        x0 += delta

        if np.linalg.norm(delta) < e:
            break

    return x0, counter

def iter(x0, a, epsilon):
    counter = 0
    x1, x2 = x0
    
    while True:
        counter += 1
        x1_new = np.sqrt(a ** 2 - x2 ** 2)
        x2_new = np.log(x1_new + a)

        q = 0.5

        if (q / (1 - q)) * np.sqrt((x1_new - x1) ** 2 + (x2_new - x2) ** 2) <= epsilon:
            return np.array([x1_new, x2_new]), counter
        
        x1, x2 = x1_new, x2_new

def graf(eq1, eq2, a, x2_range):
    theta = np.linspace(0, 2 * np.pi, 100)
    x1_circle = a * np.cos(theta)
    x2_circle = a * np.sin(theta)

    x2 = np.linspace(*x2_range, 400)
    x1_curve = eq2(x2, a)

    plt.figure(figsize=(8, 6))
    plt.plot(x1_circle, x2_circle, label='Circle: $x_1^2 + x_2^2 = a^2$')
    plt.plot(x1_curve, x2, label=eq1.__name__)

    plt.xticks(np.arange(-1, 2, 0.3))
    plt.yticks(np.arange(-2, 2, 0.3))

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.title('Graphs of the system of equations')
    plt.show()

def main():
    e = float(input())
    a = float(input())

    eq1 = lambda x2, a: np.exp(x2) - a
    eq2_name = 'x1 = e^(x2) - a'
    eq1.__name__ = eq2_name

    # graf(eq1, eq1, a=1, x2_range=(-2, 2))

    x0_1 = np.array([-10., 10.])

    root_newton, counter_newton = newton(x0_1, a, e)
    root_iter, counter_iter = iter(x0_1, a, e)

    file_name = sys.argv[1]

    output_file_name = file_name.replace(".txt", "_answer.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(f"x1 by Newton: {root_newton[0]}\n")
        output_file.write(f"x2 by Newton: {root_newton[1]}\n")
        output_file.write(f"Number of iterations: {counter_newton}\n")
        output_file.write(f"x1 by iter: {root_iter[0]}\n")
        output_file.write(f"x2 by iter: {root_iter[1]}\n")
        output_file.write(f"Number of iterations: {counter_iter}\n")

    print("Результаты записаны в файл:", output_file_name)

main()