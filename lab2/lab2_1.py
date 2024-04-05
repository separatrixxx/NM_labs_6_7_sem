import numpy as np
import sys
import matplotlib.pyplot as plt


def newton(x0, equation, equation_vp, e):
    max_iter = 1000

    counter = 0

    if (equation(x0) * equation_vp(x0) > 0):
        for _ in range(max_iter):
            counter += 1

            f_x = np.log(x0 + 1) - 2 * x0 ** 2 + 1
            df_x = 1 / (x0 + 1) - 4 * x0
            x1 = x0 - f_x / df_x

            if abs(x1 - x0) < e:
                return x1, counter

            x0 = x1

    return x0, counter

def iter(x0, e):
    max_iter = 1000

    counter = 0

    for _ in range(max_iter):
        counter += 1

        x1 = np.sqrt((np.log(x0 + 1) + 1) / 2)
        x1_pp = lambda x: np.sqrt(2) / (4 * np.sqrt(np.log(x + 1) + 1) * (x + 1))

        q = x1_pp(x0)

        if (q / (1 - q)) * abs(x1 - x0) <= e:
            return x1, counter

        x0 = x1

    return x0, counter

def graf(eq1, eq2):
    x = np.linspace(-0.6, 1, 100)
    plt.figure(figsize=(10, 6), dpi=80)

    plt.plot(x, eq1(x), label='f1(x) = ln(x + 1)')
    plt.plot(x, eq2(x), label='f2(x) = 2x^2 - 1')

    plt.xticks(np.arange(-0.6, 1.1, 0.1))
    plt.yticks(np.arange(-1, 1, 0.1))

    plt.legend()

    plt.title('Графики уравнений')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

def main():
    e = float(input())

    equation = lambda x: np.log(x + 1) - 2 * (x ** 2) + 1
    equation_vp = lambda x: -1 / ((x + 1) ** 2) - 4
    eq1 = lambda x: np.log(x + 1)
    eq2 = lambda x:  2 * (x ** 2) - 1

    # graf(eq1, eq2)
    
    root_newton, counter_newton = newton(0.95, equation, equation_vp, e)
    root_iter, counter_iter = iter(0.95, e)

    file_name = sys.argv[1]

    output_file_name = file_name.replace(".txt", "_answer.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(f"x by Newton: {root_newton}\n")
        output_file.write(f"Number of iterations: {counter_newton}\n")
        output_file.write(f"x by iter: {root_iter}\n")
        output_file.write(f"Number of iterations: {counter_iter}\n")

    print("Результаты записаны в файл:", output_file_name)

main()