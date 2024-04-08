import numpy as np
import sys
import re


def format_polynom(polynom_str):
    polynom_str = re.sub('--', '+', polynom_str)
    polynom_str = re.sub('\+-', '-', polynom_str)
    polynom_str = re.sub('-', ' - ', polynom_str)
    polynom_str = re.sub('\+', ' + ', polynom_str)
    polynom_str = re.sub('\* ', ' ', polynom_str)

    return polynom_str

def lagrange_with_check(f, x, test_point):
    res1, err1 = lagrange(f, x[1:], test_point)
    res2, err2 = lagrange(f, x[:-1], test_point)

    if err1 > err2:
        return res2, err2
    else:
        return res1, err1


def lagrange(f, x, test_point):
    y = [f(t) for t in x]
    assert len(x) == len(y)

    polynom_str = 'L(x) ='
    polynom_test_value = 0

    for i in range(len(x)):
        cur_enum_str = ''
        cur_enum_test = 1
        cur_denom = 1

        for j in range(len(x)):
            if i == j:
                continue

            cur_enum_str += f'(x-{x[j]:.2f})'
            cur_enum_test *= (test_point[0] - x[j])
            cur_denom *= (x[i] - x[j])

        polynom_str += f'+{(y[i] / cur_denom):.2f}' + cur_enum_str
        polynom_test_value += y[i] * cur_enum_test / cur_denom

    return format_polynom(polynom_str), abs(polynom_test_value - test_point[1])


def newton(f, x, test_point):
    y = [f(t) for t in x]
    assert len(x) == len(y)

    n = len(x)
    coefs = [y[i] for i in range(n)]

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            coefs[j] = float(coefs[j] - coefs[j - 1]) / float(x[j] - x[j - i])

    polynom_str = 'P(x) = '
    polynom_test_value = 0

    cur_multipliers_str = ''
    cur_multipliers = 1

    for i in range(n):
        polynom_test_value += cur_multipliers * coefs[i]
        
        if i == 0:
            polynom_str += f'{coefs[i]:.2f}'
        else:
            polynom_str += '+' + cur_multipliers_str + '*' + f'{coefs[i]:.2f}'

        cur_multipliers *= (test_point[0] - x[i])
        cur_multipliers_str += f'(x-{x[i]:.2f})'

    return format_polynom(polynom_str), abs(polynom_test_value - test_point[1])

def main():
    x_a_str = input().strip().split()
    x_b_str = input().strip().split()
    
    x_a = np.array([float(num) for num in x_a_str])
    x_b = np.array([float(num) for num in x_b_str])
    x_ = float(input())

    equation = lambda x: np.arcsin(x)


    polynomL1, errorL1 = lagrange_with_check(equation, x_a, (x_, equation(x_)))
    polynomL2, errorL2 = lagrange_with_check(equation, x_b, (x_, equation(x_)))
    polynomN1, errorN1 = newton(equation, x_a, (x_, equation(x_)))
    polynomN2, errorN2 = newton(equation, x_b, (x_, equation(x_)))

    file_name = sys.argv[1]

    output_file_name = file_name.replace(".txt", "_answer.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(f"Lagrange interpolation:\n")
        output_file.write(f"Points A polynom: {polynomL1}\n")
        output_file.write(f"Error in X*: {errorL1}\n")
        output_file.write(f"Points B polynom: {polynomL2}\n")
        output_file.write(f"Error in X*: {errorL2}\n\n")
        output_file.write(f"Newton interpolation:\n")
        output_file.write(f"Points A polynom: {polynomN1}\n")
        output_file.write(f"Error in X*: {errorN1}\n")
        output_file.write(f"Points B polynom: {polynomN2}\n")
        output_file.write(f"Error in X*: {errorN2}\n")

    print("Результаты записаны в файл:", output_file_name)

main()