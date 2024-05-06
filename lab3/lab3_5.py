import numpy as np
import sys


def rectangle_trapeze(f, l, r, h, is_rectangle=True):
    if l > r:
        return None
    
    result = 0
    cur_x = l

    while cur_x < r:
        if is_rectangle:
            result += f((cur_x + cur_x + h) * 0.5)
        else:
            result += 0.5*(f(cur_x + h) + f(cur_x))
        cur_x += h

    return h*result

def simpson(f, l, r, h):
    if l > r:
        return None
    
    while ((l - r) // h) % 2 != 0:
        h *= 0.9

    result = 0
    cur_x = l + h

    while cur_x < r:
        result += f(cur_x - h) + 4 * f(cur_x) + f(cur_x + h)
        cur_x += 2 * h

    return result * h / 3

def runge_romberg(Fh, Fkh, k, p):
    return (Fh - Fkh) / (k**p - 1)

def main():
    x0 = float(input())
    xk = float(input())
    h1 = float(input())
    h2 = float(input())

    equation = lambda x: 1 / (x ** 2 + 4)

    rectangle_h1 = rectangle_trapeze(equation, x0, xk, h1)
    rectangle_h2 = rectangle_trapeze(equation, x0, xk, h2)
    trapeze_h1 = rectangle_trapeze(equation, x0, xk, h1, False)
    trapeze_h2 = rectangle_trapeze(equation, x0, xk, h2, False)
    simpson_h1 = simpson(equation, x0, xk, h1)
    simpson_h2 = simpson(equation, x0, xk, h2)

    rectangle_runge_rombert = runge_romberg(rectangle_h1, rectangle_h2, h2 / h1, 2)
    trapeze_runge_rombert = runge_romberg(trapeze_h1, trapeze_h2, h2 / h1, 2)
    simpson_runge_rombert = runge_romberg(simpson_h1, simpson_h2, h2 / h1, 2)

    file_name = sys.argv[1]

    output_file_name = file_name.replace(".txt", "_answer.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(f"Rectangle method:\n")
        output_file.write(f"Step {h1}: {rectangle_h1}\n")
        output_file.write(f"Step {h2}: {rectangle_h2}\n")
        output_file.write(f"Trapeze method:\n")
        output_file.write(f"Step {h1}: {trapeze_h1}\n")
        output_file.write(f"Step {h2}: {trapeze_h2}\n")
        output_file.write(f"Simpson method:\n")
        output_file.write(f"Step {h1}: {simpson_h1}\n")
        output_file.write(f"Step {h2}: {simpson_h2}\n")
        output_file.write(f"Runge Roberg method:\n")
        output_file.write(f"Rectangle: {rectangle_runge_rombert}\n")
        output_file.write(f"Trapeze: {trapeze_runge_rombert}\n")
        output_file.write(f"Simpson: {simpson_runge_rombert}\n")

    print("Результаты записаны в файл:", output_file_name)

main()