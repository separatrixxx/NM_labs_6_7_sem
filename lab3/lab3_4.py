import numpy as np
import sys
import numpy as np
import matplotlib.pyplot as plt


def df(x, y, x_):
    assert len(x) == len(y)

    for interval in range(len(x)):
        if x[interval] <= x_ < x[interval+1]:
            i = interval
            break

    a1 = (y[i+1] - y[i]) / (x[i+1] - x[i])
    a2 = ((y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - a1) / (x[i+2] - x[i]) * (2*x_ - x[i] - x[i+1])

    return a1 + a2

def d2f(x, y, x_):
    assert len(x) == len(y)

    for interval in range(len(x)):
        if x[interval] <= x_ < x[interval+1]:
            i = interval
            break

    num = (y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - (y[i+1] - y[i]) / (x[i+1] - x[i])

    return 2 * num / (x[i+2] - x[i])

def main():
    x_str = input().strip().split()
    y_str = input().strip().split()

    x = np.array([float(num) for num in x_str])
    y = np.array([float(num) for num in y_str])
    x_ = float(input())
    
    file_name = sys.argv[1]

    output_file_name = file_name.replace(".txt", "_answer.txt")
    with open(output_file_name, "w") as output_file:
        output_file.write(f"f'({x_}) = {df(x, y, x_):.4f}\n")
        output_file.write(f"f''({x_}) = {d2f(x, y, x_):.4f}\n")

    print("Результаты записаны в файл:", output_file_name)

main()