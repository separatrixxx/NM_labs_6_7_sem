import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import norm
import argparse
import matplotlib.pyplot as plt


# Описание класса для метода Ланцоша
class Lanczos:
    def __init__(self, A, num_eigenvalues):
        """
        Инициализация метода Ланцоша.
        
        A: разреженная симметричная матрица (csr_matrix)
        num_eigenvalues: количество собственных значений для нахождения
        """
        self.A = A  # Матрица
        self.num_eigenvalues = num_eigenvalues  # Количество собственных значений
        self.n = A.shape[0]  # Размерность матрицы
        self.eigenvalues = []  # Список собственных значений
        self.eigenvectors = []  # Список собственных векторов

    def run(self):
        """
        Запуск метода Ланцоша для нахождения собственных значений и векторов.
        """
        q = np.random.rand(self.n)  # Начальное приближение для вектора
        q /= norm(q)  # Нормируем вектор

        # Массивы для хранения коэффициентов
        alphas = []
        betas = []
        Q = []  # Хранилище ортонормированных векторов
        Q.append(q)

        # Матрица T (Тридиагональная матрица)
        T = np.zeros((self.num_eigenvalues, self.num_eigenvalues))

        for j in range(self.num_eigenvalues):
            # Вычисляем z = A * q_j
            z = self.A.dot(Q[-1])
            
            # Альфа-коэффициент (диагональный элемент)
            alpha_j = Q[-1].dot(z)
            alphas.append(alpha_j)
            T[j, j] = alpha_j
            
            # Коррекция вектора z: ортогонализация на Q
            if j > 0:
                z -= betas[-1] * Q[-2]  # Убираем вклад предыдущего вектора
            z -= alpha_j * Q[-1]  # Убираем текущий вклад
            
            # Проверка на норму z
            beta_j = norm(z)

            if beta_j < 1e-9:
                print(f"Итерация {j}: малое значение beta = {beta_j}, метод завершён досрочно.")
                break
            
            # Сохраняем бета и добавляем новый вектор в Q
            betas.append(beta_j)
            Q.append(z / beta_j)
            
            # Обновляем тридиагональную матрицу T
            if j < self.num_eigenvalues - 1:
                T[j, j+1] = beta_j
                T[j+1, j] = beta_j

        # Решаем задачу для тридиагональной матрицы T
        eigvals, eigvecs = self.solve_tridiagonal(T, len(alphas))

        # Восстанавливаем собственные векторы для A
        self.eigenvectors = self.reconstruct_eigenvectors(Q, eigvecs)
        self.eigenvalues = eigvals

        return eigvals

    def solve_tridiagonal(self, T, size):
        """
        Решение задачи на собственные значения для тридиагональной матрицы с использованием QR-алгоритма.
        
        T: Тридиагональная матрица
        size: Размер используемой части T
        """
        A = T[:size, :size].copy()
        max_iter = 1000
        tol = 1e-10
        n = A.shape[0]
        for _ in range(max_iter):
            Q, R = np.linalg.qr(A)
            A = R @ Q
            off_diagonal = np.sum(np.abs(np.tril(A, -1)))
            if off_diagonal < tol:
                break
        eigvals = np.diag(A)
        eigvecs = np.eye(n)  # Собственные векторы пока оставляем как единичную матрицу (для упрощения)
        return eigvals, eigvecs

    def reconstruct_eigenvectors(self, Q, eigvecs):
        """
        Восстановление собственных векторов исходной матрицы A.
        
        Q: Ортонормированные векторы Ланцоша
        eigvecs: Собственные векторы тридиагональной матрицы T
        """
        m = len(eigvecs)
        n = len(Q[0])
        eigenvectors = np.zeros((m, n))

        for i in range(m):
            for j in range(m):
                eigenvectors[i] += eigvecs[j, i] * Q[j]

        return eigenvectors

# Функция для использовния метода Ланцоша с разреженной матрицей
def lanczos_method(A, num_eigenvalues):
    lanczos = Lanczos(A, num_eigenvalues)
    
    return lanczos.run()

# Собственная реализация QR-алгоритма для точного решения задачи собственных значений
def qr_eigenvalues(A, max_iter=1000, tol=1e-10):
    """
    Реализация QR-алгоритма для вычисления собственных значений симметричной матрицы.
    
    A: Матрица
    max_iter: Максимальное количество итераций
    tol: Допуск для остановки
    """
    A = A.copy()
    for _ in range(max_iter):
        Q, R = np.linalg.qr(A)
        A = R @ Q
        off_diagonal = np.sum(np.abs(np.tril(A, -1)))
        if off_diagonal < tol:
            break
    eigvals = np.diag(A)

    return eigvals

# Функция для чтения матрицы из файла
def read_matrix_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    matrix_data = [list(map(float, line.split())) for line in lines]

    return csr_matrix(np.array(matrix_data))

# Функция для вычисления ошибок
def calculate_errors(true_vals, approx_vals):
    """
    Вычисляет ошибки для собственных значений и векторов.
    """
    # Обрезка до минимального размера
    min_size = min(len(true_vals), len(approx_vals))
    true_vals = true_vals[:min_size]
    approx_vals = approx_vals[:min_size]
    
    eigval_errors = np.abs(true_vals - approx_vals)
    
    return eigval_errors

def plot_eigenvalues(true_vals, approx_vals, num_eigenvalues):
    """
    Строит график для сравнения собственных значений.
    
    true_vals: точные собственные значения (метод QR, все значения)
    approx_vals: собственные значения, найденные методом Ланцоша (только часть)
    num_eigenvalues: количество собственных значений, найденных Ланцошем
    """
    plt.figure(figsize=(8, 6))

    # Отображение значений, найденных методом Ланцоша
    plt.plot(range(1, num_eigenvalues + 1), approx_vals[:num_eigenvalues], 'go-', label='Метод Ланцоша')

    # Отображение всех значений, найденных методом QR
    plt.plot(range(1, len(true_vals) + 1), true_vals, 'ro-', label='Метод QR (все значения)')

    plt.title("Сравнение собственных значений")
    plt.xlabel("Индекс собственных значений")
    plt.ylabel("Собственные значения")
    plt.legend()
    plt.grid()

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Метод Ланцоша для симметричных разреженных матриц.")
    parser.add_argument('filename', type=str, help="Имя файла с матрицей")
    parser.add_argument('-e', '--num_eigenvalues', type=int, default=5, help="Количество собственных значений")
    parser.add_argument('-f', '--output_file', type=str, help="Файл для записи результатов")
    args = parser.parse_args()

    A = read_matrix_from_file(args.filename).toarray()  # Считываем матрицу и преобразуем в плотный формат
    num_eigenvalues = args.num_eigenvalues

    if num_eigenvalues >= A.shape[0]:
        print(f"Ошибка: количество СЗ должно быть меньше размерности матрицы ({A.shape[0]}).")

        return

    # Точное решение методом QR-алгоритма
    true_vals = qr_eigenvalues(A)

    # Решение методом Ланцоша
    lanczos_vals = lanczos_method(csr_matrix(A), num_eigenvalues)

    eigval_errors = calculate_errors(true_vals, lanczos_vals)

    plot_eigenvalues(true_vals, lanczos_vals, num_eigenvalues)

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"Собственные значения: {lanczos_vals}\n")
            f.write(f"Собственные значения (точные): {true_vals}\n")
            f.write(f"Ошибка для СЗ: {eigval_errors}\n")

if __name__ == "__main__":
    main()
