import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
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
        # Начальное приближение для вектора
        q = np.random.rand(self.n)
        q /= norm(q)  # Нормируем вектор

        # Массивы для хранения коэффициентов
        alphas = []
        betas = []

        # Матрица T (Тридиагональная матрица)
        T = np.zeros((self.num_eigenvalues, self.num_eigenvalues))

        for i in range(self.num_eigenvalues):
            # Умножаем A на q
            v = self.A.dot(q)
            
            # Альфа-коэффициент
            alpha = np.dot(q, v)
            T[i, i] = alpha
            v -= alpha * q  # Отнимаем компоненту по q
            
            # Бета-коэффициент
            if i > 0:
                v -= betas[i-1] * q_prev

            beta = norm(v)

            if i < self.num_eigenvalues - 1:
                T[i, i+1] = beta
                T[i+1, i] = beta

            if beta < 1e-12:
                break  # Прерываем, если значение бета слишком мало (для разреженных матриц)

            # Нормируем новый вектор
            q_prev = q
            q = v / beta
            
            # Сохраняем коэффициенты
            alphas.append(alpha)
            betas.append(beta)

        # Решаем задачу о собственных значениях для матрицы T
        eigvals, eigvecs = np.linalg.eigh(T[:len(alphas), :len(alphas)])  # Только нужная часть матрицы T
        
        # Применяем собственные векторы для вычисления собственных векторов A
        for i in range(self.num_eigenvalues):
            v = np.zeros(self.n)

            for j in range(i + 1):
                v += eigvecs[j, i] * q

            self.eigenvectors.append(v)
        
        # Возвращаем собственные значения и векторы
        self.eigenvalues = eigvals
        
        return eigvals, self.eigenvectors

# Функция для использования метода Ланцоша с разреженной матрицей
def lanczos_method(A, num_eigenvalues):
    """
    Метод Ланцоша для нахождения собственных значений и векторов для симметричных разреженных матриц.

    A: csr_matrix - разреженная симметричная матрица
    num_eigenvalues: int - количество собственных значений для вычисления
    """
    lanczos = Lanczos(A, num_eigenvalues)
    eigenvalues, eigenvectors = lanczos.run()

    return eigenvalues, eigenvectors

# Функция для чтения матрицы из файла
def read_matrix_from_file(filename):
    """
    Чтение симметричной разреженной матрицы из текстового файла.
    
    Файл должен быть в формате:
    - Каждая строка представляет строку матрицы, элементы разделены пробелами.
    """
    with open(filename, 'r', encoding='utf-8') as f:  # Указание кодировки для правильного чтения
        lines = f.readlines()
    
    # Преобразуем строки в массивы чисел
    matrix_data = []
    for line in lines:
        matrix_data.append(list(map(float, line.split())))

    # Преобразуем список в numpy массив и затем в разреженную матрицу csr
    matrix = np.array(matrix_data)
    return csr_matrix(matrix)


# Функция для вычисления ошибок между точными и численно полученными значениями
def calculate_errors(true_vals, true_vecs, approx_vals, approx_vecs):
    """
    Вычисление ошибок для собственных значений и векторов.

    true_vals: собственные значения (точные)
    true_vecs: собственные векторы (точные)
    approx_vals: собственные значения (по методу Ланцоша)
    approx_vecs: собственные векторы (по методу Ланцоша)
    """
    eigval_errors = np.abs(true_vals - approx_vals)
    eigvec_errors = [norm(true_vec - approx_vec) for true_vec, approx_vec in zip(true_vecs.T, approx_vecs)]
    
    return eigval_errors, eigvec_errors

def main():
    # Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description="Решение задачи нахождения собственных значений и векторов для симметричных разреженных матриц методом Ланцоша.")
    parser.add_argument('filename', type=str, help="Имя файла с матрицей")
    parser.add_argument('-e', '--num_eigenvalues', type=int, default=None, help="Количество собственных значений для нахождения")
    parser.add_argument('-f', '--output_file', type=str, default=None, help="Файл для записи результата (необязательный)")
    args = parser.parse_args()

    # Чтение матрицы из файла
    A = read_matrix_from_file(args.filename)

    # Установка значений по умолчанию, если флаги не заданы
    num_eigenvalues = args.num_eigenvalues if args.num_eigenvalues else A.shape[0]

    # Ограничения на параметры
    if num_eigenvalues >= A.shape[0]:
        print(f"Ошибка: количество СЗ должно быть меньше размерности матрицы ({A.shape[0]})")
        
        return
    
    # Запуск метода Ланцоша
    lanczos_vals, lanczos_vecs = lanczos_method(A, num_eigenvalues)
    
    # Точные значения и векторы с использованием scipy
    true_vals, true_vecs = eigsh(A, k=num_eigenvalues, which='LM')
    
    # Вывод результатов
    output = []
    output.append("Собственные значения (метод Ланцоша):")
    output.append(str(lanczos_vals))
    for i, v in enumerate(lanczos_vecs):
        output.append(f"Собственный вектор {i+1} (метод Ланцоша): {v}")
    
    output.append("\nСобственные значения (точное вычисление):")
    output.append(str(true_vals))
    for i, v in enumerate(true_vecs.T):
        output.append(f"Собственный вектор {i+1} (точное вычисление): {v}")
    
    # Вычисление ошибок
    eigval_errors, eigvec_errors = calculate_errors(true_vals, true_vecs, lanczos_vals, lanczos_vecs)

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # График ошибок для собственных значений
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_eigenvalues + 1), eigval_errors, marker='o', linestyle='-', color='b')
    plt.xlabel("Индекс собственных значений")
    plt.ylabel("Ошибка")
    plt.title("Ошибка для собственных значений")

    # График ошибок для собственных векторов
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_eigenvalues + 1), eigvec_errors, marker='o', linestyle='-', color='r')
    plt.xlabel("Индекс собственных векторов")
    plt.ylabel("Ошибка")
    plt.title("Ошибка для собственных векторов")

    plt.tight_layout()
    plt.show()

    # Запись в файл, если задан ключ -f
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:  # Указание кодировки для записи
            f.write("\n".join(output))
    else:
        print("\n".join(output))


if __name__ == "__main__":
    main()
