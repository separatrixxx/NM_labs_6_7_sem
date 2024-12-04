import numpy as np
from scipy.sparse import coo_matrix
import sys


def generate_symmetric_sparse_matrix(size, num_elements=50):
    """
    Генерирует разреженную симметричную матрицу заданного размера.
    :param size: Размер матрицы (число строк/столбцов).
    :param num_elements: Количество ненулевых элементов.
    :return: Симметричная разреженная матрица в формате COO.
    """
    # Генерация случайных ненулевых элементов
    row = np.random.randint(0, size, size=num_elements)
    col = np.random.randint(0, size, size=num_elements)
    data = np.random.rand(num_elements)

    # Создаем разреженную матрицу в формате COO
    sparse_matrix = coo_matrix((data, (row, col)), shape=(size, size))

    # Симметризация матрицы
    symmetric_matrix = sparse_matrix + sparse_matrix.T

    # Преобразуем обратно в формат COO для разреженности
    return coo_matrix(symmetric_matrix)

def save_matrix_to_file(matrix, filename):
    """
    Сохраняет матрицу в файл.
    :param matrix: Разреженная матрица (формат COO).
    :param filename: Имя файла для сохранения.
    """
    dense_matrix = matrix.toarray()
    with open(filename, 'w') as f:
        for row in dense_matrix:
            f.write(" ".join(map(str, row)) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python gen.py <размер_матрицы>")
        sys.exit(1)

    try:
        size = int(sys.argv[1])
    except ValueError:
        print("Ошибка: размер матрицы должен быть целым числом.")
        sys.exit(1)

    # Генерация матрицы
    symmetric_sparse_matrix = generate_symmetric_sparse_matrix(size)

    # Сохранение матрицы в файл
    save_matrix_to_file(symmetric_sparse_matrix, "test.txt")
    print(f"Матрица размером {size}x{size} успешно сохранена в файл test.txt.")
