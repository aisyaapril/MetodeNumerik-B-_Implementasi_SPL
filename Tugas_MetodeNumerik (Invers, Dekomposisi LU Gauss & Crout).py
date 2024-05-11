import numpy as py
import numpy as np

#Metode Matriks Balikan

print("Penyelesaian Menggunakan Metode Matriks Balikan")
print("the system of equations are:")
print('x - y + 2z = 5')
print('3x + z = 10')
print('x + 2z = 5')

A = py.array ([[1,-1,2],[3,0,1],[1,0,2]])
B=py.array([5,10,5])

C=py.linalg.solve(A,B)

print("The solution to the linear equation using matrix method is :")
print("[x,y]=",C)

#Metode Dekomposisi LU Gauss

print(" ")
print("Penyelesaian Menggunakan Metode Dekompoisi LU Gauss")
A = np.array([[2, -2, -2], 
              [0, -2, 2], 
              [-1, 5, 2]])
y = np.array([-4, -2, 6])

A_inv = np.linalg.inv(A)

x = np.dot(A_inv, y)
print(x)

from scipy.linalg import lu

P, L, U = lu(A)
print('P:\n', P)
print('L:\n', L)
print('U:\n', U)
print('LU:\n',np.dot(L, U))

#Metode Dekomposisi Crout

print(" ")
print("Penyelesaian Menggunakan Metode Dekomposisi Crout")
def crout_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for p in range(n):
        for j in range(p, n):
            sum = 0
            for k in range(p):
                sum += L[p, k] * U[k, j]
            U[p, j] = A[p, j] - sum

        q = p
        for i in range(q, n):
            if i == q:
                L[i, q] = 1
            else:
                sum = 0
                for k in range(q):
                    sum += L[i, k] * U[k, q]
                L[i, q] = (A[i, q] - sum) / U[q, q]

    return L, U

# Example usage
A = np.array([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]])
L, U = crout_decomposition(A)

print("Matrix L:")
print(L)
print("\nMatrix U:")
print(U)