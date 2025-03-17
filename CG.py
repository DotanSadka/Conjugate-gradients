import numpy as np
import pickle
import os
import scipy.sparse as sp
import matplotlib.pyplot as plt

def load_data(file_dir):
    path = os.path.join(file_dir, 'CG_data.pkl')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    A, x_0, b = data['A'], data['x_0'], data['b']
    return A, x_0, b

file_dir = 'C:/Users/dotan/OneDrive/מסמכים/בר אילן/סמסטר ח/אלגברה לינארית נומרית/תכנות/2'  

A, x_0, b = load_data(file_dir)

# resonable tolerance and num of iteration to get an accurate solution without going to infinite loop
def CG(A, x_0, b, tol=1e-5, max_iter=1000):
    x = x_0
    r = b - A.dot(x)
    p = r
    rsold = np.dot(r.T, r)
    errors = []

    for i in range(max_iter):
        Ap = A.dot(p)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        errors.append(np.linalg.norm(A.dot(x) - b, 2)) #calculate the error

        if np.sqrt(rsnew) < tol:
            break #if the solution is accurate enough

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x, errors

# Ensure that A is in the sparse matrix format
A = sp.csr_matrix(A)

# Run the Conjugate Gradient function
x, errors = CG(A, x_0, b)

# Verify the shape of A, x_0, and b
print(f'Shape of A: {A.shape}')
print(f'Shape of x_0: {x_0.shape}')
print(f'Shape of b: {b.shape}')

# Plotting the error
plt.plot(np.log10(errors))
plt.xlabel('Iteration')
plt.ylabel('log10(Error)')
plt.title('Convergence of Conjugate Gradient Method')
plt.grid(True)
plt.show()
