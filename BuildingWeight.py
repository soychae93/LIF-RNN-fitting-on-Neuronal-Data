# This code generates a recurrent neural network (RNN) weight matrix with 
# a biologically plausible excitatory/inhibitory (E/I) balance and ensures 
# the dynamical stability of the system, as proposed by Hennequin et al., Neuron (2014).
# It is a Python adaptation of the original MATLAB code written by Jake Stroud 
# (https://github.com/jakepstroud/gain_modulation_stability_optimised_circuits/blob/master/soc_function.m).
# The purpose of writing the code in python is to make the code compatible with the framework 
# used in Ta-Chu Kao et al., Neuron (2021).


import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import matplotlib.pyplot as plt

def initialnet(N, p, R, gamma):
    NN = int(np.round(p * N * (N - 1)))
    tmp = [0] * (N * (N - 1) - NN) + [1] * NN
    fill = np.array(tmp)[np.random.permutation(N * (N - 1))].reshape(N, N - 1)

    W1 = np.zeros((N, N))
    W1[0:N - 1, 1:] = fill[0:N - 1, :]

    W2 = np.zeros((N, N))
    W2[1:, 0:N - 1] = fill[1:, :]
    W = np.triu(W1, 1) + np.tril(W2, -1)

    w0 = np.sqrt(2) * R / (np.sqrt(p * (1 - p) * (1 + gamma ** 2)))
    W *= (w0 / np.sqrt(N))
    W[:, int(N / 2):] *= -gamma
    return W

def ssaCode(Wi, rate, gamma, N):
    end_exc = int(N / 2)
    eigval = np.linalg.eigvals(Wi)
    Emax = max(np.real(eigval))

    s = max(Emax * 1.5, Emax + 0.2)
    A = Wi - s * np.eye(N)
    X = -2 * np.eye(N)

    Q = solve_continuous_lyapunov(A.T, X)
    P = solve_continuous_lyapunov(A, X)

    grad = np.matmul(Q, P) / np.trace(np.matmul(Q, P))
    Wo = Wi.copy()
    Wo[:, end_exc:] -= rate * grad[:, end_exc:]

    # Remove positive inhibitory weights
    mask = Wo > 0
    mask[:, :end_exc] = False
    Wo[mask] = 0

    # Normalize inhibitory weights
    meanEE = np.mean(Wo[0:end_exc, 0:end_exc])
    meanEI = np.mean(Wo[end_exc:, 0:end_exc])
    meanIE = np.mean(Wo[0:end_exc, end_exc:])
    meanII = np.mean(Wo[end_exc:, end_exc:])

    Wo[0:end_exc, end_exc:] *= -gamma * (meanEE / meanIE)
    Wo[end_exc:, end_exc:] *= -gamma * (meanEI / meanII)

    # Sparsify inhibitory weights (keep top 40%)
    inh_weights = Wo[:, end_exc:].flatten()
    threshold = int(0.4 * len(inh_weights))
    sorted_indices = np.argsort(inh_weights, axis=None)
    inh_weights[sorted_indices[threshold:]] = 0
    Wo[:, end_exc:] = inh_weights.reshape(N, end_exc)

    # Remove self-connections
    np.fill_diagonal(Wo, 0)
    return Wo, Emax, eigval

def soc_function(W_initial, rate, desired_SA, gamma, N):
    Wsoc = W_initial.copy()
    e_values = []

    i = 0
    while True:
        Wsoc, emax, eigval = ssaCode(Wsoc, rate, gamma, N)
        e_values.append(emax)
        print(f"Iteration {i+1}: Spectral abscissa = {emax:.4f}")
        i += 1
        if emax <= desired_SA:
            break
    return Wsoc

# ======= Main =======
if __name__ == "__main__":
    N = 200               # Number of neurons
    p = 0.1               # Connection density
    R = 10                # Initial spectral abscissa
    gamma = 3             # Inhibition/excitation ratio

    W = initialnet(N, p, R, gamma)
    rate = 10
    desired_SA = 0.15

    Wsoc = soc_function(W, rate, desired_SA, gamma, N)

    # Optional: visualize eigenvalue spectrum
    eigvals = np.linalg.eigvals(Wsoc)
    plt.figure(figsize=(6, 4))
    plt.plot(np.real(eigvals), np.imag(eigvals), '.k')
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.title('Eigenvalue Spectrum of Optimized Weight Matrix')
    plt.grid(True)
    plt.show()
