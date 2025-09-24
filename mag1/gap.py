import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
plt.rcParams['font.family'] = 'serif'

def gap(num_instances):
    l = np.arange(2, 3, dtype=int)

    s = np.linspace(0, 1, 51)
    no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}

    # gaps, gaps_std = [], []

    for L in l:
        print(L, end=':  ')
        # basis_sym = spin_basis_1d(L, zblock=-1 if L % 2 == 1 else 1)
        L = int(L)
        basis_nosym = spin_basis_1d(L)
        gap = []
        for k in range(num_instances):
            print(k, end=', ')

            # # Random interactions on a full graph
            # J = np.random.rand(L, L) * 2 - 1
            # J = np.triu(J, 1)
            # J = J / np.max(np.abs(J))
            # d = []
            # for i in range(L):
            #     for j in range(i+1, L):
            #         d += [[J[i,j], i, j]]

            # Random interactions on a chain with PBC
            J = np.random.rand(L) * 2 - 1
            d = [[J[i], i, (i+1)%L] for i in range(L)]

            # eigvals_sym = np.zeros((2**(L-1), len(s)))
            # eigvals_nosym = np.zeros((2**(L), len(s)))
            eigvals = np.zeros((3, len(s)))
            for i in range(len(s)):
                Jzz = [[s[i] * d[j][0], d[j][1], d[j][2]] for j in range(len(d))]
                Jx = [[(1-s[i]), j] for j in range(L)]
                static = [['zz', Jzz], ['x', Jx]]
                dynamic = []
                # H_sym = hamiltonian(static, dynamic, basis=basis_sym, dtype=np.float64, **no_checks)
                H_nosym = hamiltonian(static, dynamic, basis=basis_nosym, dtype=np.float64, **no_checks)
                if L == 2:
                    E_nosym = H_nosym.eigvalsh()[:3]
                else:
                    E_nosym = H_nosym.eigsh(k=3, which='SA')[0] # Algebraically smallest two eigenvalues
                eigvals[:,i] = E_nosym

                # E_nosym = H_nosym.eigvalsh() # full spectrum
                # eigvals_nosym[:,i] = E_nosym
                # E_sym = H_sym.eigvalsh() # full spectrum
                # eigvals_sym[:,i] = E_sym

            gap += [np.min(eigvals[2] - eigvals[0])]
        
        print()
        fname = os.path.join('data', 'gaps', str(L)+'.txt')
        if os.path.exists(fname):
            g = np.loadtxt(fname)
            g = np.array(list(g) + gap)
        else:
            g = np.array(gap)
        np.savetxt(fname, g)
        # gaps += [np.average(gap)]
        # gaps_std += [np.std(gap)]


from multiprocessing import Process

def main():
    p = []
    for i in range(5):
        pi = Process(target=gap, args=([20]))
        pi.start()
        p += [pi]
    
    for i in range(5):
        p[i].join()


if __name__ == '__main__':
    main()