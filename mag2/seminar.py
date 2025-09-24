# ======================================================================================
# %% Imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
from qutip import *
import scipy.constants as constants
import networkx as nx

plt.rcParams['figure.figsize'] = (4.3, 2.8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

np.set_printoptions(precision=3, suppress=True)

# =============================================================================
# %% Interpolating functions A(s) and B(s)

# def A(s):
#     return 1 - s

# def B(s):
#     return s

# s = np.linspace(0, 1, 100)
# a = A(s)
# b = B(s)

# fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.1))
# ax.plot(s, a, label=r'$A(s)$')
# ax.plot(s, b, label=r'$B(s)$')

# ax.set_xlabel(r'$s$', fontsize=14)
# ax.legend(fontsize=11)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_xticks([0, 1])
# ax.set_yticks([0, 1])

# fig.tight_layout()
# plt.savefig('plots/interpolating_functions.pdf')
# plt.show()
# plt.close()

# =============================================================================
# %% Lattice example

# Lx, Ly = 4, 3
# x, y = np.meshgrid(range(Lx), range(Ly))

# # dict of random couplings
# J = {}
# for i in range(Lx):
#     for j in range(Ly):
#         if i < Lx - 1:
#             J[(i, j), ((i+1)%Lx, j)] = np.random.rand()
#         if j < Ly - 1:
#             J[(i, j), (i, (j+1)%Ly)] = np.random.rand()

# # Generate random directions for the spins
# angles = np.random.rand(Lx, Ly) * 2 * np.pi

# # Plot the lattice with arrows
# plt.figure(figsize=(Lx, Ly))
# plt.quiver(x, y, np.cos(angles), np.sin(angles), scale=16, color='k', pivot='middle')

# # Plot the couplings
# cmap = plt.get_cmap('coolwarm')
# for edge, coupling in J.items():
#     (i, j), (k, l) = edge
#     col = cmap(coupling)
#     if i == k:
#         line = [(i, k), (j+0.2, l-0.2)]
#     else:
#         line = [(i+0.2, k-0.2), (j, l)]
#     plt.plot(*line, c=col, lw=1.2)

# # Add labels
# plt.text(1.25, 1.1, r'$J_{ij}(t)$', fontsize=20)
# plt.text(0.7, 0.7, r'$q_{i}$', fontsize=20)
# plt.text(2.1, 0.7, r'$q_{j}$', fontsize=20)

# plt.gca().set_axis_off()
# plt.gca().set_aspect('equal', adjustable='box')

# plt.tight_layout()
# plt.savefig('plots/lattice.pdf')
# plt.show()

# =============================================================================
# %% Quantum circuit example

# import qiskit as qk

# qc = qk.QuantumCircuit(3, 3)

# qc.h(0)
# qc.cx(0, 1)
# qc.cx(0, 2)
# qc.z(1)
# qc.cx(1, 2)

# qc.draw(output='mpl',
#         style='bw',
#         idle_wires=False,
#         initial_state=True,
#         filename='plots/quantum_circuit.pdf')

# =============================================================================
# %% Graph instance

# # Define a graph with nodes and edges
# G = nx.Graph()
# G.add_nodes_from([0, 1, 2, 3, 4])
# G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (3, 4), (0, 4)])

# node_labels = {v: f'$s_{v}$' for v in G.nodes()}
# edge_labels = {(u, v): f'$J_{{{str(u) + str(v)}}}$' for u, v in G.edges}

# # Plot the graph with edge weights represented by colors
# pos = nx.bfs_layout(G, 1)
# nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=500, node_color='white', edgecolors='black', font_weight='bold', font_color='black', font_size=16)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=16, label_pos=0.35)
# plt.tight_layout()
# plt.savefig('plots/graph_instance.pdf')
# plt.show()

# =============================================================================
# %% Max-cut instance

# Define a graph with nodes and edges
G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3, 4])
G.add_edges_from([(0, 1), (1, 2), (3, 0), (2, 4), (3, 4), (0, 2)])

node_labels = {v: f'$s_{v}$' for v in G.nodes()}
edge_weights = {edge: 1 for edge in G.edges}
edge_labels = {(u, v): f'$J_{{{str(u) + str(v)}}}$' for u, v in G.edges}

pos = {1: (-1, 0), 2: (0, 1), 0: (0, -1), 3: (1, -1), 4: (1, 1)}
x_cut = 0.5

fig, ax = plt.subplots(1, 1, figsize=(9, 3))
nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=800, node_color='white', edgecolors='black', font_weight='bold', font_color='black', font_size=20, ax=ax)
ax.axvline(x_cut, c='r')

G.remove_edge(2, 4)
G.remove_edge(0, 3)
G1 = G.subgraph(nx.node_connected_component(G, 1))
G2 = G.subgraph(nx.node_connected_component(G, 4))
G1_labels = {v: f'$s_{v}$' for v in G1.nodes()}
G2_labels = {v: f'$s_{v}$' for v in G2.nodes()}

x_offset = 3
pos1 = {node: (pos[node][0] + x_offset, pos[node][1]) for node in G1.nodes()}
pos2 = {node: (pos[node][0] + x_offset, pos[node][1]) for node in G2.nodes()}

nx.draw(G1, pos1, with_labels=True, labels=G1_labels, node_size=800, node_color='skyblue', edgecolors='black', font_weight='bold', font_color='black', font_size=20, ax=ax)
nx.draw(G2, pos2, with_labels=True, labels=G2_labels, node_size=800, node_color='orange', edgecolors='black', font_weight='bold', font_color='black', font_size=20, ax=ax)

ax.text(0.5*x_offset, 0, r'$\to$', fontsize=28, ha='center', va='center')

ax = plt.gca()
plt.savefig('plots/max_cut_instance.pdf')
plt.show()
plt.close()

# ======================================================================================
# %%

def number_of_crossings(eigenbasis):
    """
    Calculates the number of spectrum crossings of a time-dependent (annealing) Hamiltonian
    
    Parameters
    ----------
    eigenbasis : np.ndarray
        The eigenbasis of the Hamiltonian at each point in the annealing schedule.
        Three indices (t, i, j) represent the time and the i-th component of the j-th eigenvector
        
    Returns
    -------
    crossings : int
        The number of crossings in the spectrum
    """
    diff = np.diff(eigenbasis, axis=0)
    crossings = np.any(np.linalg.norm(diff, axis=1), axis=1)
    return np.sum(crossings)

# =============================================================================
# %% Generate the 1D transverse Ising model Hamiltonian with open boundary conditions
# and random z-z interactions

# N = 3
# gamma = 1

# J = 0
# sigma_J = 1
# h = 0
# sigma_h = 1

# sx = sigmax()
# sz = sigmaz()
# id = qeye(2)

# rng = np.random.default_rng(np.random.randint(0, 100000))
# ops = []

# # Add the transverse field term to the list of operators
# Hx_ops = []
# for n in range(N):
#     op_list = [id] * N
#     op_list[n] = sx
#     Hx_ops.append(gamma * tensor(op_list))
# Hx = sum(Hx_ops)

# # Add zz interaction terms to the list of operators
# Hzz_ops = []
# for n in range(N):
#     sz_list = [id] * N
#     sz_list[n] = sz
#     sz_list[(n+1)%N] = sz
#     Hzz_ops.append(rng.normal(J, sigma_J) * tensor(sz_list))
# Hzz = sum(Hzz_ops)

# # Add random z field terms to the list of operators
# Hz_list = []
# for n in range(N):
#     op_list = [id] * N
#     op_list[n] = sz
#     Hz_list.append(rng.normal(h, sigma_h) * tensor(op_list))
# Hz = sum(Hz_list)

# -----------------------------------------------------------------------------
# %% Calculate the spectrum of the Hamiltonian for different values of s

# s = np.linspace(0, 1, 51)
# spectrum = np.zeros((len(s), 2**N))
# eigenbasis = np.zeros((len(s), 2**N, 2**N))

# for i, alpha in enumerate(s):
#     H = (1 - alpha) * Hx + 0.7*alpha * (Hzz + Hz)
#     H = H.full().real
#     e, v = sla.eigh(H)
#     spectrum[i] = e
#     eigenbasis[i] = v.T

# -----------------------------------------------------------------------------
# %% Plot the flow of the spectrum
    
# gap = spectrum[:, 1] - spectrum[:, 0]
# min_gap = np.min(gap)
# s_mingap = s[np.argmin(gap)]
# E0_mingap = spectrum[np.argmin(gap), 0]

# fig, ax = plt.subplots(1, 1, figsize=(4.1, 2.6))
# for i in range(2**N):
#     ax.plot(s, spectrum[:, i], 'k', lw=1)
# ax.set_xlabel(r'$s$', fontsize=13)
# ax.set_ylabel(r'Energy')

# # draw a 2-headed arrow between the first and second lines at s=s_mingap
# ax.annotate('', xy=(s_mingap, E0_mingap-0.16), xytext=(s_mingap, E0_mingap+1.15*min_gap),
#             arrowprops=dict(arrowstyle='<->', lw=1))
# ax.text(s_mingap-0.04, E0_mingap-0.8*min_gap, r'$\Delta_{min}$', fontsize=12)

# # Denote the lowest two energy levels at s=1 with text E_0 and E_1
# ax.text(1, spectrum[-1, 0]-0.25, r'$E_0$', fontsize=12)
# ax.text(1, spectrum[-1, 1]-0.25, r'$E_1$', fontsize=12)
# ax.text(1.01, spectrum[-1, 2]+0.35, r'$\vdots$', fontsize=12)

# ax.set_xticks([0, 1])
# ax.set_yticks([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# fig.tight_layout()
# # plt.savefig('plots/ising_gap.pdf')
# plt.show()
# plt.close()


# =============================================================================
# %% Plot the benefit of custom annealing schedules

# gap = spectrum[:, 1] - spectrum[:, 0]
# min_gap = np.min(gap)
# s_mingap = s[np.argmin(gap)]
# E0_mingap = spectrum[np.argmin(gap), 0]
# T_default = 1 / min_gap * len(s)

# t = np.zeros_like(s)
# for i in range(1, len(s)):
#     t[i] = t[i-1] + 1 / (gap[i-1]**2)
    
# fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.6))
# for i in range(2**N):
#     ax[0].plot(s, spectrum[:, i], 'k', lw=1)
# ax[0].fill_between(s, spectrum[:, 0], spectrum[:, 1], color='C0', alpha=0.3)
# ax[0].text(0.8, 0.58*(spectrum[-10, 0]+spectrum[-10, 1]), r'$\Delta \, (s)$', fontsize=12, color='C0')
    
# ax[0].set_xlabel(r'$s$', fontsize=13)
# ax[0].set_ylabel(r'Energy')

# # draw a 2-headed arrow between the first and second lines at s=s_mingap
# ax[0].annotate('', xy=(s_mingap, E0_mingap-0.16), xytext=(s_mingap, E0_mingap+1.15*min_gap),
#             arrowprops=dict(arrowstyle='<->', lw=1))
# ax[0].text(s_mingap-0.04, E0_mingap-0.8*min_gap, r'$\Delta_{min}$', fontsize=12)

# # Denote the lowest two energy levels at s=1 with text E_0 and E_1
# ax[0].text(1.01, spectrum[-1, 0]-0.25, r'$E_0$', fontsize=12)
# ax[0].text(1.01, spectrum[-1, 1]-0.25, r'$E_1$', fontsize=12)
# ax[0].text(1.02, spectrum[-1, 2]+0.35, r'$\vdots$', fontsize=12)

# ax[0].set_xticks([0, 1])
# ax[0].set_yticks([])
# ax[0].spines['top'].set_visible(False)
# ax[0].spines['right'].set_visible(False)

# ax[1].plot(t, s, 'C0', label=r'$\dot{s} \, \propto \Delta^2(s)$')
# ax[1].plot([0, T_default], [0, 1], 'k', lw=0.8, label=r'$\dot{s} \, \propto \Delta_{min}^2$')
# ax[1].set_xlabel(r'$t$', fontsize=13)
# ax[1].set_ylabel(r'$s$', fontsize=13)
# ax[1].set_xticks([0, t[-1], T_default])
# ax[1].set_yticks([0, 1])
# ax[1].set_xticklabels([r'$0$', r'$T_\mathrm{non-linear}$', r'$T_\mathrm{linear}$'])
# ax[1].set_yticklabels([r'$0$', r'$1$'])
# ax[1].legend()

# fig.tight_layout()
# plt.savefig('plots/ising_gap_with_annealing_schedule.pdf')
# plt.show()
# plt.close()

# ##############################################################################
# %% QA vs SA

# rng = np.random.default_rng(np.random.randint(0, 100000))

# N_cos = 7
# k = 20 * np.pi * rng.random(N_cos)
# amp = rng.random(N_cos)

# Nx = 500
# x = np.linspace(0, 1, Nx)
# V = np.zeros((Nx, N_cos))
# for i in range(N_cos):
#     V[:, i] = 1 * (np.cos(k[i] * x) + np.sin(k[i] * x))
#     # V[:, i] = amp[i] * np.cos(k[i] * x)
# V = np.sum(V, axis=1)
# V -= np.min(V)
# V /= np.max(V)

# Nt = 100
# eps = 1e-8
# V0 = np.linspace(10, 1, Nt)

# p = np.zeros((Nt, Nx))
# p_init = 1/2 * np.ones(Nx)
# p_middle = (V - V0[Nt//2])**20
# p_middle -= np.min(p_middle)
# p_middle /= np.max(p_middle)
# p_final = (V - V0[-1])**20
# p_final -= np.min(p_final)
# p_final /= np.max(p_final)

# for i in range(Nt):
#     s = ((i / (Nt-1)) - 1)**3 + 1
#     P = (1-s) * p_init + (1-(2*s-1)**2) * p_middle + s * p_final
#     P /= np.sum(P)
#     p[i, :] = P

# p_max = np.max(p)
# for i in range(Nt):
#     p[i, :] += 0.15 * (p_max - np.max(p[i, :]))

# # =============================================================================
# # %%  
    
# fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.5))

# cmap = plt.get_cmap('coolwarm')
# ax.plot(x, V / np.max(V) * np.max(p), 'k-', lw=1.2, label='Energy')
# q = 15
# for t in [i*Nt//q for i in range(q)] + [Nt-1]:
#     col = cmap(t / (Nt-1))
#     ax.plot(x, p[t, :], c=col, lw=0.8)

# # add the colorbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=Nt-1))
# sm._A = []
# cbar = plt.colorbar(sm, ax=ax, ticks=[0, Nt-1])
# cbar.set_label(r'time', fontsize=11)
# cbar.ax.set_yticklabels([r'$0$', r'$T$'])

# ax.legend(fontsize=9.5)
# ax.set_xlabel(r'Configuration', fontsize=10.5)
# ax.set_ylabel(r'Probability density', fontsize=10.5)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# fig.tight_layout()
# plt.savefig('plots/QA.pdf')
# plt.show()
# plt.close()

# # %%

# x = np.linspace(0, 1, 100)
# y = 1-(2*x-1)**2

# fig, ax = plt.subplots(1, 1, figsize=(4.1, 2.6))
# ax.plot(x, y, 'k')

# fig.tight_layout()
# plt.show()
# plt.close()

# ##############################################################################
# %% Pause anneal schedule

# s_pause = 0.4

# t_init_quench = 5
# t_final_quench = 2.5*(1-s_pause)
# t_anneal = 30
# t_pause = t_anneal - t_init_quench - t_final_quench

# t = [0, t_init_quench, t_init_quench+t_pause, t_init_quench+t_pause+t_final_quench]
# s = [0, s_pause, s_pause, 1]


# fig, ax = plt.subplots(1, 1, figsize=(4.1, 2.4))

# ax.plot(t, s, 'k')

# ax.set_xlabel(r'$\mathrm{Time} \, [ \mu s ]$', fontsize=11)
# ax.set_ylabel(r'$s$', fontsize=13)
# ax.set_yticks([0, s_pause, 1])
# ax.set_yticklabels([r'$0$', r'$s_p$', r'$1$'])
# ax.set_xlim(0, None)
# # ax.set_xlim(0, t[-1])
# ax.set_ylim(0, 1)

# ax.axvline(t_anneal - 0.03, c='k', ls='-', lw=0.8)
# ax.axvline(t_anneal - t_final_quench, c='k', ls='-', lw=0.8)
# ax.axhline(1, c='k', ls='-', lw=0.8)

# ax.annotate(r'$\tau$', xy=(t_anneal+0.3, 0.2), xytext=(t_anneal - 2.77, 0.175),
#             arrowprops=dict(arrowstyle='<->, head_width=0.2, head_length=0.2', lw=1))

# ax.spines['top'].set_visible(False)

# fig.tight_layout()
# plt.savefig('plots/pause_schedule.pdf')
# plt.show()
# plt.close()

# =============================================================================
# %%
