import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# --- Parameters ---
N = 300                 # Total grid points
m = 3                   # Magnetic quantum number
k = 50.0                 # Free particle limit (no potential)
neigs=6

# Full grid for setting up dtheta
dtheta = np.pi / N
theta_full = np.linspace(dtheta/2, np.pi - dtheta/2, N)
# dtheta = theta_full[1] - theta_full[0]

# Solve ONLY on interior points to enforce Dirichlet boundary conditions perfectly
theta = theta_full[1:-1]
N_int = len(theta)

# --- 1. Construct Perfectly Symmetric Second Derivative Matrix ---
D2 = (np.diag(np.ones(N_int-1), 1) - 2 * np.eye(N_int) + np.diag(np.ones(N_int-1), -1)) / (dtheta**2)

# --- 2. Effective Geometric Potential (from chi-transformation) ---
sin2_theta = np.sin(theta)**2
V_diag = 0.5 * k * (theta**2)

# The transformation eliminates the 1st derivative and modifies the effective potential
diag_kinetic_eff = 0.5 * ((m**2 + 0.75) / sin2_theta - 0.25)

# --- 3. Assemble the Blocks ---
H_theta_theta = -0.5 * D2 + np.diag(diag_kinetic_eff + V_diag)
H_phi_phi     = -0.5 * D2 + np.diag(diag_kinetic_eff + V_diag)

# Cross-coupling terms (Now cleanly Hermitian)
H_theta_phi = np.diag(1j * m / sin2_theta)
H_phi_theta = np.diag(-1j * m / sin2_theta)

H = np.block([
    [H_theta_theta, H_theta_phi],
    [H_phi_theta,   H_phi_phi  ]
])

# --- 4. Solve ---
print("Diagonalizing perfectly symmetric Vector Hamiltonian...")
eigenvalues, eigenvectors = la.eigh(H)

# --- 5. Extract and Plot ---
fig, axes = plt.subplots(neigs, 1, figsize=(10, 12), sharex=True)

for state_idx in range(neigs):
    chi_vec = eigenvectors[:, state_idx]
    
    # Extract interior chi components
    chi_theta = chi_vec[:N_int]
    chi_phi = chi_vec[N_int:]
    
    # Transform BACK to physical wavefunctions: psi = chi / sqrt(sin(theta))
    psi_theta = chi_theta / np.sqrt(np.sin(theta))
    psi_phi = chi_phi / np.sqrt(np.sin(theta))
    
    # Get magnitudes
    mag_psi_theta = np.abs(psi_theta)
    mag_psi_phi = np.abs(psi_phi)
    
    # Normalize for plotting scale
    max_val = np.max(np.sqrt(mag_psi_theta**2 + mag_psi_phi**2))
    mag_psi_theta /= max_val
    mag_psi_phi /= max_val
    
    ax = axes[state_idx]
    ax.plot(np.degrees(theta), mag_psi_theta, label=r'$|\psi_\theta|(\theta)$', color='crimson', lw=2)
    ax.plot(np.degrees(theta), mag_psi_phi, label=r'$|\psi_\phi|(\theta)$', color='royalblue', lw=2, linestyle='--')

    ## Fitting first mode
    # ax.plot(np.degrees(theta), mag_psi_theta/theta, label=r'$|\psi_\theta|(\theta)$', color='crimson', lw=2)
    # ax.plot(np.degrees(theta), 4.35*(k/50)**0.25 * np.exp(-theta**2*3.45*(k/50)**0.5), label=r'$e^{-\theta^2}$', color='green', lw=2, linestyle=':')
    # ax.plot(np.degrees(theta), mag_psi_phi/theta, label=r'$|\psi_\phi|(\theta)$', color='royalblue', lw=2, linestyle='--')
    
    ax.set_title(f"Free Vector Field State {state_idx} (Energy E = {eigenvalues[state_idx]:.4f})")
    ax.set_ylabel("Wavefunction Magnitude")
    ax.grid(True, linestyle=':')
    ax.set_xlim(0, 180) # Look at the entire globe
    ax.legend(loc='upper right')

axes[-1].set_xlabel(r"Theta $\theta$ (degrees)")
plt.tight_layout()
plt.show()