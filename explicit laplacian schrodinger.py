import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# --- Parameters ---
N = 300                  # Number of grid points along theta
m = 10                    # Magnetic quantum number
k = 5000.0                # Potential strength V = 0.5 * k * theta^2
neigs = 4

# Setup grid avoiding exact 0 and pi to prevent division by zero (singularities)
theta = np.linspace(1e-4, np.pi - 1e-4, N)
dtheta = theta[1] - theta[0]

# --- 1. Construct 1D Finite Difference Operators ---
# Identity matrix
I = np.eye(N)

# First derivative matrix (Central Difference)
D1 = (np.diag(np.ones(N-1), 1) - np.diag(np.ones(N-1), -1)) / (2 * dtheta)
# Second derivative matrix (Central Difference)
D2 = (np.diag(np.ones(N-1), 1) - 2 * np.eye(N) + np.diag(np.ones(N-1), -1)) / (dtheta**2)

# Boundary Conditions: Dirichlet at the edges
D1[0, :] = D1[-1, :] = 0
D2[0, :] = D2[-1, :] = 0

# --- 2. Construct Scalar Laplacian Components ---
cot_theta = 1.0 / np.tan(theta)
sin2_theta = np.sin(theta)**2

# Term: cot(theta) * d/d_theta
D1_geometric = np.diag(cot_theta) @ D1

# Scalar Laplacian operator matrix (excluding the m^2/sin^2 part)
Laplacian_scalar = D2 + D1_geometric - np.diag(m**2 / sin2_theta)

# --- 3. Construct the Coupled Vector Laplacian Blocks ---
# Diagonal Blocks: Includes scalar Laplacian, geometric vector shift, and potential
V_diag = 0.5 * k * (theta**2)

H_theta_theta = -0.5 * (Laplacian_scalar - np.diag(1.0 / sin2_theta)) + np.diag(V_diag)
H_phi_phi     = -0.5 * (Laplacian_scalar - np.diag(1.0 / sin2_theta)) + np.diag(V_diag)

# Off-diagonal cross-coupling Blocks (Notice the '1j' for complex coupling)
H_theta_phi = -0.5 * np.diag(-2j * m / sin2_theta)
H_phi_theta = -0.5 * np.diag(2j * m / sin2_theta)

# Assemble Full Vector Hamiltonian Block Matrix
# H = [[H_theta_theta, H_theta_phi],
#      [H_phi_theta,   H_phi_phi  ]]
H = np.block([
    [H_theta_theta, H_theta_phi],
    [H_phi_theta,   H_phi_phi  ]
])

# --- 4. Solve the Eigensystem ---
print("Diagonalizing the finite-difference Vector Hamiltonian...")
eigenvalues, eigenvectors = la.eigh(H)

# --- 5. Extract and Plot Components |\psi_\theta| and |\psi_\phi| ---
print("\nLowest 3 Vector Energy Eigenvalues:")
for i in range(neigs):
    print(f"E_{i} = {eigenvalues[i]:.4f}")

# Plotting
fig, axes = plt.subplots(neigs, 1, figsize=(10, 12), sharex=True)

for state_idx in range(neigs):
    psi = eigenvectors[:, state_idx]
    
    # Extract theta and phi components back from the joint vector
    psi_theta = psi[:N]
    psi_phi = psi[N:]
    
    # Calculate magnitudes |\psi_\nu|
    mag_psi_theta = np.abs(psi_theta)
    mag_psi_phi = np.abs(psi_phi)
    
    # Normalize profiles for clean spatial visualization
    norm_factor = np.max(np.sqrt(mag_psi_theta**2 + mag_psi_phi**2))
    mag_psi_theta /= norm_factor
    mag_psi_phi /= norm_factor
    
    ax = axes[state_idx]
    ax.plot(np.degrees(theta), mag_psi_theta, label=r'$|\psi_\theta|(\theta)$', color='crimson', lw=2)
    ax.plot(np.degrees(theta), mag_psi_phi, label=r'$|\psi_\phi|(\theta)$', color='royalblue', lw=2, linestyle='--')
    
    # Overlay the potential well structure
    ax_twin = ax.twinx()
    ax_twin.plot(np.degrees(theta), V_diag, ':', color='gray', alpha=0.6, label='V')
    if state_idx == 0:
        ax_twin.set_ylabel("Potential V", color='gray')
    ax_twin.tick_params(axis='y', labelcolor='gray')
    
    ax.set_title(f"State {state_idx} (Energy E = {eigenvalues[state_idx]:.2f})")
    ax.set_ylabel("Wavefunction Magnitude")
    ax.grid(True, linestyle=':')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 90) # Focus near the harmonic well center

axes[-1].set_xlabel(r"Theta $\theta$ (degrees)")
plt.tight_layout()
plt.show()