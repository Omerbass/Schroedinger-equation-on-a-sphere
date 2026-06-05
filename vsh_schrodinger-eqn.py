import numpy as np
import scipy.integrate as integrate
import scipy.linalg as la
import sympy as sp
import matplotlib.pyplot as plt

# --- 1. SymPy Setup for Vector Spherical Harmonics (VSH) ---
# We use SymPy to analytically get the theta/phi components of VSHs 
# to avoid hardcoding complex geometric derivative terms.
th, ph = sp.symbols('theta phi', real=True)

def get_vsh_components(l, m):
    """Returns analytical functions for the theta and phi components of VSH."""
    if l == 0:
        return lambda t: 0.0, lambda t: 0.0, lambda t: 0.0, lambda t: 0.0
        
    # Associated Legendre
    P = sp.assoc_legendre(l, m, sp.cos(th))
    # Scalar Y_lm (dropping constant phase for simplicity, normalized later)
    Y = P * sp.exp(sp.I * m * ph)
    
    # Type 1: Gradient-like VSH (Electric/Irrotational) -> Psi_lm
    # Psi_theta = dY/d_theta, Psi_phi = (1/sin_theta) * dY/d_phi
    psi_th_sym = sp.diff(Y, th)
    psi_ph_sym = (sp.I * m * Y) / sp.sin(th)
    
    # Type 2: Curl-like VSH (Magnetic/Solenoidal) -> Y_lm
    # Y_theta = -Psi_phi, Y_phi = Psi_theta
    y_th_sym = -psi_ph_sym
    y_ph_sym = psi_th_sym
    
    # Lambdify for fast numerical evaluation (setting phi=0 since potential is symmetric)
    # We evaluate at phi=0, keeping the m-dependence in the derivative integration.
    psi_th = sp.lambdify(th, psi_th_sym.subs(ph, 0), 'numpy')
    psi_ph = sp.lambdify(th, psi_ph_sym.subs(ph, 0), 'numpy')
    y_th = sp.lambdify(th, y_th_sym.subs(ph, 0), 'numpy')
    y_ph = sp.lambdify(th, y_ph_sym.subs(ph, 0), 'numpy')
    
    # Normalization constant for VSH
    norm = np.sqrt((2*l + 1) / (4 * np.pi * l * (l + 1)))
    
    return (lambda t: norm * np.real(psi_th(t)), lambda t: norm * np.imag(psi_ph(t)),
            lambda t: norm * np.imag(y_th(t)), lambda t: norm * np.real(y_ph(t)))

# --- 2. Numerical Integration & Matrix Building ---
L_MAX = 15   # Kept small here for speed; increase for better convergence
k = 20.0     # Harmonic potential strength
m_fixed = 1  # For vector fields, m=0 is often trivial/vanishing for certain components

# Build a list of basis elements: tuples of (Type, l) where Type 0 = Psi, Type 1 = Y
basis = []
for l in range(1, L_MAX + 1):
    if l >= abs(m_fixed):
        basis.append((0, l)) # Electric type
        basis.append((1, l)) # Magnetic type

N_basis = len(basis)
H = np.zeros((N_basis, N_basis))

def potential(theta):
    return 0.5 * k * (theta**2)

print(f"Building Vector Hamiltonian Matrix of size {N_basis}x{N_basis}...")

# Pre-compute component functions for efficiency
vsh_funcs = {}
for (v_type, l) in basis:
    vsh_funcs[(v_type, l)] = get_vsh_components(l, m_fixed)

for i, (type1, l1) in enumerate(basis):
    psi_th1, psi_ph1, y_th1, y_ph1 = vsh_funcs[(type1, l1)]
    f1_th = psi_th1 if type1 == 0 else y_th1
    f1_ph = psi_ph1 if type1 == 0 else y_ph1
    
    for j, (type2, l2) in enumerate(basis):
        if j < i: continue # Symmetric matrix
        
        # 1. Kinetic Energy (Diagonal in VSH basis)
        T_elem = 0.5 * l1 * (l1 + 1) if (l1 == l2 and type1 == type2) else 0.0
        
        # 2. Potential Energy matrix element: <V_i | V(theta) | V_j>
        psi_th2, psi_ph2, y_th2, y_ph2 = vsh_funcs[(type2, l2)]
        f2_th = psi_th2 if type2 == 0 else y_th2
        f2_ph = psi_ph2 if type2 == 0 else y_ph2
        
        # Integrand: (Component_Theta_1*Component_Theta_2 + Component_Phi_1*Component_Phi_2) * V(theta) * sin(theta)
        # 2*pi from phi integration
        def integrand(theta):
            dot_product = f1_th(theta) * f2_th(theta) + f1_ph(theta) * f2_ph(theta)
            return 2 * np.pi * dot_product * potential(theta) * np.sin(theta)
        
        V_elem, _ = integrate.quad(integrand, 0.001, np.pi - 0.001, limit=100)
        
        H[i, j] = T_elem + V_elem
        H[j, i] = H[i, j]

# --- 3. Diagonalization ---
print("Diagonalizing Vector Hamiltonian...")
eigenvalues, eigenvectors = la.eigh(H)

print("\nLowest 3 Vector Energy Eigenvalues:")
for i in range(3):
    print(f"E_{i} = {eigenvalues[i]:.4f}")

# --- 4. Plotting Magnitude Profiles ---
theta_grid = np.linspace(0.01, np.pi/2, 200)
plt.figure(figsize=(10, 6))

for state_idx in range(3):
    vec_density = np.zeros_like(theta_grid)
    
    # Reconstruct vector field magnitude profile |\psi|^2 = |\psi_theta|^2 + |\psi_phi|^2
    for i, (v_type, l) in enumerate(basis):
        psi_th, psi_ph, y_th, y_ph = vsh_funcs[(v_type, l)]
        f_th = psi_th if v_type == 0 else y_th
        f_ph = psi_ph if v_type == 0 else y_ph
        
        coeff = eigenvectors[i, state_idx]
        vec_density += coeff * (f_th(theta_grid)**2 + f_ph(theta_grid)**2)

    plt.plot(np.degrees(theta_grid), vec_density, label=f"Vector State {state_idx} (E={eigenvalues[state_idx]:.2f})")

plt.title(f"Vector Schrödinger Equation on a Sphere ($m={m_fixed}$, $k={k}$)")
plt.xlabel("Theta $\theta$ (degrees)")
plt.ylabel("Vector Probability Density $|\mathbf{\psi}(\theta)|^2$")
plt.grid(True, linestyle=':')
plt.legend()
plt.show()