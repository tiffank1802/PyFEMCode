import numpy as np
import matplotlib.pyplot as plt

# Define problem parameters
L = 1.0  # Length of the bar
E = 200e9  # Young's Modulus
A = 1e-4  # Cross-sectional Area
num_elements = 4  # Number of elements
num_nodes = num_elements + 1 # Number of nodes

# Dynamic simulation parameters
rho = 7850  # Material density (kg/m^3)
total_time = 0.001  # Total simulation time (s)
c_wave = np.sqrt(E / rho)  # Wave speed (m/s)
L_e_avg = L / num_elements  # Average element length (m)
dt_critical = L_e_avg / c_wave  # Critical time step for stability
dt = dt_critical / 10  # Time step for integration (s), chosen as a fraction of critical dt

# Discretization
node_coords = np.linspace(0, L, num_nodes)

# Element stiffness matrix function
def element_stiffness(E, A, L_e):
    return (E * A / L_e) * np.array([[1, -1], [-1, 1]])

# Global stiffness matrix assembly
K_global = np.zeros((num_nodes, num_nodes))
F_global = np.zeros(num_nodes)

for i in range(num_elements):
    # Length of each element
    L_e = node_coords[i+1] - node_coords[i]
    k_e = element_stiffness(E, A, L_e)

    # Assemble into global matrix
    K_global[i:i+2, i:i+2] += k_e

# Apply Boundary Conditions and Loads
# Fix node 0 (left end)
fixed_dofs = [0]
penality=np.max(np.max(K_global))*1e14
# Apply a force at the end node (right end)
F_global[-1] = 1e6 # 1000 N tension

# Modify K_global and F_global for boundary conditions
K_mod = K_global.copy()
F_mod = F_global.copy()

for dof in fixed_dofs:
    K_mod[dof,dof]+=penality

# Solve for displacements
u = np.linalg.solve(K_mod, F_mod)

print("Nodal Displacements (u):")
print(u)
print(f"Material Density (rho): {rho} kg/m^3")
print(f"Total Simulation Time (total_time): {total_time} s")
print(f"Time Step (dt): {dt} s")

# Post-processing and Visualization
plt.figure(figsize=(10, 4))
plt.plot(node_coords, np.zeros_like(node_coords), 'bo-', label='Original Position')
plt.plot(node_coords + u, np.zeros_like(node_coords), 'ro--', label='Deformed Position')
plt.xlabel('Position (m)')
plt.ylabel('Displacement')
plt.title('1D Bar FEM Analysis')
plt.legend()
plt.grid(True)
plt.show()