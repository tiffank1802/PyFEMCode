import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


noeuds=np.array([[0,0],[1,0],[1,1],[0,1],[.5,.5],[0,.5],[.5,0],[.5,1],[1,.5]])
noeuds=2*noeuds
elements=np.array([
    [0,1],[1,2],[2,3],[3,0],  # Outer square
    [0,2],                    # Diagonal to stabilize main square
    [0,4],[2,4],[3,4],[1,4],  # Corners to center node 4
    [6,4],[7,4],[8,4],[5,4],  # Mid-points to center node 4
    [5,0], [5,3],             # Stabilize node 5 (left mid) vertically by connecting to fixed nodes 0 and 3
    [6,0], [6,1],             # Stabilize node 6 (bottom mid) horizontally by connecting to nodes 0 and 1
    [7,2], [7,3],             # Stabilize node 7 (top mid) horizontally by connecting to nodes 2 and 3
    [8,1], [8,2]              # Stabilize node 8 (right mid) vertically by connecting to nodes 1 and 2
])
E=210e9
A=0.01
L0=1.0

def plot_structure(noeuds, elements, deplacements=None):
    plt.figure()
    for element in elements:
        x = [noeuds[element[0], 0], noeuds[element[1], 0]]
        y = [noeuds[element[0], 1], noeuds[element[1], 1]]
        plt.plot(x, y, 'b-o')
    if deplacements is not None:
        noeuds_def = noeuds + deplacements.reshape((-1, 2))
        for element in elements:
            x = [noeuds_def[element[0], 0], noeuds_def[element[1], 0]]
            y = [noeuds_def[element[0], 1], noeuds_def[element[1], 1]]
            plt.plot(x, y, 'r--o')
    plt.axis('equal')
    plt.show()
def assemble_global_stiffness(noeuds, elements, E, A):
    n_dofs = noeuds.shape[0] * 2
    K_global = np.zeros((n_dofs, n_dofs))
    for element in elements:
        n1, n2 = element
        x1, y1 = noeuds[n1]
        x2, y2 = noeuds[n2]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        k_local = (E * A / L) * np.array([[ c*c,  c*s, -c*c, -c*s],
                                           [ c*s,  s*s, -c*s, -s*s],
                                           [-c*c, -c*s,  c*c,  c*s],
                                           [-c*s, -s*s,  c*s,  s*s]])
        dof_indices = [n1*2, n1*2+1, n2*2, n2*2+1]
        for i in range(4):
            for j in range(4):
                K_global[dof_indices[i], dof_indices[j]] += k_local[i, j]
    return K_global
def apply_boundary_conditions(K, F, fixed_dofs):
    for dof in fixed_dofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1
        F[dof] = 0
    return K, F
def main():
    K_global = assemble_global_stiffness(noeuds, elements, E, A)
    F_global = np.zeros(noeuds.shape[0] * 2)
    F_global[3] = -100000.0  # Apply a downward force at node 1 in y-direction
    fixed_dofs = [0, 1, 6, 7]  # Fix nodes 0 and 3 (both x and y)
    K_mod, F_mod = apply_boundary_conditions(K_global.copy(), F_global.copy(), fixed_dofs)
    displacements = np.linalg.solve(K_mod, F_mod)
    print("Nodal Displacements:\n", displacements.reshape((-1, 2)))
    plot_structure(noeuds, elements, displacements)

if __name__ == "__main__":
    main()