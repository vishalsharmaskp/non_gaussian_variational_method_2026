import numpy as np
import thewalrus as tw
import scipy.linalg as spla

# to save date to file code

def sigma(N_b, dtype = "float"):
    mat_1 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    mat_2 = np.identity(N_b)
    final_mat = np.kron(mat_1, mat_2)   
    final_mat = final_mat.astype(dtype)
    return final_mat

def delta(i, j): 
    if (i == j): 
        return 1
    else: 
        return 0
    
def coordinate_array_creator_function(L_min, L_max, number_of_points, spin_index = True, dtype = "float"): 

    if (len(np.shape(L_max)) > 1 or len(np.shape(L_min)) > 1): 
       print("L_max and L_min are not 1D arrays")
    
    elif(np.size(L_max) != np.size(L_min)): 
       print("L_max and L_min are not of the same size")

    elif (np.size(number_of_points) != 1 and np.size(number_of_points) != np.size(L_max)): 
       print("number_of_points is not a 1d array or does not have same length as L_max and L_min")
    
    else: 
        if (np.size(number_of_points) == 1 and np.size(L_max) >  1): 
            number_of_points = np.tile(number_of_points, np.size(L_max))

        if (np.size(L_max) == 1): 
            coordinate_array = np.linspace(L_min, L_max, number_of_points, endpoint = False)
        
        else: 
            slices = [slice(L_min[dim], L_max[dim], (L_max[dim] - L_min[dim])/number_of_points[dim]) for dim in range(np.size(L_max))]

            coordinate_array = np.mgrid[slices].reshape(np.size(L_max), -1).T

        if (spin_index == True):
            coordinate_array = np.append(coordinate_array, coordinate_array, axis = 0)
        
        return coordinate_array 
    

    
def creating_J_0_matrix_PBC(position_array:np.ndarray, J_0, Length, spin_index = True) -> np.ndarray:
    
    N_f = position_array.shape[0]
    nearest_neighbour_vector = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    if (spin_index == True): 
        size_of_individual_block = int(N_f/2)
    else:
        size_of_individual_block = N_f
    
    J_0_block_matrix = np.zeros((int(size_of_individual_block), int(size_of_individual_block)))
    for i in range(size_of_individual_block): 
        for j in range(size_of_individual_block): 
            for ele in nearest_neighbour_vector: 
                if (False in set(np.mod(position_array[i] + ele, Length) == position_array[j])): 
                    continue
                J_0_block_matrix[i, j] = J_0

    if (spin_index == True): 
        J_0_matrix = np.kron([[1, 0], [0, 1]], J_0_block_matrix)
    else: 
        J_0_matrix = J_0_block_matrix
    
    return J_0_matrix

def creating_J_0_matrix_with_NNN_PBC(position_array: np.ndarray, J_nn, J_nnn, Length, spin_index=True) -> np.ndarray:
    N_f = position_array.shape[0]

    nearest_neighbour_vector = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    next_nearest_neighbour_vector = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

    if spin_index:
        size_of_individual_block = int(N_f / 2)
    else:
        size_of_individual_block = N_f

    J_0_block_matrix = np.zeros((size_of_individual_block, size_of_individual_block))

    for i in range(size_of_individual_block):
        for j in range(size_of_individual_block):
            # Nearest neighbour check
            for vec in nearest_neighbour_vector:
                if np.all(np.mod(position_array[i] + vec, Length) == position_array[j]):
                    J_0_block_matrix[i, j] = J_nn
            # Next-nearest neighbour check
            for vec in next_nearest_neighbour_vector:
                if np.all(np.mod(position_array[i] + vec, Length) == position_array[j]):
                    J_0_block_matrix[i, j] = J_nnn

    if spin_index:
        J_0_matrix = np.kron(np.eye(2), J_0_block_matrix)
    else:
        J_0_matrix = J_0_block_matrix

    return J_0_matrix




def create_triangular_lattice(n_x, n_y, spin_index=True, dtype="float"):
    """
    Create 2D triangular lattice coordinates with row-major ordering:
    - n_x: number of points along a1 (x-like)
    - n_y: number of points along a2 (y-like)
    """
    a1 = np.array([1.0, 0.0], dtype=dtype)
    a2 = np.array([0.5, np.sqrt(3)/2], dtype=dtype)

    coords = []
    for j in range(n_y):          # step along a2
        for i in range(n_x):      # step along a1
            pos = i * a1 + j * a2
            coords.append(pos)

    coords = np.array(coords, dtype=dtype)

    if spin_index:
        coords = np.append(coords, coords, axis=0)
    return coords


def create_triangular_hopping_matrix(position_array, Length, t=-1.0, spin_index=True, tol=1e-6):
    N = position_array.shape[0]
    number_of_sites = N // 2 if spin_index else N
    print(f"Number of sites: {number_of_sites}")

    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    nn_vectors = [a1, -a1, a2, -a2, a2 - a1, a1 - a2]
    A = np.array([a1, a2]).T        # 2x2 matrix, columns are primitive vectors
    A_inv = np.linalg.inv(A)        # to convert to lattice coordinates

    Lx = Length  # number of repeats along a1
    Ly = Length  # number of repeats along a2

    J_0_block_spinless = np.zeros((int(number_of_sites), int(number_of_sites)))

    for i in range(number_of_sites):
        for j in range(number_of_sites):
            for ele in nn_vectors:
                # Convert to lattice coordinates (n1, n2)
                lat_coords = A_inv @ (position_array[i] + ele)
                target_coords = A_inv @ position_array[j]
                
                # Apply periodic BC in lattice coordinates
                lat_coords_mod = np.mod(lat_coords, [Lx, Ly])
                
                # Check match
                if np.allclose(lat_coords_mod, target_coords, atol=1e-8):
                    # print(f"  --> Nearest neighbor found: i={i}, pos={position_array[i]}, j={j}, pos={position_array[j]}")
                    J_0_block_spinless[i, j] = t
    # Spin duplication
    print(f"  --> J_0_block_spinless shape: {J_0_block_spinless.shape}")
    if spin_index == True:
        hopping_matrix = np.kron(np.eye(2), J_0_block_spinless)
    else:
        hopping_matrix = J_0_block_spinless
    print(f"  --> H matrix shape: {hopping_matrix.shape}")

    return hopping_matrix

# the omega matrix for einstein type is created directly in the main.py file. 
def creating_omega_matrix_with_nearest_neighbour(bosonic_position_array, omega_0:float, omega_1:float, sign_x:int, sign_y:int, Length:int):

    assert omega_0 > 0, "omega_0 should be positive"
    assert omega_1 > 0, "omega_1 should be positive"
    assert sign_x == 1 or sign_x == -1, "sign_x should be 1 or -1"
    assert sign_y == 1 or sign_y == -1, "sign_y should be 1 or -1"

    N_b = int(bosonic_position_array.shape[0]) 

    omega_einstein_onsite = np.identity(N_b) * omega_0
    omega_bar_einstein = np.kron(np.identity(2), omega_einstein_onsite)

    nearest_neighbour_vector_x = np.array([[1, 0], [-1, 0]])
    nearest_neighbour_vector_y = np.array([[0, 1], [0, -1]])

    omega_nn_mat = np.zeros((N_b, N_b))
    for i in range(N_b):
        for j in range(N_b):
            for vec in nearest_neighbour_vector_x:
                if np.all(np.mod(bosonic_position_array[i] + vec, Length) == bosonic_position_array[j]):
                    omega_nn_mat[i, j] = sign_x 
            for vec in nearest_neighbour_vector_y:
                if np.all(np.mod(bosonic_position_array[i] + vec, Length) == bosonic_position_array[j]):
                    omega_nn_mat[i, j] = sign_y 

    omega_bar_hook = np.zeros((2 * N_b, 2 * N_b))

    omega_hook_onsite = (2 * omega_1 * omega_1) / omega_0 * np.eye(N_b)
    omega_hook_nearest_neighbour = (omega_1 * omega_1) / omega_0 * omega_nn_mat
    omega_bar_hook[:N_b, :N_b] = omega_hook_onsite + omega_hook_nearest_neighbour

    final_omega_bar_matrix = omega_bar_einstein + omega_bar_hook

    assert np.shape(final_omega_bar_matrix) == (2 * N_b, 2 * N_b), "Final omega matrix shape is incorrect"

    return final_omega_bar_matrix

def creating_J_0_matrix_OBC(position_array:np.ndarray, J_0, Length, spin_index = True) -> np.ndarray:
    
    N_f = position_array.shape[0]
    nearest_neighbour_vector = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    if (spin_index == True): 
        size_of_individual_block = int(N_f/2)
    else:
        size_of_individual_block = N_f
    
    J_0_block_matrix = np.zeros((int(size_of_individual_block), int(size_of_individual_block)))
    for i in range(size_of_individual_block): 
        for j in range(size_of_individual_block): 
            for ele in nearest_neighbour_vector:
                if np.array_equal(position_array[i] + ele, position_array[j]):
                    J_0_block_matrix[i, j] = J_0

    if (spin_index == True): 
        J_0_matrix = np.kron([[1, 0], [0, 1]], J_0_block_matrix)
    else: 
        J_0_matrix = J_0_block_matrix
    
    return J_0_matrix


def peierls_phase_matrix(position_array_spinful, vector_potential_polarizations, spin_index=True):
    number_of_points = int(len(position_array_spinful)/2)
    phase_matrix = np.zeros((number_of_points, number_of_points), dtype=np.complex128)
    print(phase_matrix.shape)
    for i in range(number_of_points):
        for j in range(number_of_points):
            position_array_diff = position_array_spinful[i] - position_array_spinful[j]
            phase_matrix[i, j] = np.dot(vector_potential_polarizations, position_array_diff)
    if spin_index:
        phase_matrix = np.kron(np.eye(2), phase_matrix)
    return phase_matrix



def creating_fourier_matrix(momentum_space_array:np.ndarray,position_space_array:np.ndarray):
    if(len(position_space_array.shape)==1):
        fourier_sapce_matrix = np.exp(1j*np.einsum('q,m->qm',momentum_space_array,position_space_array))
    if(len(position_space_array.shape)>1):
        fourier_sapce_matrix = np.exp(1j*np.einsum('ql,ml->qm',momentum_space_array,position_space_array))
    return(fourier_sapce_matrix)


def initialise_delta_R_matrix(N_b:int, seed:float) -> np.ndarray:
    print("shape delta_R = ", np.shape(np.random.rand(2*N_b)))
    return np.random.rand(2*N_b)


def initialize_gamma_m_matrix(N_f):
    O_m = tw.random.random_interferometer(2*N_f, real= True)
    sigma_1 = sigma(N_f)
    gamma_m = -np.matmul(O_m, np.dot(sigma_1, O_m.T))
    print("shape Gamma_m = ", np.shape(gamma_m))
    # print(f"Max value of Gamma_m = {np.max(np.abs(gamma_m))}")
    return gamma_m

def initialize_gamma_b_matrix(N_b):
    S_b = tw.random.random_symplectic(N_b)
    gamma_b = np.matmul(S_b, S_b.T)
    # print("shape Gamma_b = ", np.shape(gamma_b))
    # print(f"Max value of Gamma_b = {np.max(np.abs(gamma_b))}")
    return gamma_b


def intialize_lambda_bar(N_b, N_f): 
    lambda_spinless = 2*(np.random.rand(N_b, int(N_f/2))) - 1
    lmbda = np.append(lambda_spinless, lambda_spinless, axis = 1)  # we have assume \lambda_{l, n \sigma} does not depend on \sigma. 
    lambda_bar = np.append(np.zeros((N_b, N_f)), lmbda, axis = 0)

    return lambda_bar   










# if __name__ == "__main__": 
    # import matplotlib.pyplot as plt
    # print("This is a test")
    # position_space_grid = coordinate_array_creator_function([0, 0], [6, 6], 6, spin_index = False)

    # omega_matrix = creating_omega_matrix_with_nearest_neighbour(position_space_grid, 5, 2, 1, 1, Length=6)
    # plt.pcolormesh(omega_matrix)
    # plt.colorbar()
    # plt.show()


    # print(position_space_grid)    
    # creating_J_0_matrix_PBC = creating_J_0_matrix_PBC(position_space_grid, -1, 6, spin_index = True)
    # plt.pcolormesh(creating_J_0_matrix_PBC)
    # plt.colorbar()
    # plt.show()

    # J_0_matrix_PBC = creating_J_0_matrix_with_NNN_PBC(position_space_grid, -1, 0, 10, spin_index = True)
    # print(J_0_matrix_PBC)
    # plt.pcolormesh(J_0_matrix_PBC)
    # plt.colorbar()
    # plt.show()

    # J_n_nn_matrix_PBC = creating_J_0_matrix_with_NNN_PBC(position_space_grid, -1, -0.2, 10, spin_index = True)
    # print(J_n_nn_matrix_PBC)
    # plt.pcolormesh(J_n_nn_matrix_PBC)
    # plt.colorbar()
    # plt.show()


    # creating_omega_matrix_with_nearest_neighbour= creating_omega_matrix_with_NNN(position_space_grid, 5, 2, 10, position_space_grid.shape[0])
    # print(creating_omega_matrix_with_nearest_neighbour)
    # plt.pcolormesh(creating_omega_matrix_with_nearest_neighbour)
    # plt.colorbar()
    # plt.show()
    # J_0_matrix_OBC = creating_J_0_matrix_OBC(position_space_grid, -1, 10, spin_index = True)
    # print(J_0_matrix_OBC)
    # plt.pcolormesh(J_0_matrix_OBC)
    # plt.colorbar()
    # plt.show()