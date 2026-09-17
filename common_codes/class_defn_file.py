import numpy as np
import scipy.sparse as sp
import torch 
import math 
import matplotlib.pyplot as plt


class input_variables: 

    def __init__(self,
                N_b: int,                              # l
                N_f: int, 
                j_nn: float, 
                j_nnn: float,                              # n\sigma
                J_0_matrix: np.ndarray,  
                omega_number: float,                     #  J_0 is the hopping matrix J^{0}_{n\sigma, ms}
                omega_matrix: np.ndarray, 
                gamma_number: float,                   # phonon dispersion
                gamma_matrix: np.ndarray,                   #electon-phonon coupling, shape: N_b x N_f = g * delta(l, n \sigma)
                chemical_potential_val: float      # chemical potential
                ): 
        
        self.N_f = N_f
        self.N_b = N_b
        self.j_nn = j_nn
        self.j_nnn = j_nnn
        self.gamma_number = gamma_number
        self.omega_number = omega_number
        
        self.J_0 = torch.tensor(J_0_matrix, dtype=torch.complex128)        #shape: N_f x N_f: J_{n\sigma, m\sigma}
        self.gamma = torch.tensor(gamma_matrix, dtype =torch.complex128)   #shape: N_b x N_f: g_{l, n \sigma}: g * delta(l, n \sigma)
        self.omega = torch.tensor(omega_matrix, dtype=torch.complex128)    #shape: N_b x N_b: omega_{l, m} = omega * delta(l, m)
        self.chemical_potential_val = chemical_potential_val        #chemical potential: float

class variational_parameters:

    r""" 
    1. Stores the variational parameters Delta_r, Gamma_b, Gamma_m, lambda_bar_{l, n \sigma}
    2. Update them after every time step while solving the imaginary time evolution equations
    """

    def __init__(self, N_f:int, N_b:int): 
        self.delta_r = torch.zeros(2*N_b, dtype=torch.complex128)
        self.Gamma_b = torch.zeros((2*N_b, 2*N_b), dtype=torch.complex128)
        self.Gamma_m = torch.zeros((2*N_f, 2*N_f), dtype=torch.complex128)
        self.lambda_bar = torch.zeros((2*N_b, N_f), dtype=torch.complex128) # [0_{N_b x N_f}, lambda_{N_b x N_f}]

    def initialize_and_update_variational_parameters(self,
                                                      delta_r:np.ndarray, 
                                                      Gamma_b:np.ndarray, 
                                                      Gamma_m:np.ndarray, 
                                                      lambda_bar:np.ndarray): 
        """ 
        this function initializes the variational parameters
        and updates them after every time step in Imaginary time evolution
        """

        self.delta_r = torch.tensor(delta_r).to(dtype=torch.complex128)
        
        N_b = int(len(self.delta_r) / 2)

        delta_x_abs = torch.abs(self.delta_r[:int(N_b)])
        print("delta_x_absolute_avg, std: ", f"{torch.mean(delta_x_abs):.2e}", f"{torch.std(delta_x_abs):.2e}") 
        del delta_x_abs

        # np.savetxt("delta_r_current_qr.csv", delta_r.real, delimiter=",", fmt='%.2e')

        self.Gamma_b = torch.tensor(Gamma_b).to(dtype=torch.complex128)
        Gamma_b_xx_diag_abs = torch.abs(torch.diag(self.Gamma_b[:N_b, :N_b]))
        print("Gamma_b_xx_diag_abs_avg, std: ", f"{torch.mean(Gamma_b_xx_diag_abs):.3e}", f"{torch.std(Gamma_b_xx_diag_abs):.3e}")
        del Gamma_b_xx_diag_abs

        # np.savetxt("Gamma_b_current.csv", Gamma_b, delimiter=",", fmt='%.6e')

        self.Gamma_m = torch.tensor(Gamma_m).to(dtype=torch.complex128)

        self.lambda_bar = torch.tensor(lambda_bar).to(dtype=torch.complex128) # shape: 2N_b x N_f = [0_{N_b x N_f}, lambda_{N_b x N_f}]

        self.sanity_check_variational_parameters() 
#        print("variational parameters updated")

    def sigma(self, N_b, dtype=torch.float):
        mat_1 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=dtype)
        mat_2 = torch.eye(N_b, dtype=dtype)
        final_mat = torch.kron(mat_1, mat_2)
        return final_mat

    def check_majorana_covariance_matrix(self, Gamma_m: torch.Tensor):
        # Gamma_m^{2} = -I
        max_deviation_pure_state = torch.max(torch.abs(torch.matmul(Gamma_m, Gamma_m) + torch.eye(len(Gamma_m), dtype=Gamma_m.dtype)) )
        if max_deviation_pure_state > 1e-8:
            print(f"WARNING: Max Gamma_m^2 + I: {max_deviation_pure_state:.1e}")
        # if max_deviation_pure_state > 1e+1:
            # raise Exception("Majorana covariance matrix is not a pure state.")

        # Gamma_m = -Gamma_m^{T}
        max_deviation_anti_symmetry = torch.max(torch.abs(Gamma_m + Gamma_m.T))
        if max_deviation_anti_symmetry > 1e-10:
            print(f"WARNING: Max Gamma_m + Gamma_m^T: {max_deviation_anti_symmetry:.1e}")
            # raise Exception("The condition for the Majorana covariance matrix being Anti-Symmetric is not satisfied.")
        # if max_deviation_anti_symmetry > 1e+1:
            # raise Exception("AMajorana convariance matrix is not Anti-Symmetric.")
        
        # Gamma_m = real
        max_deviation_real = torch.max(torch.abs(torch.imag(Gamma_m)))
        if max_deviation_real > 1e-10:
            print(f"WARNING: Largest absolute value in the imaginary part of Gamma_m: {max_deviation_real:.1e} ")
            # raise Exception("The condition for the Majorana covariance matrix being Real is not satisfied.")
        # if max_deviation_real > 1e+1:
            # raise Exception("Majorana covariance matrix is not Real.")
                
        return
    

    def check_bosonic_covariance_matrix(self, Gamma_b: torch.Tensor):

        N_b = int(Gamma_b.shape[0]/ 2)
        sigma_mat = self.sigma(N_b, dtype=Gamma_b.dtype)
        # Gamma_b sigma_mat Gamma_b^{T} = sigma_mat
        deviation = torch.abs(torch.matmul(Gamma_b, torch.matmul(sigma_mat, Gamma_b.T)) - sigma_mat)

        max_deviation = torch.max(deviation)
        if max_deviation > 1e-8:
            print(f"WARNING: symplectic condition: {max_deviation:.1e}")
        # if max_deviation > 1e+1:            
            # raise Exception("Bosononic covariance matrix is not Symplectic.")
        
            # raise Exception("The condition for the quadrature covariance matrix to be Symplectic is not satisfied.")
        # Gamma_b = Gamma_b^{T}

        max_deviation_symmetric = torch.max(torch.abs(Gamma_b.T - Gamma_b))
        if max_deviation_symmetric > 1e-10:
            print(f"Largest deviation from symmetric condition: {max_deviation_symmetric:.1e}")
        # if max_deviation_symmetric > 1e+1:
            # raise Exception("Bosonic covariance matrix is not Symmetric.")
        
 
        max_deviation_real = torch.max(torch.abs(torch.imag(Gamma_b)))
        if max_deviation_real > 1e-10:
            print(f"Largest absolute value in the imaginary part of Gamma_b: {max_deviation_real:1e}")
        # if max_deviation_real > 1e+1:
            # raise Exception("Bosonic covariance matrix is not Real.")
       
        return

    def check_bosonic_quadrature_average_matrix(self, Delta_r: torch.Tensor):
        # Delta_r = real
        max_imag_delta = torch.max(torch.abs(torch.imag(Delta_r)))
        if max_imag_delta > 1e-10:
            max_index_imag_delta = torch.argmax(torch.abs(torch.imag(Delta_r)))
            print(f"WARNING: Largest absolute value in the imaginary part of Delta_r: {max_imag_delta:.1e} at index {max_index_imag_delta}")
            # raise Exception("The condition for the bosonic_quadrature_average covariance matrix being Real is not satisfied.")

        # if max_imag_delta > 1e+1:
            # raise Exception("Bosonic displacement is not Real.")
        
        return
    
    def sanity_check_variational_parameters(self):
        """
        Perform sanity checks on the variational parameters.
        """
        self.check_majorana_covariance_matrix(self.Gamma_m)
        self.check_bosonic_covariance_matrix(self.Gamma_b)
        self.check_bosonic_quadrature_average_matrix(self.delta_r)


class fermionic_correlation_matrices: 

    def __init__(self, variational_parameters_instance: variational_parameters):
        self.update_fermionic_correlation(variational_parameters_instance)

    def _calculate_all_matrices(self):
        G_m, N_f = self.Gamma_m, self.N_f
        
        eye_Nf = torch.eye(N_f, dtype=torch.complex128)
        G_m_11 = G_m[0:N_f, 0:N_f]
        G_m_12 = G_m[0:N_f, N_f:]
        G_m_21 = G_m[N_f:, 0:N_f]
        G_m_22 = G_m[N_f:, N_f:]

        self.c_c_dagger = 0.25 * (2 * eye_Nf - 1j * (G_m_11 + G_m_22) - G_m_12 + G_m_21)
        self.c_dagger_c = 0.25 * (2 * eye_Nf - 1j * (G_m_11 + G_m_22) + G_m_12 - G_m_21)
        self.c_c =                    0.25 * (-1j * (G_m_11 - G_m_22) + (G_m_12 + G_m_21))
        self.c_dagger_c_dagger =      0.25 * (-1j * (G_m_11 - G_m_22) - (G_m_12 + G_m_21))
        self.density_density_connected_correlator = self._calculate_density_density_connected_correlator()
        self.density_density_anticommutator_connected_correlator = self._calculate_density_density_anticommutator_connected_correlator()

        diag_real = torch.diag(self.c_dagger_c).real
        print("avrg,  std of diag(c_dagger_c).real = ", f"{torch.mean(diag_real):.3e}", f"{torch.std(diag_real):.3e}")
        print("Order of magnitude of avg diag(c_dagger_c).real:", int(torch.log10(torch.abs(torch.mean(diag_real)))))

        if torch.any(diag_real < 0):
            print("WARNING: Negative values in the diagonal of c_dagger_c")
            max_neg_val = torch.min(diag_real)
            print(f"Max negative value in the diagonal of c_dagger_c: {max_neg_val:.3e}")

        if torch.any(diag_real > 1):
            print("WARNING: Values in the diagonal of c_dagger_c greater than 1")
            print(f"Max value in the diagonal of c_dagger_c greater than 1: {torch.max(diag_real):.3e}")

 
    def _calculate_density_density_connected_correlator(self):
        final_mat = -torch.einsum('ji,ji->ji', self.c_dagger_c_dagger, self.c_c) + torch.einsum('ji,ji->ji', self.c_dagger_c, self.c_c_dagger)
        print("Max abs of density_density_connected_correlator:", torch.max(torch.abs(final_mat)).item())
        print("Sum of density_density_connected_correlator:", torch.sum(final_mat).item())
        return final_mat

    def _calculate_density_density_anticommutator_connected_correlator(self):
        final_mat = -torch.einsum('ji,jk->jik', self.c_dagger_c_dagger, self.c_c) - torch.einsum('ij,kj->jik', self.c_dagger_c_dagger, self.c_c)
        final_mat += torch.einsum('jk,ji->jik', self.c_dagger_c, self.c_c_dagger) + torch.einsum('ij,kj->jik', self.c_dagger_c, self.c_c_dagger)
        return final_mat

    def update_fermionic_correlation(self, variational_parameters_instance: variational_parameters):
        self.Gamma_m = variational_parameters_instance.Gamma_m
        self.N_f = int(self.Gamma_m.shape[0] / 2)
        self._calculate_all_matrices()

    def calculate_cdw_structure_factor(self):
        c_dagger_c_mat = self.c_dagger_c
        c_c_dagger_mat = self.c_c_dagger
        c_c_mat = self.c_c
        c_dagger_c_dagger_mat = self.c_dagger_c_dagger

        cdw_structure_factor_ij = (torch.einsum('i,j-> ij', torch.diag(c_dagger_c_mat), torch.diag(c_dagger_c_mat))
                                       - torch.einsum('ij,ij->ij', c_dagger_c_dagger_mat, c_c_mat)
                                       + torch.einsum('ij,ij->ij', c_dagger_c_mat, c_c_dagger_mat))
        return cdw_structure_factor_ij

    def calculate_sc_structure_factor(self):
        N_f = self.N_f
        c_up_c_down_mat = self.c_c[:int(N_f/2), int(N_f/2):]
        c_dagger_down_c_dagger_up_mat = self.c_dagger_c_dagger[int(N_f/2): , :int(N_f/2)]
        c_up_c_dagger_down_mat = self.c_c_dagger[:int(N_f/2), int(N_f/2):]
        c_down_c_dagger_up_mat = self.c_c_dagger[int(N_f/2):, :int(N_f/2)]
        c_up_c_dagger_up_mat = self.c_c_dagger[:int(N_f/2), :int(N_f/2)]
        c_down_c_dagger_down_mat = self.c_c_dagger[int(N_f/2):, int(N_f/2):] 

        sc_structure_factor_ij = (torch.einsum('i,j->ij', torch.diag(c_up_c_down_mat), torch.diag(c_dagger_down_c_dagger_up_mat))
                                       - torch.einsum('ij,ij->ij', c_up_c_dagger_down_mat, c_down_c_dagger_up_mat)
                                       + torch.einsum('ij,ji->ij', c_up_c_dagger_up_mat, c_down_c_dagger_down_mat))
        return sc_structure_factor_ij
    
#        print("fermionic correlation matrices updated")


class computed_variables:

    def __init__(self, N_b: int, N_f: int):
        """
        Data attribute which stores matrices and tensors which are computed from the 
        input variables and variational parameters as a numpy tensors. 
        """
        self.omega_bar_mat = torch.zeros((2 * N_b, 2 * N_b), dtype=torch.complex128)
        self.w_ln_mat = torch.zeros((2 * N_b, N_f, N_f), dtype=torch.complex128)
        self.J_i_j_mat = torch.zeros((N_f, N_f), dtype=torch.complex128)
        self.delta_gamma_tilde_mat = torch.zeros((2 * N_b, N_f), dtype=torch.complex128)
        self.Ve_i_j_mat = torch.zeros((N_f, N_f), dtype=torch.complex128)
        self.onsite_chemical_potential_mat = torch.zeros((N_f, N_f), dtype=torch.complex128)

    def compute_omega_bar(self, input_variables_instance: input_variables) -> None:
        self.omega_bar_mat = torch.kron(torch.eye(2), input_variables_instance.omega)
        return 
    
    def compute_w_ln(self, input_variables_instance: input_variables, variational_parameters_instance: variational_parameters) -> None:
        N_f = input_variables_instance.N_f
        N_b = input_variables_instance.N_b

        lambda_up = variational_parameters_instance.lambda_bar[N_b:, :N_f // 2]
        lambda_down = variational_parameters_instance.lambda_bar[N_b:, N_f // 2:]

        w_ln_up = lambda_up.unsqueeze(2) - lambda_up.unsqueeze(1)
        w_ln_down = lambda_down.unsqueeze(2) - lambda_down.unsqueeze(1)

        self.w_ln_mat[N_b:, :N_f // 2, :N_f // 2] = w_ln_up
        self.w_ln_mat[N_b:, N_f // 2:, N_f // 2:] = w_ln_down

        return
     
    def compute_J_i_j(self, input_variables_instance: input_variables, variational_parameters_instance: variational_parameters) -> None:
        delta_r = variational_parameters_instance.delta_r
        Gamma_b = variational_parameters_instance.Gamma_b
        J_0 = input_variables_instance.J_0
        w_ln_mat = self.w_ln_mat

        valid_indices = torch.nonzero(J_0, as_tuple=True)

        matrix_1 = torch.einsum('k, kij -> ij', delta_r, w_ln_mat)[valid_indices]
        matrix_1 = torch.remainder(torch.real(matrix_1), 2 * math.pi).to(dtype=torch.complex128)

        matrix_2 = torch.einsum('kij, kl, lij -> ij', w_ln_mat, Gamma_b, w_ln_mat)[valid_indices]

        self.J_i_j_mat = torch.zeros_like(J_0, dtype=torch.complex128)
        self.J_i_j_mat[valid_indices] = J_0[valid_indices] * torch.exp(-1j * matrix_1) * torch.exp(-0.5 * matrix_2)

        return 

    def compute_delta_gamma_tilde(self, input_variables_instance: input_variables, variational_parameters_instance: variational_parameters) -> None:
        N_b = input_variables_instance.N_b
        gamma = input_variables_instance.gamma
        lmbda = variational_parameters_instance.lambda_bar[N_b:, :]
        omega = input_variables_instance.omega

        delta_gamma_mat = gamma - torch.einsum('lk, kn -> ln', omega, lmbda)
        self.delta_gamma_tilde_mat = torch.cat((delta_gamma_mat, torch.zeros((N_b, lmbda.shape[1]), dtype=torch.complex128)), dim=0) #lambda.shape[1] = N_f
        return 
     
    def compute_Ve_i_j(self, input_variables_instance: input_variables, variational_parameters_instance: variational_parameters) -> None:
        N_b = input_variables_instance.N_b
        N_f = input_variables_instance.N_f
        lmbda = variational_parameters_instance.lambda_bar[N_b:, :]
        omega = input_variables_instance.omega
        gamma = input_variables_instance.gamma

        temp_mat_1 = torch.einsum('ki, kl, lj -> ij', lmbda, omega, lmbda)
        temp_mat_2 = torch.einsum('ki, kj -> ij', gamma, lmbda) 

        # g = gamma[0, 0]
        # lambda_ns1_ms2 = torch.cat((lmbda , lmbda), dim=0)
        # temp_mat_2 = g* (lambda_ns1_ms2)

        self.Ve_i_j_mat = 2.0 * (temp_mat_1 -  temp_mat_2 - temp_mat_2.T)
    
        return 
    
    def compute_onsite_chemical_potential(self, input_variables_instance: input_variables) -> None:
        mu = input_variables_instance.chemical_potential_val
        N_f = input_variables_instance.N_f

        self.onsite_chemical_potential_mat = mu * torch.eye(N_f, dtype=torch.complex128) - 0.5 * torch.diag(torch.diag(self.Ve_i_j_mat))
        return 
    
    
    def initialize_and_update_computed_variables(self, input_variables_instance: input_variables, variational_parameters_instance: variational_parameters) -> None:
        self.compute_omega_bar(input_variables_instance)
        self.compute_w_ln(input_variables_instance, variational_parameters_instance)
        self.compute_J_i_j(input_variables_instance, variational_parameters_instance)
        self.compute_delta_gamma_tilde(input_variables_instance, variational_parameters_instance)
        self.compute_Ve_i_j(input_variables_instance, variational_parameters_instance)
        self.compute_onsite_chemical_potential(input_variables_instance)
 #       print("computed variables updated")
        return None 


class functional_derivatives:

    def __init__(self, input_variables_instance: input_variables):
        N_b = input_variables_instance.N_b
        N_f = input_variables_instance.N_f
        self.h_delta_mat = torch.zeros(N_b, dtype=torch.complex128)
        self.h_b_mat = torch.zeros((2 * N_b, 2 * N_b), dtype=torch.complex128)
        self.h_m_mat = torch.zeros((2 * N_f, 2 * N_f), dtype=torch.complex128)
        self.O_delta_mat = torch.zeros(2 * N_b, dtype=torch.complex128)
        self.O_b_mat = torch.zeros((2 * N_b, 2 * N_b), dtype=torch.complex128)
        self.O_m_mat = torch.zeros((2 * N_f, 2 * N_f), dtype=torch.complex128)

    def compute_h_delta(self,variational_parameters_instance: variational_parameters,
                        computed_variables_instance: computed_variables, 
                        fermionic_correlation_matrices_instance: fermionic_correlation_matrices) -> None:
        
        omega_bar_mat = computed_variables_instance.omega_bar_mat 
        delta_r_mat = variational_parameters_instance.delta_r
        c_dagger_c = fermionic_correlation_matrices_instance.c_dagger_c
        delta_gamma_tilde_mat = computed_variables_instance.delta_gamma_tilde_mat
        j_i_j_mat = computed_variables_instance.J_i_j_mat
        w_ln_mat = computed_variables_instance.w_ln_mat

        final_mat = torch.einsum('kl, l-> k', omega_bar_mat, delta_r_mat)
        final_mat += 2 * torch.einsum('ki, i -> k', delta_gamma_tilde_mat, torch.diag(c_dagger_c))
        final_mat += -2j * torch.einsum('ij, ij, kij -> k', j_i_j_mat, c_dagger_c, w_ln_mat)

        self.h_delta_mat = final_mat

        return None

    def compute_h_b(self,computed_variables_instance: computed_variables, 
                    fermionic_correlation_matrices_instance: fermionic_correlation_matrices) -> None:
        
        omega_bar_mat = computed_variables_instance.omega_bar_mat
        j_i_j_mat = computed_variables_instance.J_i_j_mat
        c_dagger_c = fermionic_correlation_matrices_instance.c_dagger_c
        w_ln_mat = computed_variables_instance.w_ln_mat

        final_mat = omega_bar_mat - 2 * torch.einsum("ij,ij,kij,lij->kl", j_i_j_mat, c_dagger_c, w_ln_mat, w_ln_mat)

        self.h_b_mat = final_mat

        # print(f"norm of h_b: {torch.linalg.norm(0.25* self.h_b_mat)}")
        
        return None

    def compute_h_m(self, variational_parameter_instance: variational_parameters,
                    computed_variables_instance: computed_variables, 
                    fermionic_correlation_matrices_instance: fermionic_correlation_matrices) -> None:
        
        delta_r_mat = variational_parameter_instance.delta_r
        Ve_mat = computed_variables_instance.Ve_i_j_mat
        J_i_j_mat = computed_variables_instance.J_i_j_mat
        c_dagger_c_expectation_value_mat = fermionic_correlation_matrices_instance.c_dagger_c
        c_c_expectation_value_mat = fermionic_correlation_matrices_instance.c_c
        delta_gamma_tilde_mat = computed_variables_instance.delta_gamma_tilde_mat
        onsite_chemical_potential_matrix = computed_variables_instance.onsite_chemical_potential_mat

        epsilon_i_j_mat = J_i_j_mat - onsite_chemical_potential_matrix
        epsilon_i_j_mat += torch.diag(torch.einsum('ki, k -> i', delta_gamma_tilde_mat, delta_r_mat))
        epsilon_i_j_mat += torch.diag(torch.einsum('ij, j -> i', Ve_mat, torch.diag(c_dagger_c_expectation_value_mat)))
        epsilon_i_j_mat += -torch.einsum('ij, ji -> ij', Ve_mat, c_dagger_c_expectation_value_mat)

        Delta_m_mat = torch.einsum('ij, ji -> ij', Ve_mat, c_c_expectation_value_mat)
        N_f = Delta_m_mat.shape[0] // 2

        Delta_m_12_block_diag = torch.diag(Delta_m_mat[:N_f, N_f:])

        avg_12_block = torch.mean(torch.abs(Delta_m_12_block_diag))
        std_12_block = torch.std(torch.abs(Delta_m_12_block_diag))

        print(f"Average of |Delta_m|: {avg_12_block:.3e}, Std: {std_12_block:.3e}")

        final_mat = 0.5 * (torch.kron(torch.tensor([[-1j, 1], [-1, -1j]]), epsilon_i_j_mat) +
                           torch.kron(torch.tensor([[1j, 1], [-1, 1j]]), epsilon_i_j_mat.t().contiguous()) +
                           torch.kron(torch.tensor([[-1j, -1], [-1, 1j]]), Delta_m_mat) +
                           torch.kron(torch.tensor([[-1j, 1], [1, 1j]]), Delta_m_mat.t().conj().contiguous()))

        self.h_m_mat = final_mat 


        return None 

    def compute_O_delta(self, time_derivative_lambda_bar: np.ndarray, 
                        fermionic_correlation_matrices_instance: fermionic_correlation_matrices) -> None:
        
        time_derivative_lambda_bar_tensor = torch.tensor(time_derivative_lambda_bar, dtype=torch.complex128)
        c_dagger_c_expectation_value_mat = fermionic_correlation_matrices_instance.c_dagger_c

        final_mat = 2j * torch.einsum('n, ln -> l', torch.diag(c_dagger_c_expectation_value_mat), time_derivative_lambda_bar_tensor)

        self.O_delta_mat = final_mat

        return None

    # def compute_O_b(self) -> None:
    #     self.O_b_mat.zero_()

    def compute_O_m(self, time_derivative_lambda_bar: np.ndarray, 
                    input_variables_instance: input_variables, 
                    variational_parameter_instance: variational_parameters) -> None:
        
        N_f = input_variables_instance.N_f
        delta_r = variational_parameter_instance.delta_r
        time_derivative_lmbda_bar_tensor = torch.tensor(time_derivative_lambda_bar, dtype=torch.complex128)

        diagonal_o_m = torch.diag(1j * torch.einsum('l, ln -> n', delta_r, time_derivative_lmbda_bar_tensor))
        final_mat = torch.zeros(2 * N_f, 2 * N_f, dtype=torch.complex128)
        # final_mat[N_f:, :N_f] = diagonal_o_m
        # final_mat[:N_f, N_f:] = -diagonal_o_m
        final_mat[N_f:, :N_f] =  -diagonal_o_m
        final_mat[:N_f, N_f:] = diagonal_o_m

        self.O_m_mat = final_mat

        return None
    
    
    def initialize_and_update_functional_derivatives(self, time_derivative_lambda_bar: np.ndarray, 
                                                     input_variables_instance: input_variables, 
                                                     variational_parameters_instance: variational_parameters, 
                                                     computed_variables_instance: computed_variables, 
                                                     fermionic_correlation_matrices_instance: fermionic_correlation_matrices) -> None: 

        self.compute_h_delta(variational_parameters_instance, computed_variables_instance, fermionic_correlation_matrices_instance)
        self.compute_h_b(computed_variables_instance, fermionic_correlation_matrices_instance)
        self.compute_h_m(variational_parameters_instance, computed_variables_instance, fermionic_correlation_matrices_instance)
        self.compute_O_delta(time_derivative_lambda_bar, fermionic_correlation_matrices_instance)
        # self.compute_O_b()
        self.compute_O_m(time_derivative_lambda_bar, input_variables_instance, variational_parameters_instance)
  #      print("functional derivatives updated")

        return None

def compute_energy_expectation(input_variables_instance: input_variables, 
                        variational_parameters_instance: variational_parameters, 
                        computed_variables_instance: computed_variables, 
                        fermionic_correlation_matrices_instance: fermionic_correlation_matrices) -> float:
    """
    Compute and prints the energy expectation value of the system.
    """
    N_f = input_variables_instance.N_f
    j_i_j_mat = computed_variables_instance.J_i_j_mat
    c_dagger_c_mat = fermionic_correlation_matrices_instance.c_dagger_c

    hopping_energy = torch.einsum('ij, ij ->', j_i_j_mat, c_dagger_c_mat)

    omega_bar_mat = computed_variables_instance.omega_bar_mat
    Gamma_b_mat = variational_parameters_instance.Gamma_b
    delta_r_mat = variational_parameters_instance.delta_r

    phonon_energy = 0.25 * (
         torch.einsum("ij, ji->", omega_bar_mat, Gamma_b_mat) +  torch.einsum("i, ij, j ->", delta_r_mat, omega_bar_mat, delta_r_mat) 
         - 2* torch.trace(omega_bar_mat) 
         ) 

    delta_gamma_tilde = computed_variables_instance.delta_gamma_tilde_mat
    
    effective_electron_phonon_coupling_energy = torch.einsum("ln, l, n ->", delta_gamma_tilde, delta_r_mat, torch.diag(c_dagger_c_mat))


    c_dagger_c_dagger_mat = fermionic_correlation_matrices_instance.c_dagger_c_dagger
    c_dagger_c_mat = fermionic_correlation_matrices_instance.c_dagger_c
    c_c_mat = fermionic_correlation_matrices_instance.c_c
    Ve_ij_mat = computed_variables_instance.Ve_i_j_mat
    effective_chemical_potential = input_variables_instance.chemical_potential_val * torch.eye(N_f) - 0.5 * Ve_ij_mat

    effective_electron_electron_interaction_energy = 0.5 * (
        torch.einsum("ij, ij, ji ->", Ve_ij_mat, c_dagger_c_dagger_mat, c_c_mat) 
        + torch.einsum("ij, ij, ji ->", Ve_ij_mat, c_dagger_c_mat, c_dagger_c_mat)
        - torch.einsum("ij, ij, ji ->", Ve_ij_mat, c_dagger_c_mat, c_dagger_c_mat)
        - 2 * torch.einsum("i, i-> ", torch.diag(effective_chemical_potential), torch.diag(c_dagger_c_mat))
    )

    total_energy = hopping_energy + phonon_energy + effective_electron_phonon_coupling_energy + effective_electron_electron_interaction_energy
    
    print(f"Total Energy: {total_energy.real:.8e}, hopp: {hopping_energy.real:.2e}, p: {phonon_energy.real:.2e}, e-p: {effective_electron_phonon_coupling_energy.real:.2e}, e-e: {effective_electron_electron_interaction_energy.real:.2e}")
    return total_energy.real


