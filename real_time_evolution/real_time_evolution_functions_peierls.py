import numpy as np
from scipy.linalg import qr
import torch 


from common_codes import class_defn_file_peierls as cdf

# def real_time_EOM_for_phase(input_variables_instance:cdf.input_variables, 
#                             variational_parameters_instance:cdf.variational_parameters, 
#                             computed_variables_instance:cdf.computed_variables,
#                             functional_derivatives_instance:cdf.functional_derivatives)->np.ndarray:
    
#     Energy_expectation_value = cdf.compute_energy_expectation(input_variables_instance, variational_parameters_instance,
#                                                               computed_variables_instance, functional_derivatives_instance)
#     print("Energy = ", Energy_expectation_value)
    
#     delta_r_tensor = variational_parameters_instance.delta_r
#     h_delta_tensor = functional_derivatives_instance.h_delta_mat
#     O_delta_tensor = functional_derivatives_instance.O_delta_mat
#     h_t_delta_tensor = (h_delta_tensor - 1j*O_delta_tensor)
#     first_term  = torch.trace(torch.matmul(delta_r_tensor.t(), h_t_delta_tensor))
    
#     del delta_r_tensor
#     del h_delta_tensor
#     del O_delta_tensor
#     del h_t_delta_tensor



#     gamma_b_tensor = variational_parameters_instance.Gamma_b
#     h_b_tensor = functional_derivatives_instance.h_b_mat
#     O_b_tensor = functional_derivatives_instance.O_b_mat
#     h_b_t_tensor = (h_b_tensor - 1j*O_b_tensor)
#     second_term = torch.trace(torch.matmul(h_b_t_tensor, gamma_b_tensor))
#     del gamma_b_tensor
#     del h_b_tensor
#     del O_b_tensor
#     del h_b_t_tensor


#     gamma_m_tensor = variational_parameters_instance.Gamma_m
#     h_m_tensor = functional_derivatives_instance.h_m_mat
#     O_m_tensor = functional_derivatives_instance.O_m_mat
#     h_m_t_tensor = (h_m_tensor - 1j*O_m_tensor)
#     third_term = torch.trace(torch.matmul(h_m_t_tensor, gamma_m_tensor))
#     third_term = torch.trace(torch.matmul(h_m_t_tensor, gamma_m_tensor))
#     del gamma_m_tensor
#     del h_m_tensor
#     del O_m_tensor
#     del h_m_t_tensor

#     final_term = - Energy_expectation_value + first_term + second_term - third_term
    

#     return np.array(final_term.numpy(), dtype=np.complex128)


def real_time_EOM_for_bosonic_average(input_variables_instance:cdf.input_variables, 
                                           variational_parameters_instance:cdf.variational_parameters, 
                                           functional_derivatives_instance:cdf.functional_derivatives)->np.ndarray:
     
    N_b = input_variables_instance.N_b
    # Gamma_b = variational_parameters_instance.Gamma_b
    
    
    sigma = torch.zeros((2 * N_b, 2 * N_b), dtype=torch.complex128)
    sigma[:N_b, N_b:] = torch.eye(N_b, dtype=torch.complex128)
    sigma[N_b:, :N_b] = -torch.eye(N_b, dtype=torch.complex128)

    h_delta_tensor = functional_derivatives_instance.h_delta_mat 
    O_delta_tensor = functional_derivatives_instance.O_delta_mat
    

    h_t_delta_tensor = (h_delta_tensor - 1j*O_delta_tensor)
    # h_t_delta_tensor = (h_delta_tensor)

    final_mat =  torch.matmul(sigma, h_t_delta_tensor)

    return (final_mat.numpy())

def real_time_EOM_from_bosonic_covariance(input_variables_instance:cdf.input_variables,
                                          variational_parameters_instance:cdf.variational_parameters, 
                                          functional_derivatives_instance:cdf.functional_derivatives)->torch.Tensor:
    
    Gamma_b_tensor = variational_parameters_instance.Gamma_b
    
    N_b = input_variables_instance.N_b

    sigma = torch.zeros((2 * N_b, 2 * N_b), dtype=torch.complex128)
    sigma[:N_b, N_b:] = torch.eye(N_b, dtype=torch.complex128)
    sigma[N_b:, :N_b] = -torch.eye(N_b, dtype=torch.complex128)

 
    h_b_tensor = functional_derivatives_instance.h_b_mat
    O_b_tensor = functional_derivatives_instance.O_b_mat
    # O_b_tensor = torch.zeros((2*N_b, 2*N_b), dtype=torch.complex128)

    h_b_t_tensor = (h_b_tensor - 1j*O_b_tensor)
    h_b_t_tensor = (h_b_tensor)

    final_mat = torch.matmul(sigma, torch.matmul(h_b_t_tensor, Gamma_b_tensor)) \
                - torch.matmul(Gamma_b_tensor, torch.matmul(h_b_t_tensor, sigma)) 
    

    return final_mat.numpy()

def real_time_EOM_for_fermionic_covariance(variational_parameters_instance:cdf.variational_parameters,
                                           functional_derivatives_instance:cdf.functional_derivatives)->torch.Tensor:
    
    Gamma_m_tensor = variational_parameters_instance.Gamma_m
    h_m_tensor = functional_derivatives_instance.h_m_mat
    O_m_tensor = functional_derivatives_instance.O_m_mat
    
    # O_m_tensor = torch.zeros((2*N_f, 2*N_f), dtype=torch.complex128)

    h_m_t_tensor = (h_m_tensor- 1j*O_m_tensor)

    final_mat =  torch.matmul(h_m_t_tensor, Gamma_m_tensor) \
                - torch.matmul(Gamma_m_tensor, h_m_t_tensor)
    
    return final_mat.numpy()


 
def compute_rhs_real_time_non_gaussian_EOM(input_variables_instance:cdf.input_variables,
                                            variational_parameters_instance:cdf.variational_parameters, 
                                            computed_variables_instance:cdf.computed_variables,
                                              fermionic_correlations_instance:cdf.fermionic_correlation_matrices)->torch.tensor:
    
    N_b = input_variables_instance.N_b
    N_f = input_variables_instance.N_f

    Gamma_b_tensor = variational_parameters_instance.Gamma_b

    density_density_connected_correlation_tensor =  fermionic_correlations_instance.density_density_connected_correlator
    density_density_anticommutator_tensor =  fermionic_correlations_instance.density_density_anticommutator_connected_correlator
    c_dagger_c_expectation_value_tensor = fermionic_correlations_instance.c_dagger_c
    j_i_j_tensor = computed_variables_instance.J_i_j_mat
    w_ln_tensor = computed_variables_instance.w_ln_mat
    delta_gamma_tilde_tensor = computed_variables_instance.delta_gamma_tilde_mat


    sigma = torch.zeros((2 * N_b, 2 * N_b), dtype=torch.complex128)
    sigma[:N_b, N_b:] = torch.eye(N_b, dtype=torch.complex128)
    sigma[N_b:, :N_b] = -torch.eye(N_b, dtype=torch.complex128)

    final_mat = torch.zeros(2*N_b, N_f, dtype=torch.complex128)

    final_mat += 0.5*1j*torch.einsum("lk, kim, im, nim  -> ln", sigma, w_ln_tensor, j_i_j_tensor, density_density_anticommutator_tensor)
    final_mat += - torch.einsum("lk, km, mn -> ln", sigma, delta_gamma_tilde_tensor, density_density_connected_correlation_tensor) 

    real_part = torch.einsum("lk, knm, nm -> ln", Gamma_b_tensor, w_ln_tensor, torch.real(j_i_j_tensor * c_dagger_c_expectation_value_tensor).to(dtype=torch.complex128))

    final_mat += real_part

    return final_mat

def real_time_EOM_for_lambda_bar_lstsq(input_variables_instance:cdf.input_variables, 
                                       variational_parameters_instance:cdf.variational_parameters, 
                                       computed_variables_instance:cdf.computed_variables,
                                         fermionic_correlations_instance:cdf.fermionic_correlation_matrices)->torch.tensor:
    
    N_b = input_variables_instance.N_b
    N_f = input_variables_instance.N_f
 
    sigma = torch.zeros((2 * N_b, 2 * N_b), dtype=torch.complex128)
    sigma[:N_b, N_b:] = torch.eye(N_b, dtype=torch.complex128)
    sigma[N_b:, :N_b] = -torch.eye(N_b, dtype=torch.complex128)

    density_density_connected_correlation_tensor =  fermionic_correlations_instance.density_density_connected_correlator
    rank = torch.linalg.matrix_rank(density_density_connected_correlation_tensor)
    cond_number = torch.linalg.cond(density_density_connected_correlation_tensor)
    print(f"Rank: {rank}, Condition Number: {cond_number:.2e}")

    is_symmetric = torch.allclose(density_density_connected_correlation_tensor, density_density_connected_correlation_tensor.T, rtol=1e-05, atol=1e-10)

    if is_symmetric:
        print("density_density_connected_correlation_tensor is symmetric")
    else:
        print("density_density_connected_correlation_tensor is NOT symmetric")

    rhs_real_time_non_gaussian_EOM_tensor = compute_rhs_real_time_non_gaussian_EOM(input_variables_instance, variational_parameters_instance,
                                                                                           computed_variables_instance, fermionic_correlations_instance)
    
    transpose_rhs_real_time_non_gaussian_EOM_tensor = torch.transpose(rhs_real_time_non_gaussian_EOM_tensor, 0, 1)

    sigma_time_derivative_lambda_bar_transpose = torch.linalg.lstsq(density_density_connected_correlation_tensor, transpose_rhs_real_time_non_gaussian_EOM_tensor,  driver = "gelsss").solution

    correctness_of_solution = torch.max(torch.abs(torch.matmul(density_density_connected_correlation_tensor, sigma_time_derivative_lambda_bar_transpose)-  transpose_rhs_real_time_non_gaussian_EOM_tensor))
    print("correctness_of_solution = ", correctness_of_solution)


    sigma_time_derivative_lambda_bar = torch.transpose(sigma_time_derivative_lambda_bar_transpose, 0, 1)

    time_derivative_lambda_bar = torch.matmul(-sigma, sigma_time_derivative_lambda_bar) # sigma^{2} = -1

    time_derivative_lambda_bar_x = time_derivative_lambda_bar[:N_b, :]
    max_time_derivative_lambda_bar_x = torch.max(torch.abs(time_derivative_lambda_bar_x))
    print("max_time_derivative_lambda_bar_x = ", max_time_derivative_lambda_bar_x)

    time_derivative_lambda_bar_p = time_derivative_lambda_bar[N_b:, :]
    max_time_derivative_lambda_bar_p = torch.max(torch.abs(time_derivative_lambda_bar_p))
    print("max_time_derivative_lambda_bar_p = ", max_time_derivative_lambda_bar_p)
    
    
    return time_derivative_lambda_bar.numpy().astype(np.complex128)