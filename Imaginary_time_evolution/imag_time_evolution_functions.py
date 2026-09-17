import numpy as np
from scipy.linalg import qr
from scipy.linalg import lstsq
import torch 


from common_codes import class_defn_file as cdf


def imaginary_time_equation_of_motion_for_bosonic_average(input_variables_instance:cdf.input_variables, 
                                           variational_parameters_instance:cdf.variational_parameters, 
                                           functional_derivatives_instance:cdf.functional_derivatives)->np.ndarray:
     
    N_b = input_variables_instance.N_b
    Gamma_b = variational_parameters_instance.Gamma_b
    
    
    sigma = torch.zeros((2 * N_b, 2 * N_b), dtype=torch.complex128)
    sigma[:N_b, N_b:] = torch.eye(N_b, dtype=torch.complex128)
    sigma[N_b:, :N_b] = -torch.eye(N_b, dtype=torch.complex128)

    h_delta_tensor = functional_derivatives_instance.h_delta_mat 
    O_delta_tensor = functional_derivatives_instance.O_delta_mat


    final_mat =  -torch.matmul(Gamma_b, h_delta_tensor) - 1j*torch.matmul(sigma, O_delta_tensor)

    return (final_mat.numpy())

def imaginary_time_equation_of_motion_from_bosonic_covariance(
         input_variables_instance:cdf.input_variables,
         variational_parameters_instance:cdf.variational_parameters, 
         functional_derivatives_instance:cdf.functional_derivatives)->torch.Tensor:
    
    Gamma_b_tensor = variational_parameters_instance.Gamma_b
    
    N_b = input_variables_instance.N_b

    sigma = torch.zeros((2 * N_b, 2 * N_b), dtype=torch.complex128)
    sigma[:N_b, N_b:] = torch.eye(N_b, dtype=torch.complex128)
    sigma[N_b:, :N_b] = -torch.eye(N_b, dtype=torch.complex128)

 
    h_b_tensor = functional_derivatives_instance.h_b_mat
    O_b_tensor = functional_derivatives_instance.O_b_mat


    final_mat = torch.matmul(sigma.t(), torch.matmul(h_b_tensor, sigma)) \
                - torch.matmul(Gamma_b_tensor, torch.matmul(h_b_tensor, Gamma_b_tensor)) \
                + 1j* torch.matmul(Gamma_b_tensor, torch.matmul(O_b_tensor, sigma)) \
                - 1j*torch.matmul(sigma, torch.matmul(O_b_tensor, Gamma_b_tensor))
    
    return final_mat.numpy()

def imaginary_time_equation_of_motion_for_fermionic_covariance(
        variational_parameters_instance:cdf.variational_parameters,
        functional_derivatives_instance:cdf.functional_derivatives)->torch.Tensor:
    
    Gamma_m_tensor = variational_parameters_instance.Gamma_m
    
    h_m_tensor = functional_derivatives_instance.h_m_mat
    O_m_tensor = functional_derivatives_instance.O_m_mat
    # Print imaginary part if larger than 1e-10
    max_imag_h_m = torch.max(torch.abs(h_m_tensor.imag))
    sum_imag_h_m = torch.sum(torch.abs(h_m_tensor.imag))
    if max_imag_h_m > 1e-10:
        print("Maximum absolute value of imaginary part of h_m_tensor = {:.2e}".format(max_imag_h_m))
        print("Sum of absolute values of imaginary part of h_m_tensor = {:.2e}".format(sum_imag_h_m))

    max_real_O_m = torch.max(torch.abs(O_m_tensor.real))
    sum_real_O_m = torch.sum(torch.abs(O_m_tensor.real))
    if max_real_O_m > 1e-10:
        print("Maximum absolute value of real part of O_m_tensor = {:.2e}".format(max_real_O_m))
        print("Sum of absolute values of real part of O_m_tensor = {:.2e}".format(sum_real_O_m))


    final_mat = - h_m_tensor  - torch.matmul(Gamma_m_tensor, torch.matmul(h_m_tensor, Gamma_m_tensor )) + 1j* ( torch.matmul(Gamma_m_tensor, O_m_tensor) - torch.matmul(O_m_tensor, Gamma_m_tensor) )
    
    return final_mat.numpy()

def compute_rhs_non_gaussian_EOM(
        input_variables_instance:cdf.input_variables,
         variational_parameters_instance:cdf.variational_parameters,
         computed_variables_instance:cdf.computed_variables, 
         fermionic_correlations_instance:cdf.fermionic_correlation_matrices)-> torch.Tensor:
    """
    RHS of equation 1.47 from succint notes of real_space_model (feb 27, 2025). 
    Includes the left multiplication with sigma matrix.
    
    """
    
    N_b = input_variables_instance.N_b
    N_f = input_variables_instance.N_f

    Gamma_b_tensor = variational_parameters_instance.Gamma_b

    density_density_connected_correlation_tensor =  fermionic_correlations_instance.density_density_connected_correlator
    density_density_anticommutator_tensor =  fermionic_correlations_instance.density_density_anticommutator_connected_correlator
    c_dagger_c_expectation_value_tensor = fermionic_correlations_instance.c_dagger_c
    j_i_j_tensor = computed_variables_instance.J_i_j_mat
    w_ln_tensor = computed_variables_instance.w_ln_mat
    delta_gamma_tilde_tensor = computed_variables_instance.delta_gamma_tilde_mat


    #sigma matrix
    sigma = torch.zeros((2 * N_b, 2 * N_b), dtype=torch.complex128)
    sigma[:N_b, N_b:] = torch.eye(N_b, dtype=torch.complex128)
    sigma[N_b:, :N_b] = -torch.eye(N_b, dtype=torch.complex128)


    final_mat = torch.zeros(2*N_b, N_f, dtype=torch.complex128)

    final_mat.add_(-0.5*1j*torch.einsum("lk, kim, im, nim  -> ln", Gamma_b_tensor, w_ln_tensor, j_i_j_tensor, density_density_anticommutator_tensor))
    final_mat.add_(torch.einsum("lk, km, mn -> ln", Gamma_b_tensor, delta_gamma_tilde_tensor, density_density_connected_correlation_tensor) )
    real_part = torch.einsum("lk, knm, nm -> ln", sigma, w_ln_tensor, torch.real(j_i_j_tensor * c_dagger_c_expectation_value_tensor).to(dtype=torch.complex128))



    # imag_part = torch.einsum("lk, knm, nm -> ln", sigma, w_ln_tensor, torch.imag(j_i_j_tensor * c_dagger_c_expectation_value_tensor).to(dtype=torch.complex128))
    # print("Maximum absolute value of imag_part of rhs_non_gaussian_EOM  = ", torch.max(torch.abs(imag_part)))
    final_mat.add_(real_part.to(dtype=torch.complex128))

    final_mat_to_return = final_mat[0:N_b, :]

    # if torch.all(final_mat_to_return == 0):
    #     print("All entries in the rhs of eqn 137 matrix are ZERO.")
    # else:
    #     print("There are non-zero entries in the rhs of eqn 137 matrix. Largest = {:.2e}".format(torch.max(torch.abs(final_mat_to_return))) ) 

    # constraint = final_mat[N_b:, :]

    # if torch.all(constraint == 0):
    #     print("The constraint in equation 138 is satified.")
    # else:
    #     print("The constraint in equation 138 is NOT satisfied. The largest absolute value of the matrix is = {:.2e} ".format(torch.max(torch.abs(constraint)) )) 

    return final_mat_to_return 





def imaginary_time_EOM_for_lambda_bar_pytorch_lstsq(input_variable: cdf.input_variables,
                                      variational_parameters_instance:cdf.variational_parameters,  
                                        computed_variables_instance:cdf.computed_variables, 
                                        fermionic_correlations_instance:cdf.fermionic_correlation_matrices)->np.ndarray:
    """
        solves the non-gaussian EOM using scipy.linalg.lstsq
    """

    density_density_correlation_connected_correlator_tensor = fermionic_correlations_instance.density_density_connected_correlator

    rhs_ngs_eom = compute_rhs_non_gaussian_EOM(input_variable, variational_parameters_instance, computed_variables_instance, fermionic_correlations_instance) 
    
    imaginary_part_rhs = torch.max(torch.abs(rhs_ngs_eom.imag))
    
    if imaginary_part_rhs > 1e-10:
        print("Maximum absolute value of imaginary part of rhs_ngs_eom = {:.2e}".format(imaginary_part_rhs))

    rhs_ngs_eom_transpose = rhs_ngs_eom.t().real
 
    density_density_correlation_tensor_transpose = density_density_correlation_connected_correlator_tensor.t().real
 
    d_tau_lambda_transpose = torch.linalg.lstsq(density_density_correlation_tensor_transpose.double(), rhs_ngs_eom_transpose.double(), driver = "gelsss").solution

    print("----- properties of density-density correlation matrix:-----")
    
    print("remove this part if the code works well to opimized the code. ")

    U, s, Vh = torch.linalg.svd(density_density_correlation_tensor_transpose.double())  # Singular value decomposition
    rank = (s > 1e-12).sum().item()  # Consider a tolerance for rank

    # Print rank, condition number, and solution check
    print("rank with cutoff e-12 = ", rank)
    print(f"condition number = {s[0] / s[-1]:.2e}")

    print(f"rank with linalg.matrix_rank = {torch.linalg.matrix_rank(density_density_correlation_tensor_transpose):.2e}")
    print(f"condition number with linalg.cond = {torch.linalg.cond(density_density_correlation_tensor_transpose):.2e}")
    print("------------------------")

    print(f"check sol = {torch.max(torch.abs(torch.matmul(density_density_correlation_tensor_transpose, d_tau_lambda_transpose) - rhs_ngs_eom_transpose)):.2e}")

    return d_tau_lambda_transpose.t().numpy().astype(np.complex128)


def imaginary_time_EOM_for_lambda_bar_qr(input_variable: cdf.input_variables,
                                      variational_parameters_instance:cdf.variational_parameters,  
                                        computed_variables_instance:cdf.computed_variables, 
                                        fermionic_correlations_instance:cdf.fermionic_correlation_matrices)->np.ndarray:


    N_b = input_variable.N_b
    N_f = input_variable.N_f

    corr = fermionic_correlations_instance.density_density_connected_correlator 

    is_symmetric = torch.allclose(corr, corr.t(), atol=1e-10)

    if is_symmetric:
        print("corr is symmetric:")
    else:
        print("corr is NOT symmetric:")
        print("Maximum absolute value of corr - corr^T = {:.2e}".format(torch.max(torch.abs(corr - corr.t()))))
        print("sum of deviation from symmetric matrix= {:.2e}".format(torch.sum(torch.abs(corr - corr.t()))))
    
    corr_transpose = corr.t().real


    rhs_ngs_eom = compute_rhs_non_gaussian_EOM(input_variable, variational_parameters_instance,
                                                computed_variables_instance, fermionic_correlations_instance) 
    
    rhs_transpose = rhs_ngs_eom.t().real
    
    imaginary_part_rhs = torch.max(torch.abs(rhs_ngs_eom.imag))
    if imaginary_part_rhs > 1e-10:
        print("Maximum absolute value of imaginary part of rhs_ngs_eom = {:.2e}".format(imaginary_part_rhs))

    rank = torch.linalg.matrix_rank(corr_transpose)
    print("Effective rank of the matrix: ", rank)
    
    if rank == corr_transpose.shape[0]:
        print("rank = ", rank)
        print(f"condition number = {torch.linalg.cond(corr_transpose):.2e}")
        d_tau_lambda_transpose = torch.linalg.lstsq(corr_transpose.double(), rhs_transpose.double()).solution
        print(" ---> Correctness of Solution = ", torch.max(torch.abs(torch.matmul(corr_transpose, d_tau_lambda_transpose) - rhs_transpose)))

    else:
        print(" ---> rank = ", rank)
        print(f" ---> condition number = {torch.linalg.cond(corr_transpose):.2e}")

        Q, R, P = qr(corr_transpose.numpy(), mode='economic', pivoting=True)

        Q = torch.from_numpy(Q).to(dtype=torch.double)
        R = torch.from_numpy(R).to(dtype=torch.double)
        P = torch.from_numpy(P)

        P_matrix = torch.eye(N_f, dtype=torch.double)[:, P.long()]
        print("shape of P = ", P.shape)

        # Check if Q^T Q is identity
        identity_check_Q = torch.allclose(torch.matmul(Q.t(), Q), torch.eye(Q.shape[1], dtype=torch.double), atol=1e-10)
  
        # Check if P_matrix P_matrix^T is identity
        identity_check_P = torch.allclose(torch.matmul(P_matrix, P_matrix.t()), torch.eye(P_matrix.shape[0], dtype=torch.double), atol=1e-10)

        if identity_check_Q and identity_check_P:
            print("Identity check of Q and P_matrix is successful.")
        else:
            print("Identity check of Q and P_matrix is NOT successful.")

        q_transpose_times_rhs_transpose = torch.matmul(Q.t(), rhs_transpose).to(dtype=torch.double)

        # reduces the system of equations to a smaller system of equations
        R_tilde = R[:rank, :rank]    
        # print("shape of R_tilde = ", R_tilde.shape)
        q_transpose_times_rhs_transpose_tilde = q_transpose_times_rhs_transpose[:rank, :]
        # print("shape of q_t_times_rhs_tilde = ", q_trans_times_rhs_tilde.shape)

        P_transpose_times_d_tau_lambda_transpose_tilde = torch.linalg.solve(R_tilde.double(), q_transpose_times_rhs_transpose_tilde.double()) 
        # print("shape of P_transpose_times_d_tau_lambda_transpose_tilde = ", P_transpose_times_d_tau_lambda_transpose_tilde.shape)


        P_transpose_times_d_tau_lambda_transpose = torch.zeros(N_f, N_b, dtype=torch.double)
        P_transpose_times_d_tau_lambda_transpose[:rank, :] = P_transpose_times_d_tau_lambda_transpose_tilde
        # print("shape of P_transpose_times_d_tau_lambda =" , P_transpose_times_d_tau_lambda_transpose.shape)

        d_tau_lambda_transpose = torch.matmul(P_matrix, P_transpose_times_d_tau_lambda_transpose)

        print("largest abs entry = {:.2e}, sum abs entries = {:.2e}".format(torch.max(torch.abs(d_tau_lambda_transpose)), torch.sum(torch.abs(d_tau_lambda_transpose))))

        print(" ---> Correctness of Solution, max, sum = ", 
              torch.max(torch.abs(torch.matmul(corr_transpose, d_tau_lambda_transpose) - rhs_transpose)), 
               torch.sum(torch.abs(torch.matmul(corr_transpose, d_tau_lambda_transpose) - rhs_transpose)) )

    return d_tau_lambda_transpose.t().numpy().astype(np.complex128)






# def imaginary_time_EOM_for_lambda_bar_scipy_lstsq(input_variable: cdf.input_variables,
#                                       variational_parameters_instance:cdf.variational_parameters,  
#                                         computed_variables_instance:cdf.computed_variables, 
#                                         fermionic_correlations_instance:cdf.fermionic_correlation_matrices)->np.ndarray:
#     """
#         solves the non-gaussian EOM using scipy.linalg.lstsq
#     """

#     N_b = input_variable.N_b
#     N_f = input_variable.N_f

#     density_density_correlation_connected_correlator_tensor = fermionic_correlations_instance.density_density_connected_correlator

#     # Check if the matrix is symmetric
#     is_symmetric = torch.allclose(density_density_correlation_connected_correlator_tensor, 
#                                   density_density_correlation_connected_correlator_tensor.t(), 
#                                   atol=1e-10)
#     if is_symmetric:
#         print("The connected correlation matrix is symmetric.")
#     else:
#         print("The connected correltation matrix is NOT symmetric.")

#     rhs_ngs_eom = compute_rhs_non_gaussian_EOM(input_variable, variational_parameters_instance, computed_variables_instance, fermionic_correlations_instance) 
    
#     imaginary_part_rhs = torch.max(torch.abs(rhs_ngs_eom.imag))
    
#     if imaginary_part_rhs > 1e-10:
#         print("Maximum absolute value of imaginary part of rhs_ngs_eom = {:.2e}".format(imaginary_part_rhs))

#     rhs_ngs_eom_transpose = rhs_ngs_eom.t().real
#     rhs_ngs_eom_transpose_numpy = rhs_ngs_eom_transpose.numpy().astype(np.float64)

#     density_density_correlation_tensor_transpose = density_density_correlation_connected_correlator_tensor.t().real
#     density_density_correlation_transpose_numpy = density_density_correlation_tensor_transpose.numpy().astype(np.float64)

#     # d_tau_lambda_transpose = torch.linalg.lstsq(density_density_correlation_tensor_transpose.double(), rhs_ngs_eom_transpose.double()).solution
#     d_tau_lambda_transpose, residuals, rank, s = lstsq(density_density_correlation_transpose_numpy, rhs_ngs_eom_transpose_numpy) 


#     print("rank = ", rank)
#     print(f"condition number = {s[0]/s[-1]:.2e}")
#     print("check sol = {:.2e}".format(np.max(np.abs(np.matmul(density_density_correlation_transpose_numpy, d_tau_lambda_transpose) - rhs_ngs_eom_transpose_numpy))))



#     return d_tau_lambda_transpose.T.astype(np.complex128)


# def equation_of_motion_for_lambda_bar_real(input_variable: cdf.input_variables,
#                                       variational_parameters_instance:cdf.variational_parameters,  
#                                         computed_variables_instance:cdf.computed_variables, 
#                                         fermionic_correlations_instance:cdf.fermionic_correlation_matrices)->np.ndarray:
#     """
#     (feb 25, 2025) This code has \partial_{tau} lambda_{l, n\sigma} for l \in (1, 2_n_b) since Gamma_b has shape (2N_b, 2N_b)
#     (feb 27, 2025) Now this code can be used to calculate \partial_{tau} lambda_bar{l, n\sigma} for l \in (1, 2N_b). 
#     """
#     N_b = input_variable.N_b
#     N_f = input_variable.N_f
  
#     density_density_correlation_connected_correlator_tensor =  fermionic_correlations_instance.density_density_connected_correlator.numpy()

#     rhs_ngs_eom = compute_rhs_non_gaussian_EOM(input_variable, variational_parameters_instance, computed_variables_instance, fermionic_correlations_instance) 
    
#     imaginary_part_rhs = np.max(np.abs(rhs_ngs_eom.imag))
#     print("Maximum absolute value of imaginary part of rhs_ngs_eom = {:.2e}".format(imaginary_part_rhs))

#     rhs_ngs_eom_transpose = np.transpose(rhs_ngs_eom).real

#     density_density_correlation_tensor_transpose = np.transpose(density_density_correlation_connected_correlator_tensor).real

#     rank = np.linalg.matrix_rank(density_density_correlation_tensor_transpose)
#     if rank < density_density_correlation_tensor_transpose.shape[0]:
#         print("WARNING: rank of density_density correlation tensor transpose = ", rank)
#         print("max value of RHS of EOM if NG parameter = {:.2e}".format(np.max(np.abs(rhs_ngs_eom_transpose))))
#         final_mat = np.zeros((N_b, N_f))
        
#     else: 

#         final_mat = np.transpose(np.linalg.solve(density_density_correlation_tensor_transpose, rhs_ngs_eom_transpose)).astype(np.complex128)
 
#     check_sol = np.max(np.abs(np.matmul(final_mat, density_density_correlation_connected_correlator_tensor) - rhs_ngs_eom))

#     print("Maximum absolute value of check_sol = {:.2e} ".format(check_sol))
#     # print("Maximum absolute value of real part of check_sol = {:.2e} ", .format(np.max(np.abs(check_sol.real)) ))
    
#     # print("shape of final_mat \partial_{tau} lambda_{l, n} = ", final_mat.shape
    
#     return final_mat 



# def equation_of_motion_for_lambda_bar_v1(input_variable: cdf.input_variables,
#                                       variational_parameters_instance:cdf.variational_parameters,  
#                                         computed_variables_instance:cdf.computed_variables, 
#                                         fermionic_correlations_instance:cdf.fermionic_correlation_matrices)->np.ndarray:
#     """
#     (feb 25, 2025) This code has \partial_{tau} lambda_{l, n\sigma} for l \in (1, 2_n_b) since Gamma_b has shape (2N_b, 2N_b)
#     (feb 27, 2025) Now this code can be used to calculate \partial_{tau} lambda_bar{l, n\sigma} for l \in (1, 2N_b). 
#     """

#     Gamma_m = variational_parameters_instance.Gamma_m
  
#     density_density_correlation_connected_correlator_tensor =  fermionic_correlations_instance.density_density_connected_correlator

#     rhs_ngs_eom = compute_rhs_non_gaussian_EOM(input_variable, variational_parameters_instance, computed_variables_instance, fermionic_correlations_instance)

#     rhs_ngs_eom_transpose = np.transpose(rhs_ngs_eom)

#     density_density_correlation_tensor_transpose = np.transpose(density_density_correlation_connected_correlator_tensor)

#     final_mat = np.linalg.solve(density_density_correlation_tensor_transpose, rhs_ngs_eom_transpose) 
    
#     final_mat = np.transpose(final_mat)
    
#     print("shape of final_mat \partial_{tau} lambda_{l, n} = ", final_mat.shape)
    
#     return final_mat 




