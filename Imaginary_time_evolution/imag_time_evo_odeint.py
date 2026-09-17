import gc # garbage collector
import numpy as np
from scipy.integrate import odeint
from Imaginary_time_evolution import imag_time_evolution_functions as itef
from common_codes import class_defn_file as cdf

import os

def imag_time_evo_model_solve_ivp(t:np.ndarray, y:np.ndarray, input_variables_instance:cdf.input_variables):
    print("started time =", t)

    N_b = input_variables_instance.N_b
    N_f = input_variables_instance.N_f

    # y is a varible which contains all the variational parameters (delta_r, gamma_b, gamma_m, lambda_q). 
    if(y.dtype != np.complex128):
        delta_r = y[0:2*N_b].astype(np.complex128)
        Gamma_b = y[2*N_b: 2*N_b + (2*N_b*2*N_b)].astype(np.complex128)
        Gamma_m = y[ 2*N_b + (2*N_b*2*N_b) :  2*N_b + (2*N_b*2*N_b) + (2*N_f*2*N_f)].astype(np.complex128)
        lmbda_bar = y[ (2*N_b + (2*N_b*2*N_b) + (2*N_f*2*N_f)) : ].astype(np.complex128)

    if(y.dtype == np.complex128): 
        delta_r = y[ 0:2*N_b]
        Gamma_b = y[ 2*N_b: 2*N_b + (2*N_b)*(2*N_b)]
        Gamma_m = y[ 2*N_b + (2*N_b)*(2*N_b) :  2*N_b + (2*N_b*2*N_b) + (2*N_f*2*N_f)]
        lmbda_bar = y[ 2*N_b + (2*N_b*2*N_b) + (2*N_f*2*N_f) : ]

    delta_r = np.reshape(delta_r, 2*N_b)
    Gamma_b = np.reshape(Gamma_b, (2*N_b, 2*N_b))
    Gamma_m = np.reshape(Gamma_m, (2*N_f, 2*N_f))
    lmbda_bar = np.reshape(lmbda_bar, (2*N_b, N_f))



    if any(np.isclose(t, target_time, atol=0.02) for target_time in range(20, 300, 20)):
        output_dir = os.path.join(f"/mnt/ceph/vksharma/ite_data_jnn_{input_variables_instance.j_nn:.0f}_j_nnn_{input_variables_instance.j_nnn:.2f}_omega_{input_variables_instance.omega_number:.0f}_g_{input_variables_instance.gamma_number:.2f}_august_26_2025", f"mu_{input_variables_instance.chemical_potential_val:.2f}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Saving data at time {t} in directory {output_dir}")
        np.save(f"{output_dir}/lambda_bar_{int(t)}.npy", lmbda_bar)
        np.save(f"{output_dir}/Gamma_b_{int(t)}.npy", Gamma_b)
        np.save(f"{output_dir}/Gamma_m_{int(t)}.npy", Gamma_m)
        np.save(f"{output_dir}/Delta_R_{int(t)}.npy", delta_r)

    # input_variables.updating_lambda_from_lambda_bar(lmbda_bar = lmbda_bar) 
    if 'variational_parameters_instance' not in globals():
        global variational_parameters_instance
        variational_parameters_instance = cdf.variational_parameters(N_f, N_b)
    
    if 'fermionic_correlations_instance' not in globals():
        global fermionic_correlations_instance
        fermionic_correlations_instance = cdf.fermionic_correlation_matrices(variational_parameters_instance=variational_parameters_instance)
    
    if 'computed_variables_instance' not in globals():
        global computed_variables_instance
        computed_variables_instance = cdf.computed_variables(N_b, N_f)

    # do the sanity check and then update the variational parameters.
    variational_parameters_instance.initialize_and_update_variational_parameters(delta_r, Gamma_b, Gamma_m, lmbda_bar) 

    fermionic_correlations_instance.update_fermionic_correlation(variational_parameters_instance)

    computed_variables_instance.initialize_and_update_computed_variables(input_variables_instance, 
                                                                       variational_parameters_instance)

    cdf.compute_energy_expectation(input_variables_instance, variational_parameters_instance, computed_variables_instance,
                                    fermionic_correlations_instance)  
    
    

    d_lambda_bar_p_dtau = itef.imaginary_time_EOM_for_lambda_bar_qr(input_variables_instance, 
                                                                    variational_parameters_instance, 
                                                                    computed_variables_instance, 
                                                                    fermionic_correlations_instance)
    

    d_lambda_bar_p_dtau = np.reshape(d_lambda_bar_p_dtau, (N_b, N_f)) # is this required?

    d_lambda_bar_dtau = np.append(np.zeros((N_b, N_f), dtype=np.complex128), d_lambda_bar_p_dtau, axis=0)

    functional_derivatives_instance = cdf.functional_derivatives(input_variables_instance)
    functional_derivatives_instance.initialize_and_update_functional_derivatives(d_lambda_bar_dtau, 
                                                                                input_variables_instance, 
                                                                                variational_parameters_instance,
                                                                                computed_variables_instance, 
                                                                                fermionic_correlations_instance)
    

    d_delta_R_dtau = itef.imaginary_time_equation_of_motion_for_bosonic_average(input_variables_instance, 
                                                                              variational_parameters_instance, 
                                                                              functional_derivatives_instance)
    
    d_Gamma_b_dtau = itef.imaginary_time_equation_of_motion_from_bosonic_covariance(input_variables_instance, 
                                                                                  variational_parameters_instance, 
                                                                                  functional_derivatives_instance)
    
    d_Gamma_m_dtau = itef.imaginary_time_equation_of_motion_for_fermionic_covariance(variational_parameters_instance, 
                                                                                   functional_derivatives_instance)
    
    # Check if the imaginary part is almost zero
    if np.any(np.abs(np.imag(d_delta_R_dtau)) > 1e-10):
        print(f"Warning: Imaginary part of d_delta_R_dt is larger than 1e-10. Max value: {np.max(np.abs(np.imag(d_delta_R_dtau))):.2e}")
    if np.any(np.abs(np.imag(d_Gamma_b_dtau)) > 1e-10):
        print(f"Warning: Imaginary part of d_Gamma_b_dt is larger than 1e-10. Max value: {np.max(np.abs(np.imag(d_Gamma_b_dtau))):.2e}")
    if np.any(np.abs(np.imag(d_Gamma_m_dtau)) > 1e-10):
        print(f"Warning: Imaginary part of d_Gamma_m_dt is larger than 1e-10. Max value: {np.max(np.abs(np.imag(d_Gamma_m_dtau))):.2e}")
    if np.any(np.abs(np.imag(d_lambda_bar_p_dtau)) > 1e-10):
        print(f"Warning: Imaginary part of time_derivative_lambda_bar is larger than 1e-10. Max value: {np.max(np.abs(np.imag(d_lambda_bar_p_dtau))):.2e}")

    # if np.any(np.abs(np.imag(d_delta_R_dtau)) > 5*1e-1):
    #     raise ValueError("Imaginary part of d_delta_R_dt is larger than 5*1e-1. EXITING, max_imaginary_part = " + str(np.max(np.abs(np.imag(d_delta_R_dt)))))
    # if np.any(np.abs(np.imag(d_Gamma_b_dtau)) > 5*1e-1):
    #     raise ValueError("Imaginary part of d_Gamma_b_dt is larger than 5*1e-1. EXITING, max_imaginary_part = " + str(np.max(np.abs(np.imag(d_Gamma_b_dt)))))
    # if np.any(np.abs(np.imag(d_Gamma_m_dtau)) > 5*1e-1):
    #     raise ValueError("Imaginary part of d_Gamma_m_dt is larger than 5*1e-1. Exiting, max_imaginary_part = " + str(np.max(np.abs(np.imag(d_Gamma_m_dt)))))
    # if np.any(np.abs(np.imag(d_lambda_bar_dtau)) > 5*1e-1):
    #     print(f"Imaginary part of time_derivative_lambda_bar is larger than 5*1e-1. Max value: {np.max(np.abs(np.imag(time_derivative_lambda_bar))):.2e}")
    #     raise ValueError("Imaginary part of time_derivative_lambda_bar is larger than 1e-3. Exiting")


    # Convert to real values
    # d_delta_R_dtau = np.real(d_delta_R_dtau).astype(np.float64)
    # d_Gamma_b_dtau = np.real(d_Gamma_b_dtau).astype(np.float64)
    # d_Gamma_m_dtau = np.real(d_Gamma_m_dtau).astype(np.float64)
    # d_lambda_bar_dtau = np.real(d_lambda_bar_dtau).astype(np.float64)

    d_delta_R_dtau = np.real(d_delta_R_dtau) 
    d_Gamma_b_dtau = np.real(d_Gamma_b_dtau) 
    d_Gamma_m_dtau = np.real(d_Gamma_m_dtau) 
    d_lambda_bar_dtau = np.real(d_lambda_bar_dtau) 

    #reshaping the array to a single vector
    d_delta_R_dtau = np.reshape(d_delta_R_dtau, 2*N_b)
    d_Gamma_b_dtau = np.reshape(d_Gamma_b_dtau, 2*N_b*2*N_b)
    d_Gamma_m_dtau = np.reshape(d_Gamma_m_dtau, 2*N_f*2*N_f)
    d_lambda_bar_dtau = np.reshape(d_lambda_bar_dtau, 2*N_b*N_f)


    dydt = np.concatenate((d_delta_R_dtau, d_Gamma_b_dtau, d_Gamma_m_dtau, d_lambda_bar_dtau))

    print("completed time =" ,t , ".\n")

    return dydt