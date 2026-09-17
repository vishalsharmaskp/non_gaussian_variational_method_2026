import sys
from common_codes import generic_codes as gcs
from common_codes import class_defn_file as cdf 

import scipy.integrate as integrate
import numpy as np
from Imaginary_time_evolution.imag_time_evo_odeint import imag_time_evo_model_solve_ivp
import time as te
import gc 
import thewalrus as tw
import os
import matplotlib.pyplot as plt


from datetime import datetime
print("current date and time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    
    # if len(sys.argv) != 8:
    #     print("inside main_nnn.py")
    #     print("Usage: python main_nnn.py <j_nn> <j_nnn> <omega_0> <gamma_0> <chemical_potential_val> <r_tol> <a_tol>")
    #     sys.exit(1)

    # j_nn = float(sys.argv[1])  # J_nn value
    # j_nnn = float(sys.argv[2])  # J_nnn value
    # omega_0 = float(sys.argv[3])
    # gamma_0 = float(sys.argv[4])
    # chemical_potential_val = float(sys.argv[5])
    # r_tol = float(sys.argv[6])
    # a_tol = float(sys.argv[7])

    j_nn = -1.0
    j_nnn = -0.0
    omega_0 = 10.0
    gamma_0 = 4.0
    chemical_potential_val = -5.0
    r_tol = 5e-4
    a_tol = 1e-7


    gc.collect()

    start_time = te.time()

    t_min = 0
    t_max = 200
    
    print("t_max = ", t_max)

    # output_dir = os.path.join(f"/mnt/ceph/vksharma/ite_data_jnn_{j_nn:.0f}_j_nnn_{j_nnn:.2f}_omega_{omega_0:.0f}_g_{gamma_0:.2f}_october_2025", f"mu_{chemical_potential_val:.2f}")
    output_dir = f"final_state/jnn_{j_nn:.0f}_j_nnn_{j_nnn:.2f}_omega_{omega_0:.0f}_g_{gamma_0:.2f}_sept_2026/mu_{chemical_potential_val:.2f}"
  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    t_to_save = t_max

    t_span = (t_min, t_max)

    number_of_points = 10 
    position_value_max = [10, 10]
    position_value_min = [0, 0]
    position_space_grid = gcs.coordinate_array_creator_function(position_value_min, position_value_max, number_of_points, spin_index = True)

    momentum_value_max = [np.pi/(position_value_max[0] - position_value_min[0])*number_of_points, np.pi/(position_value_max[1] - position_value_min[1])*number_of_points]
    momentum_value_min = [-np.pi/(position_value_max[0] - position_value_min[0])*number_of_points, -np.pi/(position_value_max[1] - position_value_min[1])*number_of_points]
    momentum_space_grid = gcs.coordinate_array_creator_function(momentum_value_min, momentum_value_max, number_of_points, spin_index=False)

    boson_space_grid = gcs.coordinate_array_creator_function(position_value_min, position_value_max, number_of_points, spin_index = False)

    N_b = boson_space_grid.shape[0]
    N_f = position_space_grid.shape[0]

    volume = np.prod(np.array(position_value_max) - np.array(position_value_min))


    # J_0_matrix = gcs.creating_J_0_matrix_PBC(position_space_grid, J_0, position_value_max[0], spin_index = True)
    hopping_matrix = gcs.creating_J_0_matrix_with_NNN_PBC(position_space_grid, j_nn, j_nnn, position_value_max[0], spin_index = True)
    plt.pcolormesh(np.real(hopping_matrix))
    plt.colorbar()
    plt.title("Hopping matrix")
    plt.savefig(os.path.join(output_dir, f"hopping_matrix_{chemical_potential_val}.png"))
    plt.close()

    # omega_0 = 10*np.abs(J_0)
    omega_matrix = omega_0*np.identity(N_b) # shape: N_b*N_b, we calculate omega in class_defn file.
    plt.pcolormesh(np.real(omega_matrix))
    plt.colorbar()
    plt.title("omega matrix")
    plt.savefig(os.path.join(output_dir, f"omega_matrix_{chemical_potential_val}.png"))
    plt.close()

    # gamma_0 = 0.4*omega_0
    gamma_matrix = gamma_0*np.append(np.identity(N_b), np.identity(N_b), axis = 1)
    plt.pcolormesh(np.real(gamma_matrix))
    plt.colorbar()
    plt.title("gamma matrix")
    plt.savefig(os.path.join(output_dir, f"gamma_matrix_{chemical_potential_val}.png"))
    plt.close()
    

    input_variables_instance = cdf.input_variables(N_b, N_f, j_nn=j_nn, j_nnn=j_nnn,
                                                    J_0_matrix=hopping_matrix, omega_number=omega_0, 
                                                    omega_matrix=omega_matrix, gamma_number=gamma_0, 
                                                    gamma_matrix=gamma_matrix, chemical_potential_val=chemical_potential_val)


    print("----- Informations --------------------------------------------")
    print("chemical_potential_val = ", chemical_potential_val)
    print("J nearest neighbour coupling = ", j_nn)
    print("J next nearest neighbour coupling = ", j_nnn)
    print("omega_0 = ", omega_0)
    print("gamma_0 = ", gamma_0)
    print(f"t_range = [{t_min}, {t_max}]")
    print("r_tolerance = ", r_tol)
    print("a_tolerance = ", a_tol)
    print("output_dir = ", output_dir)
    print("--------------------------------------------------------------")

    gs_dir = output_dir

    seed = np.random.randint(0, 1000)
    Delta_R = gcs.initialise_delta_R_matrix(N_b, seed) 
    Gamma_m = gcs.initialize_gamma_m_matrix(N_f)
    lambda_bar = gcs.intialize_lambda_bar(N_b, N_f)

    while True: 
        Gamma_b = gcs.initialize_gamma_b_matrix(N_b)
        if np.all(np.abs(Gamma_b) < 20):
            break
    print("Initial variational parameters created successfully. I am starting with small Gamma_b matrix < 20. ") 



    plt.plot(np.real(Delta_R))
    plt.title("Initial Delta_R")
    plt.savefig(os.path.join(output_dir, f"initial_delta_R_{t_min}.png"))
    plt.close()

    plt.pcolormesh(np.real(Gamma_b))
    plt.colorbar()
    plt.title("Initial Gamma_b")
    plt.savefig(os.path.join(output_dir, f"initial_gamma_b_{t_min}.png"))
    plt.close()

    plt.pcolormesh(np.real(Gamma_m))
    plt.colorbar()
    plt.title("Initial Gamma_m")
    plt.savefig(os.path.join(output_dir, f"initial_gamma_m_{t_min}.png"))
    plt.close()

    plt.pcolormesh(np.real(lambda_bar))
    plt.colorbar()
    plt.title("Initial lambda_real_space")
    plt.savefig(os.path.join(output_dir, f"initial_lambda_bar_{t_min}.png"))
    plt.close()
 


    np.save(os.path.join(output_dir, f"delta_r_mu_{chemical_potential_val}_t_{t_min}.npy"), Delta_R)
    np.save(os.path.join(output_dir, f"Gamma_b_mu_{chemical_potential_val}_t_{t_min}.npy"), Gamma_b)
    np.save(os.path.join(output_dir, f"Gamma_m_mu_{chemical_potential_val}_t_{t_min}.npy"), Gamma_m)
    np.save(os.path.join(output_dir, f"lambda_bar_mu_{chemical_potential_val}_t_{t_min}.npy"), lambda_bar)

    y0 = np.concatenate((Delta_R.flatten(), Gamma_b.flatten(), Gamma_m.flatten(), lambda_bar.flatten())).astype(np.complex128)
    # y0 = np.concatenate((Delta_R.flatten(), Gamma_b.flatten(), Gamma_m.flatten(), lambda_bar.flatten())).astype(np.float64)

    model_solve_ivp = imag_time_evo_model_solve_ivp
    # r_tol = 1e-3
    # a_tol = 1e-6

    print(f"Starting the time evolution with r_tol = {r_tol:.1e} and a_tol = {a_tol:.1e}")
    # method = 'DOP853'  # 'RK45', 'RK23', 'DOP853', 'LSODA'
    
    method = 'RK45'
    print(f"Using method = {method} for ODE integration.")

    # sol_solve_ivp = integrate.solve_ivp(model_solve_ivp, t_span, y0, args = (input_variables_instance, ), method = 'RK45', rtol = r_tol, atol = a_tol)
    sol_solve_ivp = integrate.solve_ivp(model_solve_ivp, t_span, y0, args = (input_variables_instance, ), method = method, rtol = r_tol, atol = a_tol)

    # sol_solve_ivp = integrate.solve_ivp(model_solve_ivp, t_span, y0, args = (input_variables_instance, ), method = 'DOP853', rtol = r_tol, atol = a_tol)    

    print("Time taken to complete the simulation = ", te.time() - start_time)
    print("just for git commit")

    sol = sol_solve_ivp.y.T

    print("extracting the final variational parameters and saving them.")

    delta_r_final = np.real(sol[-1, 0:2*N_b])

    Gamma_b_final = np.reshape(np.real(sol[-1, 2*N_b : 2*N_b + (2*N_b)*(2*N_b)]), (2*N_b, 2*N_b))
    
    Gamma_m_final = np.reshape(np.real(sol[-1, 2*N_b + (2*N_b)*(2*N_b) : 2*N_b + (2*N_b)*(2*N_b) + (2*N_f)*(2*N_f)]), (2*N_f, 2*N_f))

    lmbda_bar_final = np.reshape(np.real(sol[-1, 2*N_b + (2*N_b)*(2*N_b) + (2*N_f)*(2*N_f) : ]), (2*N_b, N_f))

    np.save(os.path.join(output_dir, f"delta_r_mu_{chemical_potential_val}_t_{t_max}.npy"), delta_r_final)
    np.save(os.path.join(output_dir, f"Gamma_b_mu_{chemical_potential_val}_t_{t_max}.npy"), Gamma_b_final)
    np.save(os.path.join(output_dir, f"Gamma_m_mu_{chemical_potential_val}_t_{t_max}.npy"), Gamma_m_final)
    np.save(os.path.join(output_dir, f"lambda_bar_mu_{chemical_potential_val}_t_{t_max}.npy"), lmbda_bar_final)

    print("saving the solutions at all the time steps.")
    output_path_file = os.path.join(output_dir, f"ite_j_nn_{j_nn}_j_nnn_{j_nnn}_omega_0_{omega_0}_gamma_0_{gamma_0}_mu_{chemical_potential_val}.npz")
    
    np.savez(output_path_file,t=sol_solve_ivp.t , sol=sol)
    print("done saving the solutions at all the time steps.")

    # print("total time taken to complete the simulation = ", te.time() - start_time)
    print("----- Informations --------------------------------------------")
    print("chemical_potential_val = ", chemical_potential_val)
    print("J nearest neighbour coupling = ", j_nn)
    print("J next nearest neighbour coupling = ", j_nnn)
    print("omega_0 = ", omega_0)
    print("gamma_0 = ", gamma_0)
    print(f"t_range = [{t_min}, {t_max}]")
    print("tolerance = ", r_tol)
    print("a tolerance = ", a_tol)
    print("output_dir = ", output_dir)
    print("--------------------------------------------------------------")

    print("Finished the simulation at date time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
