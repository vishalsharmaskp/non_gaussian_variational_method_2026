import numpy as np 
import os

print("Current working directory:", os.getcwd())


from common_codes import generic_codes as gcs
from common_codes import class_defn_file as cdf
from scipy.linalg import block_diag

import matplotlib.pyplot as plt
   

current_directory = os.getcwd()

# figure_folder = current_directory + "/figures/"
# save_folder = figure_folder + f"chemical_potential_{chemical_potential}/"

# Ensure the directories exist
# os.makedirs(save_folder, exist_ok=True)
# print("figure_folder", figure_folder)


number_of_points = 10
positon_value_max = [10,10]
positon_value_min = [0,0]
    

momentum_value_max = [ np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points , np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points ]
momentum_value_min = [ -np.pi/(positon_value_max[0]-positon_value_min[0])*number_of_points , -np.pi/(positon_value_max[1]-positon_value_min[1])*number_of_points ]
momentum_space_grid = gcs.coordinate_array_creator_function(momentum_value_min,momentum_value_max,number_of_points,False)

 
position_space_grid_spinless = gcs.coordinate_array_creator_function(positon_value_min,positon_value_max,number_of_points, False)
position_space_grid_spinful = gcs.coordinate_array_creator_function(positon_value_min,positon_value_max,number_of_points, True)
print("\n Position space grid created")

boson_space_grid = gcs.coordinate_array_creator_function(positon_value_min,positon_value_max,number_of_points, False)
print(" Momentum space grid created")


J_0 = -1
J_0_matrix = gcs.creating_J_0_matrix_PBC(position_space_grid_spinful, J_0, positon_value_max[0], spin_index = True)


N_b = boson_space_grid.shape[0]
N_f = position_space_grid_spinful.shape[0]

alpha = 1
omega_num=10.0/alpha
print("omega_num:", omega_num)
omega_0 = omega_num*np.abs(J_0)

omega = omega_num*np.abs(J_0)*np.identity(N_b)
omega_bar = omega_num*np.abs(J_0)*np.identity(N_f)
 

g_num = 4.0/np.sqrt(alpha)
print("g_num:", g_num)
gamma = g_num*np.append(np.identity(N_b),np.identity(N_b),axis = 1)

chemical_potential = -2*(g_num **2)/(omega_num)


print( "************* information ******************")
print("omega_phonon= ", omega_0)
print("gamma_0= ", g_num)
print("cheamical potential =", chemical_potential)


# save_folder = current_directory + f"GS_data_folder/GS_data_psedugap_omega_{omega_num:.2f}_g_{g_num:.2f}_mu_{chemical_potential:.2f}/"
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)

# print("Save folder created:", save_folder)

k_x = np.linspace(-np.pi, np.pi, 10, endpoint=False)
k_y = np.linspace(-np.pi, np.pi, 10, endpoint=False)

fourier_transform_matrix_spinful = gcs.creating_fourier_matrix(momentum_space_grid, position_space_grid_spinful)
fourier_transform_matrix_spinless = gcs.creating_fourier_matrix(momentum_space_grid, position_space_grid_spinless)


def plot_delta_r_real_space(delta_r:np.ndarray):

    max_delta_x = np.max(delta_r[0:100])
    min_delta_x = np.min(delta_r[0:100])

    N_b = int(delta_r.shape[0]/2)

    delta_x = delta_r[:N_b]
    print("delta_x_max/min = ", np.max(delta_x), np.min(delta_x))

    delta_x_odd = delta_x[0]
    delta_x_even = delta_x[1]
    print("delta_x_odd, delta_x_even:", delta_x_odd, delta_x_even)

    delta_y = delta_r[N_b:]
    print("delta_p_max/min = ", np.max(delta_y), np.min(delta_y))

    mean_delta_x = np.mean(delta_x)
    print("mean_delta_x", mean_delta_x)

    abs_deviation_from_mean = np.abs(delta_x - mean_delta_x)
    print("Average, STD of deviation from mean:",np.mean(abs_deviation_from_mean), np.std(abs_deviation_from_mean))


    plt.plot(delta_r)
    plt.title(r"$\Delta_{x}$ and $\Delta_{p}, max: " + str(f"{max_delta_x:.4f}") + ", min: " + str(f"{min_delta_x:.4f}") + " $")
    # plt.savefig(save_folder + "/delta_r_GS.pdf")
    # plt.savefig("figures_november_2/gs_plots/Delta_x_GS_cdw.png", dpi=300)
    # plt.savefig(os.path.join(save_folder, f"delta_R_mu_{chemical_potential}.png"))
    plt.show()

    linear_dim = int(N_b**0.5) 
    plt.pcolormesh(delta_x.reshape(linear_dim,linear_dim))
    plt.colorbar()
    plt.title("delta_x")
    # plt.savefig(os.path.join(save_folder, f"delta_X_mu_{chemical_potential}.png"))
    # plt.savefig(os.path.join(save_folder, f"delta_X_mu_{chemical_potential}.pdf"))
    plt.show()
    # plt.pcolormesh(delta_y.reshape(linear_dim,linear_dim))
    # plt.colorbar()
    # plt.title("delta_p")

    plt.show()

def plot_Gamma_b_momentum_space_from_real_space(Gamma_b_real_space:np.ndarray, fourier_transform_matrix_spinless:np.ndarray):

    print("Gamma_b_real_space.shape",Gamma_b_real_space.shape)
    N_b = int(Gamma_b_real_space.shape[0]/2)
    linear_dim = int(N_b**0.5)

    fig, axis = plt.subplots(1, 2, figsize = (10 , 5))
    gamma_b_xx_covariance = Gamma_b_real_space[:N_b,:N_b]
    im0 = axis[0].pcolormesh(gamma_b_xx_covariance) 
    fig.colorbar(im0, ax= axis[0])
    axis[0].set_title("Gamma_b_xx")
    # plt.savefig(figure_folder + "Gamma_b_xx_mu_-5.1_t_140_qr.png")
    # plt.show()

    gamma_b_pp_covariance = Gamma_b_real_space[N_b:,N_b:]
    im1 = axis[1].pcolormesh(gamma_b_pp_covariance)
    fig.colorbar(im1, ax=axis[1])
    axis[1].set_title("Gamma_b_pp")
    # plt.savefig(figure_folder + "Gamma_b_pp_mu_-5.5_t_90_PBC.png")
    plt.tight_layout()
    plt.show()

    fig, axis = plt.subplots(1, 2, figsize = (10, 5))

    gamma_b_xx_diagonal = np.diagonal(gamma_b_xx_covariance)
    axis[0].plot(gamma_b_xx_diagonal.real)
    axis[0].set_title("gamma_b_xx_diagonal.real")
    # plt.savefig("figures_november_2/gs_plots/gamma_b_xx_diagonal_real_GS.png", dpi=300)
    # plt.show()

    gamma_b_pp_diagonal = np.diagonal(gamma_b_pp_covariance)
    axis[1].plot(gamma_b_pp_diagonal.real)
    axis[1].set_title("gamma_b_pp_diagonal.real")
    # plt.savefig("figures_november_2/gs_plots/gamma_b_pp_diagonal_real_GS.png", dpi=300)
    plt.tight_layout()
    plt.show()


    print(f"average and std of gamma_xx, " f"{np.mean(gamma_b_xx_diagonal):.3e}, {np.std(gamma_b_xx_diagonal):.3e}")
    
    print(f"average and std of gamma_pp, " f"{np.mean(gamma_b_pp_diagonal):.3e}, {np.std(gamma_b_pp_diagonal):.3e}")

    gamma_b_diagonal = np.diagonal(Gamma_b)
    print(f"sum and average of gamma_b,  " f"{np.mean(gamma_b_diagonal):.3e}, {np.std(gamma_b_diagonal):.3e}" )
    


    Gamma_b_xx_momentum_space = (1/N_b)*np.einsum('qm, nm, kn -> qk', fourier_transform_matrix_spinless.conj(), gamma_b_xx_covariance, fourier_transform_matrix_spinless)
    Gamma_b_pp_momentum_space = (1/N_b)*np.einsum('qm, nm, kn -> qk', fourier_transform_matrix_spinless.conj(), gamma_b_pp_covariance, fourier_transform_matrix_spinless)   

    plt.contourf(k_x, k_y, np.diag(Gamma_b_xx_momentum_space.real).reshape(linear_dim,linear_dim))
    plt.colorbar()    
    plt.title("Gamma_b_xx_momentum_space.real")
    # plt.savefig(f"{save_folder}/Gamma_b_xx_momentum_space_real_GS.png", dpi=300)
    # plt.savefig(os.path.join(save_folder, f"Gamma_b_xx_mu_{chemical_potential}.png"))
    plt.show()

    plt.contourf(k_x, k_y, np.diag(Gamma_b_pp_momentum_space.real).reshape(linear_dim,linear_dim))
    plt.colorbar()
    plt.title("Gamma_b_pp_momentum_space.real")
    # plt.savefig(f"{save_folder}/Gamma_b_pp_momentum_space_real_GS.png", dpi=300)
    # plt.savefig(os.path.join(save_folder, f"Gamma_b_pp_mu_{chemical_potential}.pdf"))
    plt.show()

    # diag_Gamma_b_momentum_space = np.diagonal(Gamma_b_momentum_space)
    # plt.contourf(k_x, k_y, diag_Gamma_b_momentum_space.real.reshape(linear_dim,linear_dim))
    # plt.colorbar()
    # plt.title("diag_Gamma_b_momentum_space_q.real")
    # plt.xlabel("k_x")
    # plt.ylabel("k_y")
    # plt.show()




def plot_fermion_correlations(G_m, delta_r:np.ndarray,  lambda_bar:np.ndarray, omega:np.ndarray, gamma:np.ndarray, fourier_transform_matrix_spinless:np.ndarray):
    N_f = int(G_m.shape[0]/2)
    G_m_11 = G_m[0:N_f, 0:N_f]
    G_m_12 = G_m[0:N_f, N_f:]
    G_m_21 = G_m[N_f:, 0:N_f]
    G_m_22 = G_m[N_f:, N_f:]

    eye_Nf = np.identity(N_f)

    c_dagger_c = 0.25 * (2 * eye_Nf - 1j * (G_m_11 + G_m_22) + G_m_12 - G_m_21)
    c_c =                    0.25 * (-1j * (G_m_11 - G_m_22) + (G_m_12 + G_m_21))

    c_dagger_c_dagger = 0.25 * (-1j * (G_m_11 - G_m_22) - (G_m_12 + G_m_21))
    c_c_dagger =       0.25 * (2 * eye_Nf - 1j * (G_m_11 + G_m_22) - G_m_12 + G_m_21)

    # print("filling", np.trace(c_dagger_c)/N_f)
    electron_density_i_sigma = np.diag(c_dagger_c.real)
    print("shape of electron_density_i_sigma", np.shape(electron_density_i_sigma))

    avg_c_dagger_c = np.mean(np.diag(c_dagger_c).real)
    print("filling = ", avg_c_dagger_c)
    abs_deviation_from_avg = np.abs(np.diag(c_dagger_c).real - avg_c_dagger_c)
    print("Average, STD of deviation from filling:", np.mean(abs_deviation_from_avg), np.std(abs_deviation_from_avg))

    c_dagger_c_spin_up = np.real(np.diag(c_dagger_c))[0:100]
    c_dagger_c_spin_down = (np.diag(c_dagger_c))[100:200]

    c_dagger_c_spin_up_grid = c_dagger_c_spin_up.reshape(10, 10)
    c_dagger_c_spin_down_grid = c_dagger_c_spin_down.reshape(10, 10)
    
    fig, axis = plt.subplots(1, 2, figsize=(10, 5)) 

    im0 = axis[0].pcolormesh(c_dagger_c_spin_up_grid.real)
    fig.colorbar(im0, ax=axis[0])
    axis[0].set_title("density spin up")

    im1 = axis[1].pcolormesh(c_dagger_c_spin_down_grid.real)
    fig.colorbar(im1, ax=axis[1])
    axis[1].set_title("density spin down")

    plt.tight_layout()
    plt.show()

    # Write the results to a .txt file
    # with open(os.path.join(save_folder, f"values_{chemical_potential}.txt"), "w") as file:
    #     file.write(f"filling = {avg_c_dagger_c}\n")
    #     file.write(f"Average deviation from filling: {np.mean(abs_deviation_from_avg)}\n")
    #     file.write(f"STD of deviation from filling: {np.std(abs_deviation_from_avg)}\n")

    densities = np.diag(c_dagger_c).real
    max_density = np.max(densities)
    min_density = np.min(densities)
    odd_density = densities[0]
    even_density = densities[1]
    print("max_density, min density, odd_density, even_density:", max_density, min_density, odd_density, even_density)


    plt.plot(np.diag(c_dagger_c)[0:100].real)
    # plt.colorbar()
    plt.title(r"$c^{\dagger} c_{\uparrow}$, max: " + str(f"{max_density:.3f}") + ", min: " + str(f"{min_density:.3f}"), fontsize=16)
    # plt.savefig(save_folder + "/c_dagger_c_real_GS.pdf")
    # plt.savefig("figures_november_2/gs_plots/density_cdw_GS.png", dpi=300)
    # plt.savefig("figures_november_2/c_dagger_c_real_GS.png", dpi=300)
    # plt.savefig(os.path.join(save_folder, f"c_d_i_c_j_mu_{chemical_potential}.png"))
    plt.show()





    # plt.pcolormesh(np.diag(c_dagger_c.real)[0:100].reshape(10,10))
    # plt.colorbar()
    # plt.title(r"$\langle c^{\dagger}_{i} c_{i} \rangle $")
    # # plt.savefig("figures_november_2/gs_plots/c_dagger_c_diagonal_real_GS.png", dpi=300)
    # plt.savefig(os.path.join(save_folder, f"number_operator_mu_{chemical_potential}.pdf"))
    # plt.show()

    # Create the pcolormesh plot
    cmesh = plt.pcolormesh(np.diag(c_dagger_c.real)[0:100].reshape(10, 10) - 0.5)
    # Add the colorbar and store its object
    cbar = plt.colorbar(cmesh)
    # Increase the font size of the colorbar tick labels
    cbar.ax.tick_params(labelsize=16)  # Set fontsize to 16
    # Set the title
    plt.title(r"$\langle c^{\dagger}_{i} c_{i}\rangle - 0.5 $", fontsize=16)
    plt.xticks([0, 2, 4, 6, 8, 10], [r"$0$", r"$2$", r"$4$", r"$6$", r"$8$", r"$10$"], fontsize=16)
    plt.yticks([0, 2, 4, 6, 8, 10], [r"$0$", r"$2$", r"$4$", r"$6$", r"$8$", r"$10$"], fontsize=16) 
    # Save the figure
    # plt.savefig(os.path.join(save_folder, f"number_operator_wrt_half_filling_mu_{chemical_potential}.pdf"))
    plt.show()
    
    plt.pcolormesh(abs(c_c))
    plt.colorbar()
    plt.title("c_c_abs")
    # plt.savefig(figure_folder + "c_c_real_mu_-5.1_t_250_PBC.png")
    plt.show()

    lmbda = lambda_bar[N_b:, :]
    temp_mat_1 = np.einsum("ki, kl, lj -> ij", lmbda, omega, lmbda)
    temp_mat_2 = np.einsum("ki, kj -> ij", lmbda, gamma)
    Ve_nm = 2.0* (temp_mat_1 - temp_mat_2 - temp_mat_2.T)

    # lmbda_spinless = lmbda[:, 0:N_b]
    print("shape of gamma =", np.shape(gamma))
    # print("shape of lmbda_spinless =", np.shape(lmbda_spinless) )
    delta_g = gamma - np.einsum("lk, ki -> li", omega, lmbda)

    delta_g_spinless = delta_g[:, 0:100]

    delta_g_kq =  (2/N_f)*np.einsum('qm, mn, kn -> qk', fourier_transform_matrix_spinless.conj(), delta_g_spinless , fourier_transform_matrix_spinless)

    plt.pcolormesh(np.real(delta_g_kq))
    plt.colorbar()
    plt.title("delta_g_kq")
    plt.show()

    diagonal_delta_g_kq = np.diag(np.real(delta_g_kq))
    diagonal_delta_g_kq_grid = diagonal_delta_g_kq.reshape(10, 10)


    plt.pcolormesh(diagonal_delta_g_kq_grid)
    plt.colorbar()
    plt.title(f"delta_g_q_diagonal, max = {np.max(diagonal_delta_g_kq_grid):.2f} min = {np.min(delta_g_kq):.2f}")
    # plt.savefig(os.path.join(save_folder, f"delta_g_q_colorbar_{chemical_potential}.pdf"))
    plt.show()

    plt.pcolormesh(np.real(delta_g))
    plt.colorbar()
    plt.title("delta_g_l_n spinless ...")
    # plt.savefig(os.path.join(save_folder, f"delta_g_q_{chemical_potential}.pdf"))
    plt.show()
    

    plt.pcolormesh(np.real(Ve_nm))
    plt.colorbar()
    plt.title("Ve_nsigma_ms")
    # plt.savefig(os.path.join(save_folder, f"Ve_ns1_ms2_{chemical_potential}.pdf"))
    plt.show()

    # fourier transform of Ve_nm
    Ve_nm_spinless = Ve_nm[0:100, 0:100]
    plt.pcolormesh(np.real(Ve_nm_spinless))
    plt.colorbar()
    plt.title("Re Ve_nm_spinless")
    plt.show()

    v_diag = np.diag(Ve_nm_spinless)
    plt.plot(v_diag)
    plt.title("V_e diag")
    plt.show()



    Ve_momentum_space = (2/N_f)*np.einsum('qm, mn, kn -> qk', fourier_transform_matrix_spinless.conj(), Ve_nm_spinless , fourier_transform_matrix_spinless)

    plt.pcolormesh(np.real(Ve_momentum_space))
    plt.colorbar()
    plt.title("Real_part_Ve_k_q")
    plt.show()

    plt.pcolormesh(np.imag(Ve_momentum_space))
    plt.colorbar()
    plt.title("Imag_part_Ve_k_q")
    plt.show()

    diagonal_Ve_k_q = np.diag(np.real(Ve_momentum_space))
    diagonal_Ve_k_q_grid = diagonal_Ve_k_q.reshape(10, 10)
    plt.pcolormesh(diagonal_Ve_k_q_grid)
    plt.colorbar()
    plt.title(f"Ve_q min = {np.min(diagonal_Ve_k_q_grid):.2f} max = {np.max(diagonal_Ve_k_q_grid):.2f}")
    # plt.savefig(os.path.join(save_folder, f"Ve_g_q_{chemical_potential}.pdf"))
    plt.show()



    scop = Ve_nm[:N_f, :N_f] * c_c[:N_f, :N_f]

    plt.pcolormesh(np.abs(scop))
    plt.colorbar()
    plt.title("absolute scop")
    # plt.savefig(os.path.join(save_folder, f"SCOP_real_space_mu_{chemical_potential}.png"))
    plt.show()


    plt.plot(np.abs(np.diag(scop[:int(N_f/2), int(N_f/2):])))
    plt.title("scop_abs")
    # plt.savefig(os.path.join(save_folder, f"SCOP_diag_Delta_m_mu_{chemical_potential}.png"))
    plt.show()


    plt.pcolormesh(np.abs(scop))
    plt.colorbar()
    plt.title("scop.real")
    # plt.savefig(figure_folder + "scop_real_mu_-5.3_t_95_PBC.png")
    plt.show()

    plt.pcolormesh(np.abs(scop[0:int(N_f/2), int(N_f/2):]))
    plt.colorbar()
    plt.title("scop.abs")
    # plt.savefig(os.path.join(save_folder, f"SCOP_real_space_mu_{chemical_potential}.png"))
    plt.show()

    print("scop.shape:", scop.shape)
    print("fourier_transform_matrix_spinless.shape:", fourier_transform_matrix_spinless.shape)
    scop_momentum_space = (2/N_f)*np.einsum('qm, mn, kn -> qk', fourier_transform_matrix_spinless.conj(), scop[:int(N_f/2), int(N_f/2):], fourier_transform_matrix_spinless)
    
    plt.pcolormesh(np.abs(scop_momentum_space))
    plt.colorbar()
    plt.title("scop_momentum_space.real")
    plt.show()




    plt.contourf(k_x, k_y, np.abs(np.diag(scop_momentum_space)).reshape(10,10))
    plt.colorbar()
    plt.title("scop_momentum_space_abs")
    # plt.savefig(os.path.join(save_folder, f"SCOP_momentum_space_mu_{chemical_potential}.png"))
    plt.show()

    plt.pcolormesh(scop.imag)
    plt.colorbar()
    plt.title("scop.imag")
    # plt.savefig(figure_folder + "scop_imag_mu_-5.1_t_250_PBC.png")
    plt.show()

    # lambda_ij_c_dagger_c_j_contraction = np.einsum("ij, j -> i",  lmbda, densities)
    # delta_r_ngs = delta_r + lambda_ij_c_dagger_c_j_contraction
    lambda_l_nsigma_electron_density_nsigma_contraction = np.einsum("ln, n -> l ", lambda_bar[100:, :], electron_density_i_sigma)  

    plt.plot(-2*lambda_l_nsigma_electron_density_nsigma_contraction + (2*g_num/omega_num))
    plt.title(r"$2 \sum_{js} \lambda_{i,js} \langle_{l, js} c^\dagger_is c_is \rangle + 2g/\omega$")
    plt.show()


    delta_x_ngs = delta_r[0:100] -  2 * lambda_l_nsigma_electron_density_nsigma_contraction + (2*g_num/omega_num)
    # delta_p_ngs = delta_r[100:200]

    plt.plot(delta_x_ngs)
    xmin = np.min(delta_x_ngs)
    xmax = np.max(delta_x_ngs)

    plt.title(
        rf"$"
        r"x^{{NGS}}_i = x^{{GS}}_i - 2 \lambda_{{i,j}} n_j + \frac{{2g}}{{\omega_0}}$"
        f"\nmin = {xmin:.4f}, max = {xmax:.4f}"
    )    
    # plt.savefig(f"{save_folder}/Delta_x_ngs.pdf")
    plt.show()

    # delta_x_ngs_grid = delta_x_ngs.reshape(10, 10)
    # plt.pcolormesh(delta_x_ngs_grid)
    # plt.colorbar(cmap="viridis", format="%.2f")
    # plt.title(r"$ \langle b^{\dagger}_{i} + b_{i} \rangle + 2g/\omega_0$", fontsize=16)
    # plt.xticks([0, 2, 4, 6, 8, 10], [r"$0$", r"$2$", r"$4$", r"$6$", r"$8$", r"$10$"], fontsize=16)
    # plt.yticks([0, 2, 4, 6, 8, 10], [r"$0$", r"$2$", r"$4$", r"$6$", r"$8$", r"$10$"], fontsize=16)
    # plt.savefig(f"{save_folder}/Delta_x_ngs_grid.pdf")
    # plt.show()

    delta_x_ngs_grid = delta_x_ngs.reshape(10, 10)

    # Create the pcolormesh plot
    cmesh = plt.pcolormesh(delta_x_ngs_grid)

    # Add the colorbar and store its object
    cbar = plt.colorbar(cmesh, cmap="viridis", format="%.2f")

    # Increase the font size of the colorbar tick labels
    cbar.ax.tick_params(labelsize=16)  # Set fontsize to 16

    # Set the title and other properties
    plt.title(r"$ \langle b^{\dagger}_{i} + b_{i} \rangle + 2g/\omega_0$", fontsize=16)
    plt.xticks([0, 2, 4, 6, 8, 10], [r"$0$", r"$2$", r"$4$", r"$6$", r"$8$", r"$10$"], fontsize=16)
    plt.yticks([0, 2, 4, 6, 8, 10], [r"$0$", r"$2$", r"$4$", r"$6$", r"$8$", r"$10$"], fontsize=16)

    # plt.savefig(f"{save_folder}/Delta_x_ngs_grid.pdf")
    plt.show()

    avg_lambda_density_contraction = np.mean(lambda_l_nsigma_electron_density_nsigma_contraction)

    std_lambda_density_contraction = np.std(lambda_l_nsigma_electron_density_nsigma_contraction)
    
    print("avg_lambda_density_contraction, std_lambda_density_contraction:", avg_lambda_density_contraction, std_lambda_density_contraction)
    plt.plot(lambda_l_nsigma_electron_density_nsigma_contraction)
    plt.title(r"$\sum_j \lambda_{ij} \langle c^\dagger_i c_j \rangle$ , avg: " + str(f"{avg_lambda_density_contraction:.3f}") + ", std: " + str(f"{std_lambda_density_contraction:.3f}"))
    plt.show()

    contraction_odd = lambda_l_nsigma_electron_density_nsigma_contraction[0]
    contraction_even = lambda_l_nsigma_electron_density_nsigma_contraction[1]
    print("contraction_odd, contraction_even:", contraction_odd, contraction_even)
    
def plot(lambda_bar): 
    
    plt.pcolormesh(lambda_bar)
    plt.colorbar()
    plt.title("lambda_bar full, Max value: " + str(f"{np.max(lambda_bar):.3f}"))
    # plt.savefig(save_folder + "/lambda_bar_full_GS.png", dpi=300)
    plt.show()

    lmbda = lambda_bar[N_b:, :int(N_f/2)]
    plt.pcolormesh(lmbda)
    plt.colorbar()
    plt.title("lambda_real_space spin up")
    # plt.savefig("figures_november_2/gs_plots/lambda_real_space_spin_up_GS.png", dpi=300)
    # plt.savefig(figure_folder + "lambda_real_space_mu_-5.1_t_200_PBC.png")
    plt.show()

    print("lmbda.shape:", lmbda.shape)
    print("lmbda_max, lmbda_min:", np.max(lmbda), np.min(lmbda))

    lambda_2 = lambda_bar[N_b:, int(N_f/2):]
    plt.pcolormesh(lambda_2)
    plt.colorbar()
    plt.title("lambda_real_space spin down")
    # plt.savefig(figure_folder + "lambda_real_space_mu_-5.1_t_200_PBC.png")
    plt.show()

    print("lambda_2.shape:", lambda_2.shape)
    print("lambda_2_max, lambda_2_min:", np.max(lambda_2), np.min(lambda_2))

    lambda_kl = (1/100) * np.einsum("ki, ij, lj  -> kl", fourier_transform_matrix_spinless, lmbda, fourier_transform_matrix_spinless.conj())

    plt.pcolormesh(lambda_kl.real)
    plt.colorbar()
    plt.title("lambda_kl")
    # plt.savefig(save_folder + "/lambda_kl_GS.png", dpi=300)
    plt.show()


    lmbda_q = np.diag(lambda_kl)

    plt.pcolormesh(k_x, k_y, lmbda_q.real.reshape(10,10))
    plt.colorbar()
    plt.title("lambda_bar_q")
    # plt.savefig(save_folder + "/lambda_bar_q_GS.png", dpi=300)
    plt.show()

    plt.plot(lmbda_q.real)
    plt.title("lambda_bar")

    print("lmbda_q_max, lmbda_q_min:", np.max(lmbda_q), np.min(lmbda_q))

    # plt.savefig(os.path.join(save_folder, f"lambda_bar_q_diag_mu_{chemical_potential}.png"))
    plt.show()


    # lmbda_q_square = lmbda_q.reshape(10,10)
    # lambda_q_square_real = lmbda_q_square.real
    # lambda_pi_pi = lambda_q_square_real[0,0]
    # lambda_zero_zero = lambda_q_square_real[5,5]
    # plt.pcolormesh(k_x, k_y, lmbda_q_square.real)
    # plt.colorbar(format="%.2f")
    # plt.title(r"$\lambda_{\pi} = $" + str(f"{lambda_pi_pi:.3f}" + ", " + r"$\lambda_{0} = $" + str(f"{lambda_zero_zero:.3f}")), fontsize=16)
    # ax = plt.gca()
    # # Increase the size of x-ticks and y-ticks
    # ax.tick_params(axis='both', labelsize=16)
    # ax.set_xlabel(r"$k_x$", fontsize=16)
    # ax.set_ylabel(r"$k_y$", fontsize=16)
    # ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2])
    # ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$"])
    # ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2])
    # ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$"])
    # plt.savefig(save_folder + "/lambda_q_real_GS.pdf", dpi=300)
    # plt.show()

    lmbda_q_square = lmbda_q.reshape(10, 10)
    lambda_q_square_real = lmbda_q_square.real
    lambda_pi_pi = lambda_q_square_real[0, 0]
    lambda_zero_zero = lambda_q_square_real[5, 5]

    # Create the pcolormesh plot
    cmesh = plt.pcolormesh(k_x, k_y, lmbda_q_square.real)

    # Add the colorbar and store its object
    cbar = plt.colorbar(cmesh)

    # Increase the font size of the colorbar tick labels
    cbar.ax.tick_params(labelsize=16)  # Set fontsize to 16

    # Set the title and other properties
    plt.title(
        r"$\lambda_{\pi} = $" + str(f"{lambda_pi_pi:.3f}") + ", " +
        r"$\lambda_{0} = $" + str(f"{lambda_zero_zero:.3f}"),
        fontsize=16
    )

    ax = plt.gca()
    # Increase the size of x-ticks and y-ticks
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel(r"$k_x$", fontsize=16)
    ax.set_ylabel(r"$k_y$", fontsize=16)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$"])
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$"])

    # plt.savefig(save_folder + "/lambda_q_real_GS.pdf", transparent= "True" ,bbox_inches='tight', dpi=300)
    plt.show()




# importing the data and seeing the plot. 

print("omemga_num:", omega_num)
print("g_num:", g_num)
# chemical_potential = -2*g_num**2/omega_num
# chemical_potential = -2.60
print("chemical_potential:", chemical_potential)




data_folder = f"/mnt/ceph/vksharma/warmup_ite_data_jnn_-1_j_nnn_-0.00_omega_{omega_num:.0f}_g_{g_num:.2f}_nov27_2025/mu_{chemical_potential:.2f}/"

save_folder = current_directory + f"/GS_data_folder/GS_data_june_2026_{omega_num:.1f}_g_{g_num:.2f}_mu_{chemical_potential:.2f}"


if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# delta_r = np.load(data_folder + f"Delta_R_{t_max}.npy")
# Gamma_b = np.load(data_folder + f"Gamma_b_{t_max}.npy")
# lambda_bar = np.load(data_folder + f"lambda_bar_{t_max}.npy")
# G_m = np.load(data_folder + f"Gamma_m_{t_max}.npy")

t_max = 550

try:
    delta_r = np.load(data_folder + f"delta_r_mu_{chemical_potential:.1f}_t_{t_max}.npy")
    Gamma_b = np.load(data_folder + f"Gamma_b_mu_{chemical_potential:.1f}_t_{t_max}.npy")
    lambda_bar = np.load(data_folder + f"lambda_bar_mu_{chemical_potential:.1f}_t_{t_max}.npy")
    G_m = np.load(data_folder + f"Gamma_m_mu_{chemical_potential:.1f}_t_{t_max}.npy")

except: 
    delta_r = np.load(data_folder + f"Delta_R_{t_max}.npy")
    Gamma_b = np.load(data_folder + f"Gamma_b_{t_max}.npy")
    lambda_bar = np.load(data_folder + f"lambda_bar_{t_max}.npy")
    G_m = np.load(data_folder + f"Gamma_m_{t_max}.npy")

# np.save(save_folder + f"/Delta_R_GS.npy", delta_r)
# np.save(save_folder + f"/Gamma_b_GS.npy", Gamma_b)
# np.save(save_folder + f"/lambda_bar_GS.npy", lambda_bar)
# np.save(save_folder + f"/Gamma_m_GS.npy", G_m)

# input_dir = os.path.join(pwd, "GS_data_folders/GS_data_omega_2.50_g_2.00_mu_-3.20")


plot_delta_r_real_space(delta_r.real)
plot_Gamma_b_momentum_space_from_real_space(Gamma_b.real, fourier_transform_matrix_spinless)


print("Save folder created:", save_folder)
plot_fermion_correlations(G_m, delta_r, lambda_bar, omega, gamma, fourier_transform_matrix_spinless)

plot(lambda_bar.real)

