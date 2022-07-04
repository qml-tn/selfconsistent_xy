import numpy as np
import os
from scipy.integrate import odeint
import time
#import matplotlib.pyplot as plt
#from scipy.linalg import expm, sinm, cosm
#from tqdm.notebook import tqdm 

### By analogy modify this function to have the time-evolution, change name to, for instance: single_trajectory_gamma_mat_evolution()
def single_trajectory_gamma_mat_evolution(params):
    g, eta, N, angle_init, dt, ntim, savedir, output, anglesinit = params
#    if gsinit:
#        etainit = eta
    if (output == 'save'):
#        if gsinit:
#            filepath = os.path.join(savedir, "phaseDiagram", 'etainit_eta',"ginit"+str(ginit), "n"+str(n), 'eta'+str(eta), "ntim" + str(ntim), "g"+str(g))
        #else:
        filepath = os.path.join(savedir, "phaseDiagram_gamma_mat", "angle_init"+str(angle_init), "N"+str(N), 'eta'+str(eta), "ntim" + str(ntim), "g"+str(g))
        if(os.path.exists(filepath)):
            if (len(os.listdir(filepath)) >= 2):
                return True
        else:
            os.makedirs(filepath)
    
    t_list = np.linspace(0, dt*ntim, ntim+1)
    disorder_list = np.array([0., 0.01, 0.1])*np.pi
    theta = 0.5*np.pi + angle_init # initial-condition theta = pi/2, phi = 0 is Ising-model along X 
    phi = angle_init  #
    lista_params = [N, theta, phi, g, eta] 
    Sigma_Z_LatticeMean, Gamma_t_list_for_different_eps = Sigma_Z_LatticeMean_for_diff_perturb_strength( t_list, disorder_list, lista_params ) # this is for a fixed g
    S_fromCovMAt_t_for_diff_eps = ent_entropy_from_CovMat(Gamma_t_list_for_different_eps) # this is for a fixed g
    S1_t_for_diff_eps = ent_entropy_from_Gamma_1(Gamma_t_list_for_different_eps) # this is for a fixed g
    S2_t_for_diff_eps = ent_entropy_from_Gamma_2(Gamma_t_list_for_different_eps) # this is for a fixed g  
    #if(output == 'lyap' and m > 0):
    #    _, R = np.linalg.qr(np.reshape(sol[-1, 3*n:], [3*n, m]))
    #    return np.log(abs(np.diag(R))/(dt*ntim)), ent, fid, sol
    #elif(output == 'tlyap' and m > 0):
    #    eps = 1e-7
        # tlyap = np.log(lyap)+np.cumsum(np.log(abs(rescaling[:ntim, :])), 0)
    #    return tlist, z, lyap, rescaling, ent, fid, sol
    if (output == 'save'):
        filename = os.path.join(filepath, "t_list.npy")
        np.save(filename, np.array(t_list))
        filename = os.path.join(filepath, "Sigma_Z_LatticeMean.npy")
        np.save(filename, np.array(Sigma_Z_LatticeMean))
        filename = os.path.join(filepath, "ent_cov.npy")
        np.save(filename, np.array(S_fromCovMAt_t_for_diff_eps))
        filename = os.path.join(filepath, "ent_1.npy")
        np.save(filename, np.array(S1_t_for_diff_eps))
        filename = os.path.join(filepath, "ent_2.npy")
        np.save(filename, np.array(S2_t_for_diff_eps))
        #filename = os.path.join(filepath, "fid.npy")
        #np.save(filename, np.array(fid))
        #if (m > 0):
        #    filename = os.path.join(filepath, "lyap.npy")
        #    eps = 1e-7
            # tlyap = (np.log(
            #     lyap)+np.cumsum(np.log(abs(rescaling[:ntim, :])), 0))/np.reshape(tlist+eps, [ntim, 1])
        #    np.save(filename, np.array(lyap))
        #    filename = os.path.join(filepath, "rescaling.npy")
        #    np.save(filename, np.array(rescaling))
        return filepath
    return np.array(t_list), np.array(Sigma_Z_LatticeMean), np.array(S_fromCovMAt_t_for_diff_eps), np.array(S1_t_for_diff_eps), np.array(S2_t_for_diff_eps) 

### This is the main function :  #[N, theta, phi, g, eta] = lista_params  
def Sigma_Z_LatticeMean_for_diff_perturb_strength(t_list, eps_list, lista_params):
  [N, theta, phi, g, eta] = lista_params  
  psi_0_list = [] # list of initial-conditions ( prod_i{ |theta+eps_i, phi+eps_i> } ) for each strenght of the perturb (eps_list[k])
  # now we construct the initial-state psi_0 as a product state of the N-qubits: prod_i{ |theta+eps_i, phi+eps_i> }
  for k in range(len(eps_list)): # for each perturb-strength we construct the initial-state 
    psi_0 = np.zeros([N,2], dtype = complex) 
    eps_theta_i = np.random.uniform(-eps_list[k], eps_list[k], N)
    eps_phi_i = np.random.uniform(-eps_list[k], eps_list[k], N)
    for i in range(N):
      theta_i = theta + eps_theta_i[i]
      phi_i = phi +  eps_phi_i[i]
      psi_0[i] = np.array( [np.cos( theta_i/2  ), np.exp(1j*phi_i)*np.sin( theta_i/2  )] ) 
    psi_0_list.append(psi_0)
  psi_0_list = np.array(psi_0_list)
  # now loop over the initial-states psi_0 inside the psi_0_list array to construct array sigmaZ_mean_over_lattice
  Sigma_Z_LatticeMean = []
  Gamma_t_list_for_different_eps = []
  for ind_psi_0 in range(len(psi_0_list)):
    psi_0 = psi_0_list[ind_psi_0] ### select a initial state from the list, this means a fixed eps (perturb-strenght) is fixed
    psi_0 = psi_0/np.linalg.norm(psi_0,axis=1,keepdims=True) # normalize it, although from the definition it should be already normalized
    Gamma_t_0 = Gamma_Mat_from_psi(psi_0)
    Gamma_t_0_upperLeft = Gamma_t_0[:N, :N] 
    ### from Gamma_t_0_upperLeft  construct the the h_vec_t[i] = (g)*( 2*Gamma_t_0_upperLeft[i][i]-1 ) 
    h_vec_0 = np.zeros(N, dtype = complex)
    h_vec_0 = (g)*( 2*np.diag(Gamma_t_0_upperLeft)-1 ) # this is the vector h_vec
    #eta = 0.0
    xx = np.cos(eta) + np.sin(eta)
    yy = np.cos(eta) - np.sin(eta)
    H_mat_0 = np.zeros([2*N,2*N], dtype = complex) ### this is a constant matrix, the diag-part will contain the time-evolution
    A_mat = np.zeros([N,N], dtype = complex)
    B_mat = np.zeros([N,N], dtype = complex)
    for i in range(N-1):
      A_mat[i][i+1] = -xx
      A_mat[i+1][i] = -xx
      B_mat[i][i+1] = yy
      B_mat[i+1][i] = -yy
    H_mat_0[:N, :N] = -np.conj(A_mat)
    H_mat_0[:N, N:2*N] = B_mat
    H_mat_0[N:2*N, :N] = -np.conj(B_mat)
    H_mat_0[N:2*N, N:2*N] = A_mat    
    H_mat_0 = np.array(H_mat_0) ### this is the matrix not taking yet the H_diag_0 && is always constant
    Gamma_t_0_vec = Gamma_t_0.reshape(len(Gamma_t_0)**2) # vector of len == (2*N)**2
    Gamma_t_0_vec_decomposed = np.zeros(2*len(Gamma_t_0_vec)) # we construct a vector of twice len containing the Re and Im parts of the vector comming from the matrix
    Gamma_t_0_vec_decomposed[:len(Gamma_t_0_vec)] = np.real(Gamma_t_0_vec)
    Gamma_t_0_vec_decomposed[len(Gamma_t_0_vec):] = np.imag(Gamma_t_0_vec)
    parameters = (g, H_mat_0) ### H_mat_0 equivalent to H_XY is a constant matrix (and has no diagonal terms in principle)
    Gamma_vec_decomposed_t = odeint( derivative_Gamma, Gamma_t_0_vec_decomposed, t_list, (parameters,) )  
    nn = len(Gamma_vec_decomposed_t[0]) ### nn == 2*( (2*N)**2 )
    Gamma_vec_t_Re = Gamma_vec_decomposed_t[:,:int(nn/2)]
    Gamma_vec_t_Im = Gamma_vec_decomposed_t[:,int(nn/2):]
    Gamma_vec_t = Gamma_vec_t_Re + 1j*Gamma_vec_t_Im
    Gamma_Mat_t_list = [] ### list of Gamma(t) matrices at each time
    h_vec_t_list = [] ### list of h_vec vectors at each time 
    for ind_t in range(len(Gamma_vec_t)): # len(Gamma_vec_t) == ntim + 1 == len(t_list)
      Gamma_Matrix_t = Gamma_vec_t[ind_t].reshape( int(np.sqrt(len(Gamma_vec_t[0]))),  int(np.sqrt(len(Gamma_vec_t[0]))) )
      Gamma_Mat_t_list.append(Gamma_Matrix_t)
      Gamma_t_upperLeft = Gamma_Matrix_t[:N, :N] 
      #h_vec = np.zeros(N, dtype = complex)
      h_vec = (g)*( 2*np.diag(Gamma_t_upperLeft)-1 ) # this is the vector  (h_vec[i] = g * mean_sigma_Z[i])
      h_vec_t_list.append(h_vec)
    h_vec_t_list = np.array(h_vec_t_list)  ### this is the final list containing h_vec for each time
    sigmaZ_mean_over_lattice = (1/g)*np.mean(h_vec_t_list,-1) # here is an average over the lattice sites so it will be only of len of tlist
    Sigma_Z_LatticeMean.append(sigmaZ_mean_over_lattice)
    Gamma_t_list_for_different_eps.append(np.array(Gamma_Mat_t_list))
  return np.array(Sigma_Z_LatticeMean), np.array(Gamma_t_list_for_different_eps)
### Sigma_Z_LatticeMean.shape == (len(eps_list), len(t_list))
### Gamma_t_list_for_different_eps.shape == (len(eps_list), len(t_list), 2*N, 2*N)

def Gamma_Mat_from_psi(psi_0):  ### psi_0 is here a spin-product-state, even with eps (disorder)
    s_x = np.array([ [0,1],[1,0] ] ) 
    s_y = np.array([ [0,-1j],[1j,0] ]) 
    s_z = np.array([ [1,0],[0,-1] ] ) 
    s_p = (s_x + 1j*s_y)/2  
    s_m = (s_x - 1j*s_y)/2
    N = len(psi_0)
    gamma_ap_a = np.zeros([N,N], dtype = complex)
    gamma_ap_ap = np.zeros([N,N], dtype = complex)
    gamma_a_a = np.zeros([N,N], dtype = complex)
    gamma_a_ap = np.zeros([N,N], dtype = complex)
    # we can directly fill the easy diagonal-terms:
    for i in range(N):
      gamma_ap_a[i][i] = np.dot(np.conj(np.transpose(psi_0[i])), np.dot(np.dot(s_p, s_m),psi_0[i]) ) # is the minus-sign at the beginning correct ?
      gamma_ap_ap[i][i] = np.dot(np.conj(np.transpose(psi_0[i])), np.dot(np.dot(s_p, s_p),psi_0[i]) )
      gamma_a_a[i][i] = np.dot(np.conj(np.transpose(psi_0[i])), np.dot(np.dot(s_m, s_m),psi_0[i]) )
      gamma_a_ap[i][i] = np.dot(np.conj(np.transpose(psi_0[i])), np.dot(np.dot(s_m, s_p),psi_0[i]) )
    # now we loop over i and j to fill non-diagonal terms:
    for i in range(N):
      for j in range(N):
        # we need to discriminate if i<j or i>j:
        if i<j:
          range_for_factor_prod = range(i+1,j)
          factor_prod = 1.0
          for k in range_for_factor_prod:
            factor_prod *= -np.dot(np.conj(np.transpose(psi_0[k])), np.dot(s_z,psi_0[k]) )    
          gamma_ap_a[i][j] += (-np.dot(np.conj(np.transpose(psi_0[i])), np.dot(np.dot(s_p, s_z),psi_0[i]) ))*factor_prod*np.dot(np.conj(np.transpose(psi_0[j])), np.dot(s_m,psi_0[j]) )
          gamma_ap_ap[i][j] += (-np.dot(np.conj(np.transpose(psi_0[i])), np.dot(np.dot(s_p, s_z),psi_0[i]) ))*factor_prod*np.dot(np.conj(np.transpose(psi_0[j])), np.dot(s_p,psi_0[j]) )  
          gamma_a_a[i][j] += (-np.dot(np.conj(np.transpose(psi_0[i])), np.dot(np.dot(s_m, s_z),psi_0[i]) ))*factor_prod*np.dot(np.conj(np.transpose(psi_0[j])), np.dot(s_m,psi_0[j]) )  
          gamma_a_ap[i][j] += (-np.dot(np.conj(np.transpose(psi_0[i])), np.dot(np.dot(s_m, s_z),psi_0[i]) ))*factor_prod*np.dot(np.conj(np.transpose(psi_0[j])), np.dot(s_p,psi_0[j]) )  
        elif i>j:
          range_for_factor_prod = range(j+1,i)
          factor_prod = 1.0
          for k in range_for_factor_prod:
            factor_prod *= -np.dot(np.conj(np.transpose(psi_0[k])), np.dot(s_z,psi_0[k]) )     
          gamma_ap_a[i][j] += (np.dot(np.conj(np.transpose(psi_0[i])), np.dot(s_p,psi_0[i])))*factor_prod*(-np.dot(np.conj(np.transpose(psi_0[j])), np.dot(np.dot(s_z, s_m),psi_0[j])) )
          gamma_ap_ap[i][j] += (np.dot(np.conj(np.transpose(psi_0[i])), np.dot(s_p,psi_0[i])))*factor_prod*(-np.dot(np.conj(np.transpose(psi_0[j])), np.dot(np.dot(s_z, s_p),psi_0[j])) )
          gamma_a_a[i][j] += (np.dot(np.conj(np.transpose(psi_0[i])), np.dot(s_m,psi_0[i])))*factor_prod*(-np.dot(np.conj(np.transpose(psi_0[j])), np.dot(np.dot(s_z, s_m),psi_0[j])) )
          gamma_a_ap[i][j] += (np.dot(np.conj(np.transpose(psi_0[i])), np.dot(s_m,psi_0[i])))*factor_prod*(-np.dot(np.conj(np.transpose(psi_0[j])), np.dot(np.dot(s_z, s_p),psi_0[j])) )
    Gamma_t_0 = np.zeros([2*N,2*N], dtype = complex)
    Gamma_t_0[:N, :N] = gamma_ap_a
    Gamma_t_0[:N, N:2*N] = gamma_ap_ap
    Gamma_t_0[N:2*N, :N] = gamma_a_a
    Gamma_t_0[N:2*N, N:2*N] = gamma_a_ap
    return np.array(Gamma_t_0)

# inside the main function (Sigma_Z_LatticeMean_for_diff_perturb_strength) we call this derivative-function inside the odeint solver 
def derivative_Gamma(Gamma_vec_decomposed,t, params): ### Here enter the H_mat_0 as a param !
  (g, H_mat_0) = params
  nn = len(Gamma_vec_decomposed) # this Gamma_vec_decomposed is of len == 2*((2N)**2) and real, then nn==2*((2*N)**2)
  d_Gamma_vec_decomposed = np.zeros(nn)
  Gamma_vec_Re = Gamma_vec_decomposed[:int(nn/2)]
  Gamma_vec_Im = Gamma_vec_decomposed[int(nn/2):]
  Gamma_vec = Gamma_vec_Re + 1j*Gamma_vec_Im # this Gamma_vec is a vector of len == (2N)**2 and complex
  Gamma_mat = Gamma_vec.reshape( int(np.sqrt(len(Gamma_vec))),  int(np.sqrt(len(Gamma_vec))) ) # matrix of size == (2*N) times (2*N)
  Gamma_mat_upperLeft = Gamma_mat[:N, :N] 
  h_vec_t = np.zeros(N, dtype = complex)
  h_vec_t = (g)*( 2*np.diag(Gamma_mat_upperLeft)-1 ) # this is the vector h_vec  
  #h_vec_t = (g)*(2*np.diag(Gamma_mat_upperLeft)-1) # this is the vector h_vec  
  H_diag_t = np.zeros([2*N,2*N], dtype = complex)
  H_diag_t[:N, :N] = np.eye(N)*np.conj(h_vec_t )*2   ### a change here ! : A_ii = -2*h_vec[i]
  H_diag_t[N:2*N, N:2*N] = -np.eye(N)*h_vec_t*2  
  H_FullMat_t = H_mat_0 + H_diag_t # the time-dependence of the H_mat is in the diagonal only
  #RHS_mat = 2j*(np.dot(Gamma_mat, H_FullMat_t )-np.dot(H_FullMat_t , Gamma_mat))
  RHS_mat = 1j*(np.dot(Gamma_mat, H_FullMat_t )-np.dot(H_FullMat_t , Gamma_mat))  
  #d_Gamma_vec = np.zeros(nn, dtype = complex)
  d_Gamma_vec = RHS_mat.reshape(len(RHS_mat)**2)
  d_Gamma_vec_decomposed[:int(nn/2)] = np.real(d_Gamma_vec)
  d_Gamma_vec_decomposed[int(nn/2):] = np.imag(d_Gamma_vec)
  return d_Gamma_vec_decomposed

# Here below I tried to calculate ent from the Covariance-mat (which can be found directly from Gamma(t))
# Cov_mat = -1j*Omega*(2*Gamma-Id)*Omegaˆ{adjoint},  Omegaˆ{adjoint} =  np.conj(np.transpose(Omega))
# For this function here   Gamma_Mat_t_list.shape == (len(eps_list), len(t_list), 2*N, 2*N) we get: np.array(S_A_t_for_diff_eps).shape == (len(eps_list), len(t_list))
def ent_entropy_from_CovMat(Gamma_Mat_t_list): ## a different way to calculate entropy, here I assume Cov_mat == iJ[psi] in the other paper
  eps = 1e-20 
  number_eps_values = len(Gamma_Mat_t_list) # this is == len(eps_list)
  time_steps = len(Gamma_Mat_t_list[0])     # this is == len(t_list)
  S_A_t_for_diff_eps = []
  for ind_eps in range(number_eps_values):
    S_A_t = np.zeros(time_steps)
    for ind_t in range(time_steps):
      Gamma_Mat_t = Gamma_Mat_t_list[ind_eps][ind_t]
      N =  int(len(Gamma_Mat_t)/2)
      Id = np.eye(N)
      Cov_mat = np.zeros([len(Gamma_Mat_t), len(Gamma_Mat_t)], dtype=complex)
      Omega = np.zeros([len(Gamma_Mat_t), len(Gamma_Mat_t)], dtype=complex)
      Omega[:N, :N] = Id
      Omega[:N, N:2*N] = Id
      Omega[N:2*N, :N] = 1j*Id
      Omega[N:2*N, N:2*N] = -1j*Id
      Omega = Omega*(1/np.sqrt(2))
      Cov_mat = -1j*np.dot(Omega, np.dot( 2*Gamma_Mat_t-np.eye(2*N, dtype= complex), np.conj(np.transpose(Omega)) ) ) # this is supposed to be == iJ[psi]
      Mat_A = ( np.eye(2*N) + Cov_mat )/2
      lam = abs(np.linalg.eigvalsh(Mat_A)) + eps    
      S_A = - np.sum(lam*np.log(lam))
      S_A_t[ind_t] = S_A#/(N)
    S_A_t_for_diff_eps.append(S_A_t)
  return np.array(S_A_t_for_diff_eps)
### This 1-st way: we apply the formula of Mat_A = (Id + Gamma_Mat_t)/2 (to calculate from the eigenvalues of Mat_A )
def ent_entropy_from_Gamma_1(Gamma_Mat_t_list): ## a different way to calculate entropy !
  eps = 1e-20 
  number_eps_values = len(Gamma_Mat_t_list)
  time_steps = len(Gamma_Mat_t_list[0])
  S_A_t_for_diff_eps = []
  for ind_eps in range(number_eps_values):
    S_A_t = np.zeros(time_steps)
    for ind_t in range(time_steps):
      Gamma_Mat_t = Gamma_Mat_t_list[ind_eps][ind_t]
      NN = len(Gamma_Mat_t)
      Id = np.eye(len(Gamma_Mat_t))
      Mat_A = (Id + Gamma_Mat_t)/2
      lam = abs(np.linalg.eigvalsh(Mat_A)) + eps    
      S_A = - np.sum(lam*np.log(lam))
      S_A_t[ind_t] = S_A#/(NN/4) # this normalization I made to make it coincide with ent(t=0) from the previous calculation (from the dynamics of S_q)
    S_A_t_for_diff_eps.append(S_A_t)
  return np.array(S_A_t_for_diff_eps)
### This 2-nd way: we calculate from the eigenvalues of Gamma (this is supposed to hold for a product-state only ?)
def ent_entropy_from_Gamma_2(Gamma_Mat_t_list):
  eps = 1e-20 
  number_eps_values = len(Gamma_Mat_t_list)
  time_steps = len(Gamma_Mat_t_list[0])
  S_A_t_for_diff_eps = []
  for ind_eps in range(number_eps_values):
    S_A_t = np.zeros(time_steps)
    for ind_t in range(time_steps):
      Gamma_Mat_t = Gamma_Mat_t_list[ind_eps][ind_t]
      lam_all = np.linalg.eigvalsh(Gamma_Mat_t)
      lam_abs = abs(np.linalg.eigvalsh(Gamma_Mat_t)) + eps    
      S_A = - np.sum(lam_abs*np.log(lam_abs))
      #S_A = - np.sum(lam_all*np.log(lam_abs))
      S_A_t[ind_t] = S_A#/(len(Gamma_Mat_t)/2)
    S_A_t_for_diff_eps.append(S_A_t)
  return np.array(S_A_t_for_diff_eps)
### shape of these entropies: ( len(disorder_list), len(t_list) )
