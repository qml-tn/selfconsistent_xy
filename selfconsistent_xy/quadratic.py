import numpy as np
import os
from scipy.integrate import odeint
import time

def Fidelity(S_x, S_y, S_z):
    ind_t = 0
    epsilon = 1e-12
    # at a fixed time in the index of the S_x,y,z, at each time we are going to append to a list the desired quantity to be calculated !
    c_1_q_0 = np.sqrt(np.abs((S_z[ind_t] + 1)/2)) + epsilon
    c_4_q_0 = np.divide((S_x[ind_t] + 1j*S_y[ind_t])/2, c_1_q_0)
    c_1_q_pow2_0 = np.power(np.abs(c_1_q_0), 2)
    c_4_q_pow2_0 = np.power(np.abs(c_4_q_0), 2)
    Norm_state_squared_0 = c_1_q_pow2_0 + c_4_q_pow2_0
    Norm_state_0 = np.sqrt(Norm_state_squared_0)
    c_1_q_new_0 = np.divide(c_1_q_0, Norm_state_0)
    c_4_q_new_0 = np.divide(c_4_q_0, Norm_state_0)
    # These coefficentes define the normalized-initial-state
    c_1_q_0_conj = np.conj(c_1_q_new_0)
    c_4_q_0_conj = np.conj(c_4_q_new_0)

    c_1_q = np.sqrt(np.abs((S_z + 1)/2))
    c_4_q = np.divide((S_x + 1j*S_y)/2, c_1_q)
    c_1_q_pow2 = np.power(np.abs(c_1_q), 2)
    c_4_q_pow2 = np.power(np.abs(c_4_q), 2)
    Norm_state_squared = c_1_q_pow2 + c_4_q_pow2
    Norm_state = np.sqrt(Norm_state_squared)
    c_1_q_new = np.divide(c_1_q, Norm_state)
    c_4_q_new = np.divide(c_4_q, Norm_state)

    inner_prod = c_1_q_0_conj*c_1_q_new + c_4_q_0_conj*c_4_q_new

    echo = np.sum(np.log(np.abs(inner_prod)), axis=1)
    return echo


def entanglement_entropy(n, S_x, S_y, S_z):
    ntim = len(S_x)
    S_A_t = np.zeros(ntim)
    for ind_t in range(ntim):
        Yen_11 = np.zeros((n, n), dtype=complex)
        Yen_12 = np.zeros((n, n), dtype=complex)
        Yen_21 = np.zeros((n, n), dtype=complex)
        Yen_22 = np.zeros((n, n), dtype=complex)
        k_l = (np.pi*np.arange(-n+0.5, n))/n

        j = np.array([np.arange(n)])
        pid=np.transpose(j)-j # phase index difference

        for ind_k in range(2*n):
            if k_l[ind_k] < 0:
                ind_qp = -ind_k-1
                # we need to divide by approx 2*n the actual lattice-size
                Yen_12 += np.exp(1j*pid*k_l[ind_k]) * \
                                      (-S_x[ind_t, ind_qp]-1j *
                                        S_y[ind_t, ind_qp])/(2*n)
                Yen_21 += -np.exp(1j*pid*k_l[ind_k]) * \
                    (-S_x[ind_t, ind_qp]+1j *
                      S_y[ind_t, ind_qp])/(2*n)
                Yen_11 += np.exp(1j*pid*k_l[ind_k]) * \
                                      (S_z[ind_t, ind_qp])/(2*n)
                Yen_22 += -np.exp(-1j*pid*k_l[ind_k]) * \
                    (S_z[ind_t, ind_qp])/(2*n)
            else:
                ind_qp = ind_k-n
                Yen_12 += np.exp(1j*pid*k_l[ind_k]) * \
                                      (S_x[ind_t, ind_qp] + 1j *
                                        S_y[ind_t, ind_qp])/(2*n)
                Yen_21 += -np.exp(1j*pid*k_l[ind_k]) * \
                    (S_x[ind_t, ind_qp] - 1j *
                      S_y[ind_t, ind_qp])/(2*n)
                Yen_11 += np.exp(1j*pid*k_l[ind_k]) * \
                                      (S_z[ind_t, ind_qp])/(2*n)
                Yen_22 += -np.exp(-1j*pid*k_l[ind_k]) * \
                    (S_z[ind_t, ind_qp])/(2*n)

        
        # Now construct the covariant-matrices iJ_psi
        iJ_psi = np.zeros((2*n, 2*n), dtype=complex)

        iJ_psi[:n, :n] = Yen_11
        iJ_psi[:n, n:2*n] = Yen_12
        iJ_psi[n:2*n, :n] = Yen_21
        iJ_psi[n:2*n, n:2*n] = Yen_22

        Id = np.eye(2*n)
        Mat_A = (Id + iJ_psi)/2
        eps = 1e-13
        lam = abs(np.linalg.eigvalsh(Mat_A)) + eps

        S_A = - np.sum(lam*np.log(lam))
        S_A_t[ind_t] = S_A

    return S_A_t


def single_trajectory_benettin_rescaling(params):
    g, eta, n, m, ginit, etainit, dt, ntim, savedir, output, gsinit = params

    if gsinit:
        etainit = eta

    if (output == 'save'):
        if gsinit:
            filepath = os.path.join(savedir, "phaseDiagram", 'etainit_eta',
                                    "ginit"+str(ginit), "n"+str(n), 'eta'+str(eta), "g"+str(g))
        else:
            filepath = os.path.join(savedir, "phaseDiagram", 'etainit'+str(
                etainit), "ginit"+str(ginit), "n"+str(n), 'eta'+str(eta), "g"+str(g))
        if(os.path.exists(filepath)):
            if (len(os.listdir(filepath)) >= 2):
                return True
        else:
            os.makedirs(filepath)

    yinit = initial_state(n, m=m, eta=etainit, h=ginit)

    if (m > 0):
        q = (np.array(range(n))+0.5)*np.pi/(n)
        ce = np.cos(eta)
        se = np.sin(eta)
        By = 4*np.sin(q)*(ce-se)
        Bz = 4*(np.cos(q)*(ce+se))
        par = (g, n, By, Bz, m)

        tlist = np.zeros([ntim+1])
        lyap = np.ones([ntim+1, m])
        z = np.zeros([ntim+1])
        sol = np.zeros([ntim+1, len(yinit)])
        rescaling = np.ones([ntim+1, m])
        sol[0, :] = yinit
        z[0] = np.mean(yinit[2*n:3*n])

        nt = int(dt/0.01)
        t = np.linspace(0, dt, nt)
        tstartAll = time.clock()
        for i in range(ntim):
            tstart = time.clock()
            sol0 = odeint(dfbenettin, yinit, t, (par,),
                          hmax=5e-4, atol=1e-18, rtol=1e-13)
            tend = time.clock()
            yinit, lyap[i+1, :], rescaling[i+1,
                                           :] = get_lyapunov_coefficient_and_rescale(sol0[-1], n, m, thresh=1e2)
            sol[i+1, :] = yinit
            tlist[i+1] = (i+1)*dt
            z[i+1] = np.mean(yinit[2*n:3*n])
            print('done:', (i+1)/ntim, "time: "+str(tend-tstart), end="\r")
            # print('done:',(i+1)/ntim,"time: "+str(tend-tstart))
        tendAll = time.clock()
        # print("Finished evolution: "+str(tendAll-tstartAll))

    else:
        lyap = None
        rescaling = None
        # q = (np.array(range(n)))*np.pi/(n-1)
        q = (np.array(range(n))+0.5)*np.pi/(n)
        ce = np.cos(eta)
        se = np.sin(eta)
        By = 4*np.sin(q)*(ce-se)
        Bz = 4*(np.cos(q)*(ce+se))
        par = (g, n, By, Bz)

        tstart = time.clock()
        tlist = np.linspace(0, dt*ntim, ntim+1)
        sol = odeint(derivative, yinit, tlist, (par,),
                     hmax=5e-4, atol=1e-16, rtol=1e-13)
        z = get_transverse_magnetization(sol)
        tend = time.clock()
        # print("Finished evolution: "+str(tend-tstart))

    x = sol[:, :n]
    y = sol[:, n:2*n]
    z = sol[:, 2*n:3*n]

    t1 = time.time()
    fid = Fidelity(x, y, z)
    t2 = time.time()
    # print(f"Finished fidelity: {t2-t1}s")
    t1 = time.time()
    ent = entanglement_entropy(n, x, y, z)
    t2 = time.time()
    # print(f"Finished entropy: {t2-t1}s")

    if(output == 'lyap' and m > 0):
        _, R = np.linalg.qr(np.reshape(sol[-1, 3*n:], [3*n, m]))
        return np.log(abs(np.diag(R))/(dt*ntim)), ent, fid, sol
    elif(output == 'tlyap' and m > 0):
        eps = 1e-7
        # tlyap = np.log(lyap)+np.cumsum(np.log(abs(rescaling[:ntim, :])), 0)
        return tlist, z, lyap, rescaling, ent, fid, sol
    elif(output == 'save'):
        filename = os.path.join(filepath, "tlist.npy")
        np.save(filename, np.array(tlist))
        filename = os.path.join(filepath, "z.npy")
        np.save(filename, np.array(z))
        filename = os.path.join(filepath, "ent.npy")
        np.save(filename, np.array(ent))
        filename = os.path.join(filepath, "fid.npy")
        np.save(filename, np.array(fid))
        if (m > 0):
            filename = os.path.join(filepath, "lyap.npy")
            eps = 1e-7
            # tlyap = (np.log(
            #     lyap)+np.cumsum(np.log(abs(rescaling[:ntim, :])), 0))/np.reshape(tlist+eps, [ntim, 1])
            np.save(filename, np.array(lyap))
            filename = os.path.join(filepath, "rescaling.npy")
            np.save(filename, np.array(rescaling))
        return filepath
    return np.array(tlist), np.array(z), np.array(lyap), np.array(rescaling), ent, fid, sol


def dfbenettin(y, t, params):
    (g, n, By, Bz, m) = params

    ds = derivative(y[:3*n], t, params[:4])

    M = np.reshape(y[3*n:], [3*n, m])
    Jac = Jacobian(y[:3*n], g, By, Bz)

    dM = Jac.dot(M)

    dy = np.concatenate([ds, np.reshape(dM, [-1])])
    return np.array(dy)


def get_lyapunov_coefficient_and_rescale(sol, n, m, thresh=1e4):
    Q, R = np.linalg.qr(np.reshape(sol[3*n:], [3*n, m]))
    lyap = abs(np.diag(R))
    if (np.max(lyap) < thresh):
        return sol, lyap, np.ones(m)
    else:
        sol0 = sol
        sol0[3*n:] = np.reshape(Q[:3*n, :m], [-1])
        return sol0, lyap, lyap


def Jacobian(s, g, By, Bz):
    n = int(len(s)/3)
    sx = s[:n]
    sy = s[n:2*n]
    sz = s[2*n:3*n]
    h = g*np.mean(sz)
    J = np.zeros([3*n, 3*n])
    for i in range(n):
        for j in range(n):
            J[3*i, 3*j+2] = 4*g*sy[i]/n
            J[3*i+1, 3*j+2] = -4*g*sx[i]/n
            if (i == j):
                J[3*i, 3*j+1] = J[3*i, 3*j+1] + Bz[i]+4*h
                J[3*i, 3*j+2] = J[3*i, 3*j+2] - By[i]
                J[3*i+1, 3*j] = J[3*i+1, 3*j] - Bz[i]-4*h
                J[3*i+2, 3*j] = J[3*i+2, 3*j] + By[i]
    # return  sparse.csr_matrix(J)
    return J


def derivative(s, t, params):
    g, n, By, Bz = params

    sx = s[:n]
    sy = s[n:2*n]
    sz = s[2*n:3*n]
    h = g*np.mean(sz)

    dSx = sy*(Bz+4*h)-sz*By
    dSy = -sx*(Bz+4*h)
    dSz = sx*By

    return np.concatenate([dSx, dSy, dSz])


def initial_state(n, m=0, eta=0, h=0):
    ce = np.cos(eta)
    se = np.sin(eta)
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])
    s = np.zeros(3*n)
    for i in range(n):
        q = (i+0.5)*np.pi/(n)
        cq = np.cos(q)
        sq = np.sin(q)
        H = -2*np.array([[h+cq*(ce+se), -1j*sq*(ce-se)],
                        [1j*sq*(ce-se), -h-cq*(ce+se)]])
        e, vec = np.linalg.eigh(H)
        v = vec[:, 0]
        s[i] = np.real(np.dot(np.conj(v), np.dot(sx, v)))
        s[n+i] = np.real(np.dot(np.conj(v), np.dot(sy, v)))
        s[2*n+i] = np.real(np.dot(np.conj(v), np.dot(sz, v)))
    if (m > 0):
        M0 = np.random.rand(3*n, m)
        M0, _ = np.linalg.qr(M0)
        s = np.concatenate([s, np.reshape(M0, [-1])])
    return s


def get_transverse_magnetization(sol):
    m = len(sol)
    n = int(len(sol[0])/3)
    z = np.zeros(m)
    for i in range(m):
        z[i] = np.mean(sol[i][2*n:])
    return z

