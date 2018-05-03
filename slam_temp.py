from scipy import stats
import numpy as np


# from MATLAB gets difference between two angles
def angdiff(v1, v2):
    v1x = np.cos(v1)
    v1y = np.sin(v1)
    v2x = np.cos(v2)
    v2y = np.sin(v2)
    return np.arccos(np.matmul(np.array([v1x, v1y]), np.array([v2x, v2y]).T))



def new_landmark(P, x_t, G_p_R_hat, z, th_landmark, R):
    # Add some zeros for the new landmark covariance matrices
    first_new_row = P.shape[0] + 1  # Used for numpy
    first_row_index = first_new_row - 1  # Used for plain python
    first_new_col = P.shape[1] + 1
    first_col_index = first_new_col - 1

    new_P = np.zeros((first_new_row + 1, first_new_col + 1))
    new_P[:P.shape[0], :P.shape[1]] = np.copy(P)

    # Now we compute all the bits we need to populate the new
    # covariance matrix, P_RR, H_Li, H_R, and R
    P_RR = P[0:3, 0:3]  # Just the old robot covariance

    h_li = np.sin(th_landmark) * x_t[0] - np.cos(th_landmark) * x_t[1]
    H_Li = np.array([[1, h_li], [0, 1]])

    h_r_cos = -np.cos(th_landmark)
    h_r_sin = -np.sin(th_landmark)
    H_R = np.array([[h_r_cos, h_r_sin, 0], [0, 0, -1]])

    # First we will populate the matrices in the first row and first
    # column, since those have different shape (3x2 and 2x3
    # respectively)
    new_P[0:3, first_col_index:first_col_index + 2] = np.matmul(np.matmul(-P_RR, H_R.T), H_Li)
    new_P[first_row_index:first_row_index + 2, 0:3] = np.matmul(np.matmul(-H_Li.T, H_R), P_RR)

    # Now we step through each new landmark matrix in the first new column
    for q in range(3, first_row_index, 2):
        P_Li_R = new_P[q:q + 2, 0:3]
        new_covariance = np.matmul(np.matmul(-P_Li_R, H_R.T), H_Li)
        new_P[q:q + 2, first_col_index:first_col_index + 2] = new_covariance

    # And the same, but for the first new row
    for q in range(3, first_col_index, 2):
        P_R_Li = new_P[0:3, q:q + 2]
        new_covariance = np.matmul(np.matmul(-H_Li.T, H_R), P_R_Li)
        new_P[first_row_index:first_row_index + 2, q:q + 2] = new_covariance

    # And finally, the covariance that lay on the diagonal of the
    # matrix
    inner_product = np.matmul(np.matmul(H_R, P_RR), H_R.T) + R
    new_covariance = np.matmul(np.matmul(H_Li.T, inner_product), H_Li)

    new_P[first_row_index:first_row_index + 2, first_col_index:first_col_index + 2] = new_covariance

    # Now we add the landmark location to the state matrix
    phi = x_t[2]
    G_th = z[1] + phi
    rho = G_p_R_hat[0] * np.cos(G_th) + G_p_R_hat[1] * np.sin(G_th)
    G_d = z[0] + rho
    p_hat = np.array([[G_d[0]], [G_th[0]]])
    # print("\n\nNEW LANDMARK AT:\n{}\n\n".format(p_hat))
    new_x_t = np.concatenate((x_t, p_hat), axis=0)
    return new_x_t, new_P


def SLAM_update(x_t, Sig_t, detected_features, Sig_msmt):
    # Mahalonobis threshold
    p1 = 0.9
    p2 = 0.1
    n = 2
    epsilon_1 = 1 / stats.chi2.cdf(p1, n);
    epsilon_2 = 1 / stats.chi2.cdf(p2, n);

    # formalize notation
    R = Sig_msmt
    P = Sig_t
    G_p_R_hat = x_t[0:2]
    phi = x_t[2]
    J = np.array([[0, -1], [1, 0]])
    R_C_G = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

    # Mahalonobis Distance test
    M = detected_features.shape[1]  # Num measured landmarks

    #print("********** BEGIN LOOP **********")
    for i in range(M):
        N = (x_t.shape[0] - 3) // 2  # get number of state landmarks (assume proper state vector)
        gamma = np.full(N, np.inf)  # initialize gamma vector

        z = detected_features[:, i].reshape(2, 1)  # The current measurement (matlab has it as x,y pairs)

        # Iterate through all landmarks to find a match
        for j in range(N):
            # calculate z_hat in local frame
            p_hat = x_t[j * 2 + 3:j * 2 + 5]  # p_hat global

            z_hat_d = p_hat[0][0] - x_t[0][0] * np.cos(p_hat[1][0]) - x_t[1][0] * np.sin(p_hat[1][0])
            z_hat_th = p_hat[1][0] - x_t[2][0]
            z_hat = np.array([[z_hat_d], [z_hat_th]], dtype=np.float64)

            # caluculate residual
            r = z - z_hat
            r[1] = angdiff(z[1][0], z_hat[1][0])

            # create H matrix
            h_li = np.sin(p_hat[1]) * x_t[0] - np.cos(p_hat[1]) * x_t[1]
            H_Li = np.array([[1, h_li], [0, 1]])
            h_r_cos = -np.cos(p_hat[1])[0]
            h_r_sin = -np.sin(p_hat[1])[0]
            H_R = np.array([[h_r_cos, h_r_sin, 0], [0, 0, -1]])
            H = np.concatenate((H_R, np.zeros([2, 2 * j]), H_Li, np.zeros([2, 2 * (N - j - 1)])), axis=1)
            S = np.matmul(np.matmul(H, P), H.T) + R
            gamma[j] = np.matmul(np.matmul(r.T, np.linalg.inv(S)), r)

        gamma_min_index = -1 if N == 0 else np.argmin(gamma)
        gamma_min = np.inf if gamma_min_index == -1 else gamma[gamma_min_index]

        curr_landmark = x_t[gamma_min_index * 2 + 3:gamma_min_index * 2 + 5]
        d_landmark = curr_landmark[0][0]
        th_landmark = curr_landmark[1][0]

        # Perform Mahalanobis test
        # If there are no gamma values
        if N == 0:
            x_t, P = new_landmark(P, x_t, G_p_R_hat, z, th_landmark, R)

        else:
            if gamma_min <= epsilon_1:
                # print("\nUPDATEs")
                # print("X_T: {}".format(x_t))
                # do update on gamma_min_index
                # calculate z_hat in local frame
                p_hat = x_t[gamma_min_index * 2 + 3:gamma_min_index * 2 + 5]  # p_hat global

                z_hat_d = p_hat[0][0] - x_t[0][0] * np.cos(p_hat[1][0]) - x_t[1][0] * np.sin(p_hat[1][0])
                z_hat_th = p_hat[1][0] - x_t[2][0]
                z_hat = np.array([[z_hat_d], [z_hat_th]], dtype=np.float64)

                # caluculate residual
                r = z - z_hat
                r[1] = angdiff(z[1][0], z_hat[1][0])

                # create H matrix
                h_li = np.sin(th_landmark) * x_t[0] - np.cos(th_landmark) * x_t[1]
                H_Li = np.array([[1, h_li], [0, 1]])

                h_r_cos = -np.cos(th_landmark)
                h_r_sin = -np.sin(th_landmark)
                H_R = np.array([[h_r_cos, h_r_sin, 0], [0, 0, -1]])

                H = np.concatenate(
                    (H_R, np.zeros([2, 2 * gamma_min_index]), H_Li, np.zeros([2, 2 * (N - gamma_min_index - 1)])),
                    axis=1)
                S = np.matmul(np.matmul(H, P), H.T) + R
                K = np.matmul(np.matmul(P, H.T), np.linalg.inv(S))

                # reflect update in state and covariance
                x_t = x_t + np.matmul(K, r);
                P = P - np.matmul(np.matmul(np.matmul(np.matmul(P, H.T), np.linalg.inv(S)), H), P)
                #print("Found: ", p_hat, " As Local: ", z)
                #print("X_T: {}".format(x_t))
                # print("END UPDATEs\n")

            elif epsilon_1 < gamma_min and gamma_min < epsilon_2:
                pass
            else:
                x_t, P = new_landmark(P, x_t, G_p_R_hat, z, th_landmark, R)

    return x_t, P
