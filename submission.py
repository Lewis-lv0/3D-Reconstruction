"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
np.random.seed(123)
import os
from util import refineF, _singularize
from matplotlib import pyplot as plt
import sys
from scipy.optimize import leastsq
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    T = np.array([[1 / M, 0, 0], [0, 1 / M, 0], [0, 0, 1]])
    pts1, pts2 = pts1 / M, pts2 / M
    num_points = pts1.shape[0]
    xl, yl, xr, yr = pts1[:, 0].reshape(-1, 1), pts1[:, 1].reshape(-1, 1), pts2[:, 0].reshape(-1, 1), pts2[:, 1].reshape(-1, 1)
    u = np.hstack((xr*xl, xr*yl, xr, yr*xl, yr*yl, yr, xl, yl, np.ones((num_points, 1))))
    _, _, Vt = np.linalg.svd(u)
    # eigenvector of least engenvalue of V
    f = Vt[-1, :]
    f = refineF(f, pts1, pts2)

    F = T.T @ f @ T
    # np.savez('q2_1.npz', F=F, M=M)
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T @ F @ K1
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    C11, C12, C13 = C1[0], C1[1], C1[2]
    C21, C22, C23 = C2[0], C2[1], C2[2]
    num_points = pts1.shape[0]

    # 3d reconstructed points (non-homo); each row is a 3d point
    P = np.zeros((num_points, 3)) 
    P_homo = np.zeros((num_points, 4)) 
    for i in range(num_points):
        A1 = C13 * pts1[i, 0] - C11
        A2 = C13 * pts1[i, 1] - C12
        A3 = C23 * pts2[i, 0] - C21
        A4 = C23 * pts2[i, 1] - C22
        A = np.vstack((A1, A2, A3, A4))
        assert A.shape == (4, 4)
        _, _, Vt = np.linalg.svd(A)

        p = Vt[-1, :]
        P_homo[i, :] = p[:]
        p = p / p[-1]
        P[i,:] = p[:3] # assign p to P[i]

    # print(f"P: {P}")
    # reprojection
    pts1_reproj = C1 @ P_homo.T
    pts2_reproj = C2 @ P_homo.T
    # homo -> non-homo
    pts1_reproj = (pts1_reproj[:2] / pts1_reproj[2]).T
    pts2_reproj = (pts2_reproj[:2] / pts2_reproj[2]).T
    # compute err
    err = np.sum((pts1 - pts1_reproj)**2 + (pts2 - pts2_reproj)**2)
    # print(f"Error: {err}")
    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # params
    win_size_half = 6
    search_area_half = 50 # search around 50 pixels

    pt1 = np.array([x1, y1, 1]).reshape(-1, 1)
    epip_line = F @ pt1
    line_y = np.arange(y1-search_area_half, y1+search_area_half+1, step=1)
    line_x = (-epip_line[1] * line_y - epip_line[2]) / epip_line[0]
    H, W, _  = im2.shape
    patch_im1 = im1[int(y1 - win_size_half) : int(y1 + win_size_half), int(x1 - win_size_half) : int(x1 + win_size_half), :]
    
    # Create a Gaussian kernel
    sigma = 1
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - win_size_half) ** 2 + (y - win_size_half) ** 2) / (2 * sigma ** 2)),
        shape=(2 * win_size_half, 2 * win_size_half)
    ).reshape(win_size_half*2, win_size_half*2, 1)
    
    min_diff = np.inf
    for i in range(search_area_half*2):
        x2, y2 = int(line_x[i]), int(line_y[i])
        if x2 >= win_size_half and x2+win_size_half < W and y2 >= win_size_half and y2+win_size_half < H:
            patch_im2 = im2[int(y2 - win_size_half) : int(y2 + win_size_half), int(x2 - win_size_half) : int(x2 + win_size_half), :]

            diff = np.linalg.norm((patch_im1 - patch_im2) * kernel)
            if diff < min_diff:
                min_diff = diff
                x2_out, y2_out = x2, y2
    return x2_out, y2_out

     

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=1):
    # Replace pass by your implementation
    max_inliers = -1
    N = pts1.shape[0]
    # F_out = np.zeros(shape=(3, 3)) # shape: (3, 3)
    # inliers_out = np.full(shape=(N, 1), fill_value=False, dtype=bool) # shape: (N, 1)
    pts1_homo = np.vstack((pts1.T, np.ones((1, N)))) # shape: (3, N)
    pts2_homo = np.vstack((pts2.T, np.ones((1, N)))) # shape: (3, N)
    for i in range(nIters):
        if i % 100 == 0:
            print(f"Iteration {i}")
        random_idx = np.random.choice(N, size=8, replace=False) # 8 different points
        pts1_subset = pts1[random_idx, :]
        pts2_subset = pts2[random_idx, :]
        F = eightpoint(pts1_subset, pts2_subset, M)

        # compute distance from pts2 to the predicted epipolar line
        epip_line = F @ pts1_homo # shape: (3, N)
        inliers_count = 0
        is_inliers = np.full((N, 1), False, dtype=bool)
        for j in range(N):
            # shape of pts2_homo.T[j, :] -> (3,)
            dist = np.abs(np.dot(pts2_homo.T[j, :], epip_line[:, j])) / np.linalg.norm(epip_line[:2, j])
            if dist < tol:
                inliers_count += 1
                is_inliers[j, 0] = True

        if inliers_count > max_inliers:
            max_inliers = inliers_count
            inliers_out = is_inliers
    
    """
    Refine estimate using all the inliers
    """
    all_inliers_pts1 = pts1[inliers_out.squeeze(), :]
    all_inliers_pts2 = pts2[inliers_out.squeeze(), :]
    F = eightpoint(all_inliers_pts1, all_inliers_pts2, M)
    epip_line = F @ pts1_homo # shape: (3, N)
    inliers_count = 0
    is_inliers = np.full((N, 1), False, dtype=bool)
    for j in range(N):
        # shape of pts2_homo.T[j, :] -> (3,)
        dist = np.abs(np.dot(pts2_homo.T[j, :], epip_line[:, j])) / np.linalg.norm(epip_line[:2, j])
        if dist < tol:
            inliers_count += 1
            is_inliers[j, 0] = True
    print(f"Number of inliers: {inliers_count}") # 106
    return F, is_inliers

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.identity(3)
    else:
        r_unit = (r / theta).squeeze()
        K = np.array([[0, -r_unit[2], r_unit[1]],
                      [r_unit[2], 0, -r_unit[0]],
                      [-r_unit[1], r_unit[0], 0]])
        R = np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    tr = np.trace(R)
    if tr == 3:
        return np.zeros(3)  # No rotation, return zero vector
    else:
        theta = np.arccos((np.trace(R) - 1) / 2)
        r = (1 / (2 * np.sin(theta))) * np.array([R[2, 1] - R[1, 2],
                                                  R[0, 2] - R[2, 0],
                                                  R[1, 0] - R[0, 1]])
        return r * theta

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    N = p1.shape[0]
    assert p1.shape == p2.shape
    P, r2, t2 = x[0:3*N].reshape(N, 3), x[N*3:N*3+3].reshape(3, 1), x[N*3+3:].reshape(3, 1)
    R2 = rodrigues(r2) # r2 -> R2 (3, 3)
    M2 = np.hstack((R2, t2))

    # projection matrix: p = CP
    C1 = K1 @ M1
    C2 = K2 @ M2

    # projection
    P_homo = np.hstack((P, np.ones((N, 1)))).T # (4, N)
    p1_hat_homo = C1 @ P_homo # (3, N)
    p2_hat_homo = C2 @ P_homo

    # homo -> non-homo
    p1_hat = p1_hat_homo[:2,:] / p1_hat_homo[2,:] # (2, N)
    p2_hat = p2_hat_homo[:2,:] / p2_hat_homo[2,:] 
    residuals = np.concatenate([(p1 - p1_hat.T).reshape([-1]), (p2 - p2_hat.T).reshape([-1])])
    assert residuals.shape == (4*N,)
    return residuals


'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    R2_init, t2_init = M2_init[:, :3], M2_init[:, 3]
    r2_init = invRodrigues(R2_init)
    x_init = np.concatenate([P_init.flatten(), r2_init.flatten(), t2_init.flatten()])
    assert len(x_init) == 3*p1.shape[0] + 3 + 3

    def wrapper(x_init, K1, M1, p1, K2, p2):
        return rodriguesResidual(K1, M1, p1, K2, p2, x_init)
    
    result = leastsq(wrapper, x_init, args=(K1, M1, p1, K2, p2))
    x_optimized = result[0]

    N = p1.shape[0]
    P, r2, t2 = x_optimized[0:3*N].reshape(N, 3), x_optimized[3*N:3*N+3].reshape(3, 1), x_optimized[3*N+3:].reshape(3, 1)
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))
    return M2, P


