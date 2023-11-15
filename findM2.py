'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import submission as sub
from helper import camera2
import os
if __name__ == "__main__":
    # load correspondences
    points = np.load("../data/some_corresp.npz")
    pts1, pts2 = points["pts1"], points["pts2"]
    # load intrinsics 
    K = np.load("../data/intrinsics.npz")
    K1, K2 = K["K1"], K["K2"]
    # load M and F -> fundamental matrix
    MF = np.load("q2_1.npz")
    _, F = MF["M"], MF["F"]
    # obtain essential matrix
    E = sub.essentialMatrix(F, K1, K2)
    if not os.path.exists("q3_1.npz"):
        np.savez("q3_1.npz", E)
        print("Saved")
    M2s = camera2(E)

    # pick the correct R, t
    """
    the sign of the z element of a reconstructed 3d point 
    -> whether the point is in front of the camera
    """
    C1 = np.hstack((K1, np.zeros((3, 1))))
    assert C1.shape == (3, 4)
    num_greater_zero = []
    for i in range(4):
        M2_candidate = M2s[:, :, i]
        C2 = K2 @ M2_candidate # proj matrix for camera 2
        assert C2.shape == (3, 4)
        P, _ = sub.triangulate(C1, pts1, C2, pts2)
        p_z = P[:, -1]
        num_greater_zero.append(np.sum(p_z > 0))
    
    # obtain the best M2
    M2 = M2s[:, :, np.argmax(num_greater_zero)]
    C2 = K2 @ M2
    P, err = sub.triangulate(C1, pts1, C2, pts2)
    # np.savez("q3_3.npz", M2=M2, C2=C2, P=P)




