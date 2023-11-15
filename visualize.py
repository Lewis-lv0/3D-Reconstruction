'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import submission as sub
from matplotlib import pyplot as plt


if __name__ == "__main__":
    pts1_xy = np.load("../data/templeCoords.npz")
    x1, y1 = pts1_xy["x1"], pts1_xy["y1"] # obtain points

    fm = np.load("./q2_1.npz")
    F = fm["F"] # Fundamental matrix

    # load cam intrinsics 
    K = np.load("../data/intrinsics.npz")
    K1, K2 = K["K1"], K["K2"]

    # obtain essential matrix
    E = sub.essentialMatrix(F, K1, K2)
    
    # build camera projection matrix for im1
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    C1 = K1 @ M1 
    # build cam proj matrix for im2
    m2c2 = np.load("./q3_3.npz")
    M2, C2 = m2c2["M2"], m2c2["C2"]

    # load images
    img1, img2 = plt.imread("../data/im1.png"), plt.imread("../data/im2.png")

    # find correspondences
    x2, y2 = np.zeros_like(x1), np.zeros_like(y1)
    for i in range(len(x1)):
        x2_i, y2_i = sub.epipolarCorrespondence(img1, img2, F, x1[i, 0], y1[i, 0])
        x2[i, 0], y2[i, 0] = x2_i, y2_i
    
    # make pts1 and pts2 
    pts1 = np.hstack((x1, y1))
    pts2 = np.hstack((x2, y2))

    P, _ = sub.triangulate(C1, pts1, C2, pts2)

    # visualize 3d recon result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=10, c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    











