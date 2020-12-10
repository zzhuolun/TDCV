import sys
from collections import defaultdict
from itertools import combinations
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')

# manually marked image coordinates of corners
# format: corner_img['i']['j']: i is the ith image in the folder datatask1/init_texture,
#                               j is the 2d image coordinates of the jth corner
corner_img = defaultdict(dict)
corner_img['0']['0'] = [1373, 1021]
corner_img['0']['1'] = [2237, 1005]
corner_img['0']['2'] = [2311, 1114]
corner_img['0']['3'] = [1345, 1134]
corner_img['0']['6'] = [2280, 1590]
corner_img['0']['7'] = [1374, 1615]

corner_img['1']['0'] = [1655, 932]
corner_img['1']['1'] = [2204, 1151]
corner_img['1']['2'] = [1923, 1232]
corner_img['1']['3'] = [1400, 984]
corner_img['1']['5'] = [2175, 1619]
corner_img['1']['6'] = [1913, 1734]
corner_img['1']['7'] = [1419, 1406]

corner_img['2']['0'] = [1884, 856]
corner_img['2']['1'] = [1936, 1150]
corner_img['2']['2'] = [1536, 1145]
corner_img['2']['3'] = [1584, 853]
corner_img['2']['5'] = [1922, 1649]
corner_img['2']['6'] = [1548, 1658]

corner_img['3']['0'] = [2315, 975]
corner_img['3']['1'] = [1708, 1198]
corner_img['3']['2'] = [1461, 1096]
corner_img['3']['3'] = [2074, 908]
corner_img['3']['4'] = [2287, 1395]
corner_img['3']['5'] = [1713, 1686]
corner_img['3']['6'] = [1476, 1566]

corner_img['4']['0'] = [2292, 1126]
corner_img['4']['1'] = [1308, 1119]
corner_img['4']['2'] = [1358, 995]
corner_img['4']['3'] = [2241, 1005]
corner_img['4']['4'] = [2259, 1602]
corner_img['4']['5'] = [1337, 1599]

corner_img['5']['0'] = [1746, 1178]
corner_img['5']['1'] = [1320, 937]
corner_img['5']['2'] = [1593, 882]
corner_img['5']['3'] = [2056, 1101]
corner_img['5']['4'] = [1757, 1669]
corner_img['5']['5'] = [1348, 1343]
corner_img['5']['7'] = [2044, 1573]

corner_img['6']['0'] = [1597, 1186]
corner_img['6']['1'] = [1647, 908]
corner_img['6']['2'] = [1938, 905]
corner_img['6']['3'] = [1983, 1187]
corner_img['6']['4'] = [1611, 1678]
corner_img['6']['7'] = [1971, 1679]

corner_img['7']['0'] = [1458, 1141]
corner_img['7']['1'] = [2050, 967]
corner_img['7']['2'] = [2296, 1026]
corner_img['7']['3'] = [1703, 1235]
corner_img['7']['4'] = [1468, 1607]
corner_img['7']['6'] = [2268, 1448]
corner_img['7']['7'] = [1702, 1729]

teabox = np.array([0, 0.063, 0.093, -0.666667, 0.333333, 0.666667,
                   0.165, 0.063, 0.093, 0.666667, 0.666667, 0.333333,
                   0.165, 0, 0.093, 0.333333, -0.666667, 0.666667,
                   0, 0, 0.093, -0.57735, -0.57735, 0.57735,
                   0, 0.063, 0, -0.333333, 0.666667, -0.666667,
                   0.165, 0.063, 0, 0.57735, 0.57735, -0.57735,
                   0.165, 0, 0, 0.666667, -0.333333, -0.666667,
                   0, 0, 0, -0.666667, -0.666667, -0.333333], dtype='double')
teabox = teabox.reshape(8, 6)
corner_world = teabox[:, :3]  # world coordinates of the 8 vertices of the teabox

# Camera Intrinsics
K = np.array([
    [2960.37845, 0, 1841.68855],
    [0, 2960.37845, 1235.23369],
    [0, 0, 1]
], dtype='double')

dist_coeffs = np.zeros((4, 1))


def imshow(window_name, image):
    """
    show the image with given window name
    """
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 920, 614)
    cv.imshow(window_name, image)
    cv.waitKey(0)


def draw_trajectory(R, T):
    """
    draw the camera trajectory given a list of camera poses
    Note:
        R and T are the transition from world coordinates to camera coordinates

    Args:
        R: a list of camera rotations, each element is in SO(3)
        T: a list of camera translations.

    Returns:

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # draw the teabox
    for s, e in combinations(corner_world, 2):
        if np.sum(s == e) == 2:
            ax.plot3D(*zip(s, e), color="b")
    scale = 0.05

    # draw the camera poses
    for i in range(len(R)):
        H = np.zeros((4, 4))
        r = R[i]
        t = T[i].squeeze()
        H[:3, :3] = r.T
        H[:3, 3] = - r.T @ t
        H[3, 3] = 1
        p0 = H @ np.array([0, 0, 0, 1])
        px = H @ np.array([.05, 0, 0, 1])
        py = H @ np.array([0, .05, 0, 1])
        pz = H @ np.array([0, 0, .1, 1])
        # draw the direction of x axis of the camera coordinate
        ax.plot([p0[0], px[0]], [p0[1], px[1]], [p0[2], px[2]], 'r')
        # draw the direction of y axis of the camera coordinate
        ax.plot([p0[0], py[0]], [p0[1], py[1]], [p0[2], py[2]], 'g')
        # draw the direction of z axis of the camera coordinate
        ax.plot([p0[0], pz[0]], [p0[1], pz[1]], [p0[2], pz[2]], 'b')
    plt.show()


R_trajectory = []
T_trajectory = []
for img, corners in corner_img.items():
    image_points = np.asarray(list(corners.values()), dtype='double')
    image_points = np.ascontiguousarray(image_points[:, :]).reshape((-1, 1, 2))
    idx = [int(k) for k in corners.keys()]
    object_points = corner_world[idx]
    # estimate the pose of camera with PnP using all the visible corners from current image
    retval, rvec, tvec = cv.solvePnP(object_points, image_points, K, dist_coeffs)
    if retval:
        R_trajectory.append(cv.Rodrigues(rvec)[0])
        T_trajectory.append(tvec)

if __name__ == '__main__':
    draw_trajectory(R_trajectory, T_trajectory)
