from one_a import *
import numpy as np
import cv2 as cv
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from itertools import combinations

_kp_img = defaultdict(np.ndarray)
_kp_des = defaultdict(np.ndarray)
sift = cv.SIFT_create()


def intersect(C, d):
    """
    Ray-tracing algorithm, i.e. find the intersection of a ray X = C + t*d at the teabox
    Details at https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    Args:
        C: the starting point of the ray
        d: the direction of the ray

    Returns:
        tmin: the step where the ray intersect the teabox. If no intersection found, return False
    """
    bound_min = corner_world[7]
    bound_max = corner_world[1]

    tmin = (bound_min[0] - C[0]) / d[0]
    tmax = (bound_max[0] - C[0]) / d[0]
    if tmin > tmax:
        tmin, tmax = tmax, tmin

    tymin = (bound_min[1] - C[1]) / d[1]
    tymax = (bound_max[1] - C[1]) / d[1]
    if tymin > tymax:
        tymin, tymax = tymax, tymin

    if (tmin > tymax) or (tymin > tmax):
        return False

    if tymin > tmin:
        tmin = tymin
    if tymax < tmax:
        tmax = tymax

    tzmin = (bound_min[2] - C[2]) / d[2]
    tzmax = (bound_max[2] - C[2]) / d[2]
    if tzmin > tzmax:
        tzmin, tzmax = tzmax, tzmin

    if (tmin > tzmax) or (tzmin > tmax):
        return False

    if tzmin > tmin:
        tmin = tzmin

    if tzmax < tmax:
        tmax = tzmax

    return tmin


def valid_sift(R, trans, image):
    """
    Filter out the sift keypoints that do not lie on the teabox surface

    Args:
        R: rotation matrix, from world to camera
        trans: transition from world to camera
        image: return of cv.imread()

    Returns:
        sift_img: the image coordinates of sift keypoints
        sift_des: the descriptor of sift keypoints
        sift_world: the 3d world coordinates of keypoints
        sift_obj: the return keypoint object of the detectAndCompute() function
    """

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kp_obj, kp_des_all = sift.detectAndCompute(gray, None)
    kp_img_all = np.array([point.pt for point in kp_obj])
    C = - R.T @ trans.squeeze()

    valid_idx = []
    kp_world_ls = []
    for j in range(kp_img_all.shape[0]):
        kp = np.hstack([kp_img_all[j], 1])
        d = R.T @ np.linalg.inv(K) @ kp
        tmin = intersect(C, d)
        if tmin:
            kp_world_ls.append((C + tmin * d).tolist())
            valid_idx.append(j)  # only keep the keypoints that intersect with the teabox
    sift_world = np.array(kp_world_ls)
    sift_des = kp_des_all[valid_idx]
    sift_img = kp_img_all[valid_idx]
    sift_img = sift_img.astype('int16')
    sift_obj = [kp_obj[i] for i in valid_idx]
    return sift_img, sift_des, sift_world, sift_obj


if __name__ == '__main__':
    kp_img = defaultdict(np.ndarray)
    kp_world = defaultdict(np.ndarray)
    kp_des = defaultdict(np.ndarray)
    for i in range(8):
        img = cv.imread(f'../data_task1/init_texture/DSC_97{str(i + 43)}.jpg')
        sift_img, sift_des, sift_world, _ = valid_sift(R_trajectory[i], T_trajectory[i], img)
        kp_world[str(i)] = sift_world
        kp_des[str(i)] = sift_des
        kp_img[str(i)] = sift_img

    keypoints_world = np.vstack(list(kp_world.values()))
    keypoints_des = np.vstack(list(kp_des.values()))
    with open('keypoints_world', 'wb') as f:
        pickle.dump(keypoints_world, f)
    with open('keypoints_des', 'wb') as f:
        pickle.dump(keypoints_des, f)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown']

    # draw the teabox
    for s, e in combinations(corner_world, 2):
        if np.sum(s == e) == 2:
            ax.plot3D(*zip(s, e), color="b")

    # draw the camera poses
    for i in range(len(R_trajectory)):
        H = np.zeros((4, 4))
        r = R_trajectory[i]
        t = T_trajectory[i].squeeze()
        H[:3, :3] = r.T
        H[:3, 3] = - r.T @ t
        H[3, 3] = 1
        p0 = H @ np.array([0, 0, 0, 1])
        px = H @ np.array([0.05, 0, 0, 1])
        py = H @ np.array([0, 0.05, 0, 1])
        pz = H @ np.array([0, 0, 0.1, 1])
        ax.plot([p0[0], px[0]], [p0[1], px[1]], [p0[2], px[2]], color[i])
        ax.plot([p0[0], py[0]], [p0[1], py[1]], [p0[2], py[2]], color[i])
        ax.plot([p0[0], pz[0]], [p0[1], pz[1]], [p0[2], pz[2]], color[i])

    # draw the sift keypoints
    for i in range(8):
        kps = kp_world[str(i)]
        ax.scatter(kps[:, 0], kps[:, 1], kps[:, 2], color=color[i])

    plt.show()
