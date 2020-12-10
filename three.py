import cv2 as cv
import numpy as np
from one_a import K, dist_coeffs, corner_world, draw_trajectory, imshow
from one_b import valid_sift
from two import draw_cube, keypoints_des, keypoints_world

sift = cv.SIFT_create()


def img_name(i, bbx=False):
    """
    auxilary function to return the path of image i
    Args:
        i: the index of images in the folder data_task3/tracking
        bbx: if contain the bbx suffix in the image name

    Returns:
        A string of the path of the image i
    """
    if 6 <= i <= 30:
        if bbx:
            return f'../data_task3/tracking/color_0000{i:02d}_bbx.jpg'
        else:
            return f'../data_task3/tracking/color_0000{i:02d}.jpg'
    else:
        raise ValueError('image index not existing')


def initialize(show=False):
    """
    Initialize the tracking using PnPRansac from task 2

    Args:
        show: if draw the camera pose and show

    Returns:
        rot : (3,1) np.ndarray, the estimated of the first image in the data_task3/tracking folder
        trans: (3,1) np.ndarray, the estimated of the first image in the data_task3/tracking folder
    """
    img = cv.imread(img_name(6))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    obj_a, des_a = sift.detectAndCompute(gray, None)
    uv_a = np.array([point.pt for point in obj_a])

    matcher = cv.BFMatcher(cv.NORM_L1)
    matches_all = matcher.knnMatch(des_a, keypoints_des, k=2)

    matches = []
    for m, n in matches_all:
        if m.distance < 0.75 * n.distance:
            matches.append(m)

    train_idx = [m.trainIdx for m in matches]
    query_idx = [m.queryIdx for m in matches]

    w_m0 = keypoints_world[train_idx]
    uv_m = uv_a[query_idx]

    reval, rot, trans, inliers = cv.solvePnPRansac(w_m0, uv_m, K, dist_coeffs,
                                                   reprojectionError=20)
    if show:
        # show if the matched SIFT keypoints lie on the box
        query_matchedIdx = [query_idx[i] for i in inliers.squeeze()]
        obj_m = [obj_a[i] for i in query_matchedIdx]
        gray = cv.drawKeypoints(gray, obj_m, gray)

        draw_cube(corner_world, rot, trans, gray)
        imshow('initialize', gray)

    return rot, trans


def energy_function(error, c=4.685):
    """
    The tukey energy function
    Args:
        error (ndarray: Nx1): the input of the tukey function
        c: the rescaling constant c

    Returns:
        E (float): the output of the energy function
    """
    E = c ** 2 / 6 * (1 - (1 - (error / c) ** 2) ** 3)
    E[np.abs(error) > c] = c ** 2 / 6
    return np.sum(E)


def tukey_weight(error, c=4.685):
    """
    The tukey weight function
    Args:
        error (ndarray): the input of the tukey function
        c: the rescaling constant c

    Returns:
        W (ndarray: 2Nx2N) the diagonalized weight matrix
    """
    w = (1 - (error / c) ** 2) ** 2
    w[np.abs(error) >= c] = 0
    W = np.diag(w)
    return W


def residual(X_w, uv, theta):
    """
    Computes the reprojection error given 3d points and the 2d keypoints
    Args:
        X_w (ndarray: Nx3): 3d coordinates of the valid sift keypoints in world coordinate from the previous frame
        uv (ndarray: Nx2): the matched 2d image coordinates of sift keypoints from current frame
        theta (ndarray: 6x1): camera pose to project X_w

    Returns:
        error (ndarray: 2Nx1): the computed reprojection error
        J (ndarray: 2Nx6) : the Jacobian matrix w.r.t the camera pose
    """
    x_proj, J = cv.projectPoints(X_w, theta[:3], theta[3:], K, dist_coeffs)
    x_proj = x_proj.squeeze(1)
    error = x_proj - uv
    error = error.reshape(-1)
    J = J[:, :6]
    return error, J


def IRLS(X_w, uv, rot_init, trans_init, T, tau=1e-4):
    """
    The implementation of iterative re-weighted least square algorithm
    Args:
        X_w (ndarray: Nx3): 3d coordinates of the valid sift keypoints in world coordinate from the previous frame
        uv (ndarray: Nx2): the matched 2d image coordinates of sift keypoints from current frame
        rot_init (ndarray: 3x1) : the initialization of rotation vector
        trans_init (ndarray: 3x1) : the initialization of translation vector
        T (int): iteration steps
        tau (float): the tolerance of iteration

    Returns:
        rot_irls (ndarray: 3x1) : ouput camera pose of IRLS
        trans_irls (ndarray: 3x1) : ouput camera pose of IRLS
    """
    theta = np.hstack([rot_init.reshape(-1), trans_init.reshape(-1)])
    lmd = 0.001
    c = 4.685
    mu = tau + 1

    cnt = 0
    for t in range(T):
        if mu < tau:
            break
        error, J = residual(X_w, uv, theta)

        sigma = np.median(np.abs(error)) / 0.6745
        J = J / sigma
        E = energy_function(error)
        W = tukey_weight(error / sigma)
        delta = -np.linalg.inv(J.T @ W @ J + lmd * np.eye(6)) @ J.T @ W @ (error / sigma)

        error_, _ = residual(X_w, uv, theta + delta)
        E_ = energy_function(error_)

        if E_ > E:
            lmd *= 10
            cnt += 1
        else:
            lmd /= 10
            theta += delta
        if lmd > 1e4:
            lmd = 0.001
        mu = np.linalg.norm(delta)
    rot_irls = np.expand_dims(theta[:3], 1)
    trans_irls = np.expand_dims(theta[3:], 1)
    return rot_irls, trans_irls


def tracking(rot_prev, trans_prev, img1, img2, idx, show=True):
    """
    estimate the pose of img2 given img1 and pose of img1, IRLS followed by keypoints matching
    Args:
        rot_prev (ndarray: 3x1) : the rotation of camera in img1
        trans_prev (ndarray: 3x1) : the translation of camera in img1
        img1 (ndarray) : the previous image with known camera pose
        img2 (ndarray) : the current image of which the pose to be estimated
        idx (int) : the index of images in data_task3/tracking folder

    Returns:
        rot_cur (ndarray: 3x1) : estimated camera pose of current image
        trans_cur (ndarray: 3x1) : estimated camera pose of current image
    """
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    uv_s1, des_s1, w_s1, obj_s1 = valid_sift(cv.Rodrigues(rot_prev)[0], trans_prev, img1)

    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    obj_a2, des_a2 = sift.detectAndCompute(gray2, None)
    uv_a2 = np.array([point.pt for point in obj_a2])

    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = matcher.match(des_a2, des_s1)

    train_idx = [m.trainIdx for m in matches]
    query_idx = [m.queryIdx for m in matches]

    w_m1 = w_s1[train_idx]
    uv_m2 = uv_a2[query_idx]

    rot_cur, trans_cur = IRLS(w_m1, uv_m2, rot_prev, trans_prev, 1000, 0)

    draw_cube(corner_world, rot_cur, trans_cur, img2)
    if show:
        imshow('match', cv.drawMatches(img2, obj_a2, img1, obj_s1, matches, None,
                                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
        imshow(str(idx), img2)
    else:
        cv.imwrite(img_name(idx, True), img2)
    return rot_cur, trans_cur


# initialize the camera pose of the first frame
rot, trans = initialize(False)

R_trajectory = []
T_trajectory = []
R_trajectory.append(cv.Rodrigues(rot)[0])
T_trajectory.append(trans)
for i in range(7, 31):
    img_prev = cv.imread(img_name(i - 1))
    img_cur = cv.imread(img_name(i))
    print(i - 1)
    print(rot)
    print(trans)

    # track the pose of current camera
    rot, trans = tracking(rot, trans, img_prev, img_cur, i)
    R_trajectory.append(cv.Rodrigues(rot)[0])
    T_trajectory.append(trans)

draw_trajectory(R_trajectory, T_trajectory)
