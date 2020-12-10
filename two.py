import numpy as np
import cv2 as cv
import pickle
from one_a import K, dist_coeffs, corner_world, imshow

# all the keypoints from task 1b
keypoints_des = pickle.load(open('keypoints_des', 'rb'))
keypoints_des = keypoints_des.astype('float32')
keypoints_world = pickle.load(open('keypoints_world', 'rb'))


def draw_cube(corner_w, rot, trans, image):
    """
    draw the teabox frame on the image with known camera pose
    Args:
        corner_w: the world coordinates of the teabox's corners
        rot: the rotation from the world to the camera coordinate
        trans: the translation from the world to the camera coordinate
        image: the image to draw the teabox on
    """""
    corner_w = corner_w.astype('float64')
    corner_cur, _ = cv.projectPoints(corner_w, rot, trans, K, dist_coeffs)
    corner_cur = corner_cur.squeeze(1)
    corner_cur = corner_cur.astype('int32')

    draw_line = [[0, 1], [0, 3], [1, 2], [2, 3],
                 [4, 5], [4, 7], [5, 6], [6, 7],
                 [1, 5], [2, 6], [0, 4], [3, 7]]

    for idx in draw_line:
        cv.line(image, tuple(corner_cur[idx[0]]), tuple(corner_cur[idx[1]]), (0, 255, 0), thickness=3,
                lineType=cv.LINE_AA)


if __name__ == '__main__':
    sift = cv.SIFT_create()
    # For those images, use brute force matcher to match the keypoints
    for image_name in [51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 69, 73]:

        img = cv.imread(f'../data_task2/detection/DSC_97{str(image_name)}.jpg')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        obj_a, des_a = sift.detectAndCompute(gray, None)
        uv_a = np.array([point.pt for point in obj_a])
        # Brute force matcher
        matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        matches = matcher.match(des_a, keypoints_des)

        train_idx = [m.trainIdx for m in matches]
        query_idx = [m.queryIdx for m in matches]

        w_m0 = keypoints_world[train_idx]
        uv_m = uv_a[query_idx]
        try:

            # different PnPRansac reprojection error thershold for different images
            if image_name == 64:
                reprojection_error = 15
            elif image_name == 69:
                reprojection_error = 20
            else:
                reprojection_error = 8
            reval, R, T, inliers = cv.solvePnPRansac(w_m0, uv_m, K, dist_coeffs,
                                                     reprojectionError=reprojection_error)
        except Exception as e:
            print(image_name)
            print(e)
            continue
        if reval:

            # show if the matched SIFT keypoints lie on the box
            query_matchedIdx = [query_idx[i] for i in inliers.squeeze()]
            obj_m = [obj_a[i] for i in query_matchedIdx]
            img = cv.drawKeypoints(img, obj_m, img)

            draw_cube(corner_world, R, T, img)
            imshow(str(image_name), img)
            # cv.imwrite(f'../data_task2/detection/DSC_97{str(image_name)}_bbx.jpg', img)
        else:
            print('ransac failed ', image_name)

    for image_name in [59, 65, 66, 67, 70, 74]:
        img = cv.imread(f'../data_task2/detection/DSC_97{str(image_name)}.jpg')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        obj_a, des_a = sift.detectAndCompute(gray, None)
        uv_a = np.array([point.pt for point in obj_a])
        if image_name == 59:

            # FLANN matcher
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary

            flann = cv.FlannBasedMatcher(index_params, search_params)

            matches_all = flann.knnMatch(des_a, keypoints_des, k=2)
            matches = []
            for (m, n) in matches_all:
                if m.distance < 0.6 * n.distance:
                    matches.append(m)
        else:
            # Bruteforce matcher with knn match
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

        try:

            reval, R, T, inliers = cv.solvePnPRansac(w_m0, uv_m, K, dist_coeffs,
                                                     reprojectionError=(8 if image_name == 59 else 13))
        except Exception as e:
            print(image_name)
            print(e)
            continue
        if reval:
            # show if the matched SIFT keypoints lie on the box
            query_matchedIdx = [query_idx[i] for i in inliers.squeeze()]
            obj_m = [obj_a[i] for i in query_matchedIdx]
            img = cv.drawKeypoints(img, obj_m, img)

            draw_cube(corner_world, R, T, img)
            imshow(str(image_name), img)
            # cv.imwrite(f'../data_task2/detection/DSC_97{str(image_name)}_bbx.jpg', img)
        else:
            print('ransac failed ', image_name)
