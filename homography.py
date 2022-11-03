from scipy.spatial import distance as dist
import cv2
import numpy as np
import math

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]

	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]

	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.float32([tl, tr, bl, br])

def find_document_vertices(contours):
    document_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            document_contour = contour

    document_contour = cv2.approxPolyDP(document_contour, 0.02 * cv2.arcLength(document_contour,True),True)

    vertices = document_contour.reshape((4,2))
    return np.float32(order_points(vertices))

img = cv2.imread("img.jpg")

""" Detect edges """

grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
binary_img = cv2.Canny(grayscale_img,100,200)

contours, hierarchy = cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

vertices = find_document_vertices(contours)

width = int(
    math.sqrt(
        (vertices[1][0] - vertices[0][0]) ** 2 + (vertices[0][1] - vertices[1][1]) ** 2
    )
)
height = int(
    math.sqrt(
        (vertices[0][0] - vertices[2][0]) ** 2 + (vertices[0][1] - vertices[2][1]) ** 2
    )
)

final_vertices = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

transformation_matrix = cv2.getPerspectiveTransform(vertices, final_vertices)

result_img = cv2.warpPerspective(img, transformation_matrix, (width, height))

for vertex in vertices:
    cv2.circle(img, np.int32(vertex), 5, (0, 0, 255), cv2.FILLED)

cv2.imshow("Document", img)
cv2.imshow("Bird eye document", result_img)
cv2.waitKey()
