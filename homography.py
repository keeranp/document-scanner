from turtle import width
import cv2
import numpy as np
import math

img = cv2.imread("img.jpg")
# Find these points automatically
vertices = [[364, 92], [750, 113], [180, 612], [752, 690]]

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

final_vertices = [[0, 0], [width, 0], [0, height], [width, height]]

transformation_matrix = cv2.getPerspectiveTransform(
    np.float32(vertices), np.float32(final_vertices)
)

result_img = cv2.warpPerspective(img, transformation_matrix, (width, height))

for vertex in vertices:
    cv2.circle(img, vertex, 5, (0, 0, 255), cv2.FILLED)

cv2.imshow("Document", img)
cv2.imshow("Bird eye document", result_img)
cv2.waitKey()
