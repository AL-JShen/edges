import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filepath = ''
image = mpimg.imread(filepath)
greyed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

sobel_y = np.array([[-2, -4, -2],
                    [ 0,  0,  0],
                    [ 2,  4,  2]])

sobel_x = np.array([[-2, 0, 2],
                    [-4, 0, 4],
                    [-2, 0, 2]])

wiki_gaussian = np.matrix('0.00000067 0.00002292 0.00019117 0.00038771 0.00019117 0.00002292 0.00000067 0.00002292 0.00078633 0.00655965 0.01330373 0.00655965 0.00078633 0.00002292 0.00019117 0.00655965 0.05472157 0.11098164 0.05472157 0.00655965 0.00019117 0.00038771 0.01330373 0.11098164 0.22508352 0.11098164 0.01330373 0.00038771 0.00019117 0.00655965 0.05472157 0.11098164 0.05472157 0.00655965 0.00019117 0.00002292 0.00078633 0.00655965 0.01330373 0.00655965 0.00078633 0.00002292 0.00000067 0.00002292 0.00019117 0.00038771 0.00019117 0.00002292 0.00000067')
wiki_gaussian = np.resize(wiki_gaussian, (7,7))

mean = (1/5)**2 * np.ones((5, 5))

blurred = cv2.filter2D(greyed, -1, wiki_gaussian)
blurred = cv2.filter2D(greyed, -1, mean)

x_edges = cv2.filter2D(blurred, -1, sobel_x)
y_edges = cv2.filter2D(blurred, -1, sobel_y)
final_image = np.int8(np.array([np.sqrt(x_edges[i] ** 2 + y_edges[i] ** 2) for i in range(len(x_edges))]))

plt.imshow(cv2.bitwise_not(final_image), cmap='gray')
plt.show()
