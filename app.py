from facenet_pytorch import MTCNN
import cv2
import matplotlib.pyplot as plt
mtcnn = MTCNN()
image = cv2.imread('image3.jpg') 
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes, _ = mtcnn.detect(image_rgb)
if boxes is not None:
    for box in boxes:
        x, y, w, h = [int(coord) for coord in box]
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
