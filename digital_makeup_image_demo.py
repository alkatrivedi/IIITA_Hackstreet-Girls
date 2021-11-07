# pylint: skip-file
import cv2
import face_recognition
from PIL import Image, ImageDraw
import numpy

jewel_img = cv2.imread("necklace.png")
frame = cv2.imread('face.jpg')
frame = cv2.resize(frame,(432, 576))

# Returns a list of face landmarks present on frame
face_landmarks_list = face_recognition.face_landmarks(frame)
# For demo images only one person is present in image 
face_landmarks = face_landmarks_list[0]

shape_chin = face_landmarks['chin']
# x,y cordinates on frame where jewelery will be added
x = shape_chin[3][0]
y = shape_chin[6][1]
# Jewelry width & height calculated using face chin cordinates
img_width = abs ( shape_chin[3][0] - shape_chin[14][0])
img_height = int( 1.02 * img_width) 
jewel_img = cv2.resize(jewel_img, (img_width,img_height), interpolation=cv2.INTER_AREA)
jewel_gray = cv2.cvtColor(jewel_img, cv2.COLOR_BGR2GRAY)
# All pixels greater than 230 will be converted to white and others will be converted to black
thresh, jewel_mask = cv2.threshold(jewel_gray, 230, 255, cv2.THRESH_BINARY)
# Convert to black the background of jewelry image
jewel_img[jewel_mask == 255] = 0
# Crop out jewelry area from original frame
jewel_area = frame[y:y+img_height, x:x+img_width]
# bitwise_and will convert all black regions in any image to black in resulting image
masked_jewel_area = cv2.bitwise_and(jewel_area, jewel_area, mask=jewel_mask)
# add both images so that the black region in any image will result in another image non black regions being rendered over that area
final_jewel = cv2.add(masked_jewel_area, jewel_img)
# replace original frame  jewel area with newly created jewel_area
frame[y:y+img_height, x:x+img_width] = final_jewel
# convert image to RGB format to read it in pillow library
rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(rgb_img)
draw = ImageDraw.Draw(pil_img, 'RGBA')

pil_img.show()



