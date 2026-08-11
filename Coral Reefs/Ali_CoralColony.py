import numpy as np
import cv2




def load_image(old_path, new_path):
    img_old = cv2.imread(old_path)
    img_new = cv2.imread(new_path)

    return img_old, img_new


def preprocess(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_hsv

def clean_mask(mask):

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel
    )

    return mask



def detect_colors(hsv):

    lower_pink = np.array([160, 100, 100])
    upper_pink = np.array([180, 255, 255])

    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)

    lower_white = np.array([0,0,180])
    upper_white = np.array([180,50,255])


    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    return pink_mask, white_mask

def dilate_mask(mask):
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.dilate(mask,kernel,iterations=1)
    return mask

def compare_corals(old_pink, old_white, new_pink, new_white):
    kernel = np.ones((9,9), np.uint8)
    old_pink_expand = cv2.dilate(old_pink, kernel)
    new_pink_expand = cv2.dilate(new_pink, kernel)
    growth = cv2.subtract(new_pink,old_pink_expand)
    damage = cv2.subtract(old_pink,new_pink_expand)
    bleaching = cv2.bitwise_and(old_pink,new_white)
    recovery = cv2.bitwise_and(old_white,new_pink)
    return growth, damage, bleaching, recovery


def draw_boxes(image,mask,color):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area=cv2.contourArea(c)
        if area>350:
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
    return image



old_img,new_img = load_image(
    'Coral_colony_photo_1_year_ago.jpg',
    'Coral_colony_photo_today.jpg'
)

old_img = cv2.resize(old_img,(512,512))
new_img = cv2.resize(new_img,(512,512))


result = new_img.copy()

old_hsv = preprocess(old_img)
new_hsv = preprocess(new_img)

old_pink_mask,old_white_mask = detect_colors(old_hsv)
new_pink_mask,new_white_mask = detect_colors(new_hsv)



old_pink_mask = clean_mask(old_pink_mask)
old_white_mask = clean_mask(old_white_mask)
new_pink_mask = clean_mask(new_pink_mask)
new_white_mask = clean_mask(new_white_mask)

old_pink_mask = dilate_mask(old_pink_mask)
new_pink_mask = dilate_mask(new_pink_mask)


growth,damage,bleaching,recovery=compare_corals(old_pink_mask,old_white_mask,new_pink_mask,new_white_mask)
GREEN =(0,255,0)
YELLOW=(0,255,255)
RED=(0,0,255)
BLUE=(255,0,0)


result = draw_boxes(result,growth,GREEN)
result = draw_boxes(result,damage,YELLOW)
result = draw_boxes(result,bleaching,RED)
result = draw_boxes(result,recovery,BLUE)

cv2.imshow("Coral Changes", result)

cv2.imshow("Old Pink", old_pink_mask)
cv2.imshow("New Pink", new_pink_mask)

cv2.imshow("Old White", old_white_mask)
cv2.imshow("New White", new_white_mask)

overlay = cv2.addWeighted(
    old_img,
    0.5,
    new_img,
    0.5,
    0
)

cv2.imshow("Alignment Check", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()
