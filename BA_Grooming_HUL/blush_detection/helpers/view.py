import cv2

def showImage(imageName, image):
    img = image.copy()
    cv2.imshow(imageName, img)
    while(1):
        pressedKey = cv2.waitKey(0) & 0xFF

        if(pressedKey == ord('q')):
            cv2.destroyWindow(imageName)
            break
        else:
            cv2.putText(img, "\press q to exit", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color=(255,0,0))
            cv2.imshow(imageName, img)