import cv2
photo=cv2.imread('tint1.jpg')
cv2.imshow('Krishna',photo)
cv2.waitKey(5000)
cv2.destroyAllWindows()
print(photo)
print(photo.shape)
#color is in BGR In cv2
newphoto=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
print(newphoto.shape)

cv2.imshow('KrishnaLuv',newphoto)
cv2.waitKey(5000)
cv2.destroyAllWindows()

print(photo)
