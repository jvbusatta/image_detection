import cv2

# carrega a imagem
img = cv2.imread("soybean_plantation.jpg")

# converte a imagem para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# aplica o detector de contorno
contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# marca as pragas com um contorno azul
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w > 15 and h > 15:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

# salva a imagem com as pragas marcadas
cv2.imwrite("soybean_plantation_with_pests.jpg", img)