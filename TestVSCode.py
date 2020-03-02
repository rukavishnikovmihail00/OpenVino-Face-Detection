import cv2



prototxt = 'face-detection-retail-0005.xml'
weight = 'face-detection-retail-0005.bin'

net = cv2.dnn.readNet(prototxt, weight)
stream = cv2.VideoCapture(0)
while True:
    grab, frame = stream.read()
    img = frame.copy()

    blob = cv2.dnn.blobFromImage(frame, size=(300, 300))
    net.setInput(blob) #загрузка изображения в модель
    out = net.forward() # возврат результатов обработки
    out = out.reshape(-1, 7) 

    for detection in out:
        confidence = detection[2]
        if confidence > 0.6:
            xmin = int(detection[3] * img.shape[1])
            ymin = int(detection[4] * img.shape[0])
            xmax = int(detection[5] * img.shape[1])
            ymax = int(detection[6] * img.shape[0])

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

    cv2.imshow("out", img)
    k = cv2.waitKey(1)

    if k == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()
