import cv2

# تحميل الفيديو
cap = cv2.VideoCapture('Videli Free Footage - Different Casual People Walking.mp4')

# إنشاء كاشف HOG وتفعيل كاشف الأشخاص الافتراضي
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تصغير الإطار لتسريع المعالجة
    frame = cv2.resize(frame, (640, 360))

    # كشف الأشخاص
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    # رسم مستطيل حول كل شخص
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # عرض النتيجة
    cv2.imshow('People Detection (HOG)', frame)

    # اضغطي Q للخروج
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
