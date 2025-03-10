import cv2
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO("best (1).pt")  # Model dosyasının doğru adı ve yolu

# Test edilecek görüntü yolu
image_path = "test_image.jpg"  # Test etmek istediğiniz görüntünün tam yolu

# Görüntüyü yükle
image = cv2.imread(image_path)

# Modeli çalıştır ve sonuçları al
results = model(image)

# Görüntüdeki sonuçları çiz
annotated_image = results[0].plot()

# Sonuçları ekrana yazdır
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])  # Sınıf etiketi
        conf = box.conf[0]    # Güven skoru
        print(f"Sınıf: {cls}, Güven: {conf:.2f}")

# Çizilmiş görüntüyü göster
cv2.imshow("Sonuç", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
