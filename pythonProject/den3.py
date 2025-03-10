from ultralytics import YOLO
from PIL import Image
# resim dosyası üzerinde nesne tanıma-++
#best1 en iyi sonuc buluyo
model = YOLO('best 2.pt')
im1 = Image.open("stop.jpg")
sonuc = model.predict(source=im1, save=True)

# best 2 son eğittiğim kullanılacak
#best 1 de az veri var kullanılmayacak
# best 3 deneme eğititmi kullanılmayacak