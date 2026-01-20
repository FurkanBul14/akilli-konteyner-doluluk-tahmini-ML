## Veri Seti

Projede **Smart_Bin.csv** adlı veri seti kullanılmıştır.  
Bu veri seti, farklı konteyner türlerinin çeşitli atık türleriyle ne kadar dolduğunu gösteren bilgiler içermektedir.

Veri setinde kullanılan temel sütunlar şunlardır:

- **Container Type:** Konteynerin türünü/modelini ifade eder (ör. Accordion, Diamond vb.)
- **Recyclable fraction:** Konteynerin aldığı atık türünü gösterir (geri dönüştürülebilir, karışık vb.)
- **FL_A:** Ölçümün başındaki doluluk seviyesi
- **FL_B:** Ölçümün sonundaki doluluk seviyesi
- **VS:** Ölçümle ilgili sayısal bir değerdir ve makine öğrenmesi kısmında özellik (feature) olarak kullanılmıştır.

Eksik veya hatalı veriler, analiz ve makine öğrenmesi aşamalarından önce veri setinden çıkarılmıştır.
Aşağıda, veri temizlemeve veri okuma  işleminin kod üzerinde nasıl yapıldığı gösterilmektedir:

![Veri Temizleme Kodu](resimler/resim1.png)

## Pivot Analizi

Pivot tablo ile konteyner türü ve atık türüne göre ortalama doluluk seviyesini karşılaştırdım.

Burada amaç:
- Hangi konteyner + hangi atık türünde ortalama `FL_B` değeri daha yüksek görmek

Pivot tablosundan sonra en yüksek ortalamaya sahip kombinasyonu da ekrana yazdırdım.

![Pivot Analizi Kodu](resimler/resim2.png)


## Makine Öğrenmesi (Ek)

Pivot analizden sonra ek olarak basit bir makine öğrenmesi denedim.

Amaç:
- `FL_B` değeri "yüksek mi / düşük mü?" tahmin etmek

Bu yüzden:
- `FL_B` değerlerini medyana göre iki sınıfa ayırdım (median üstü = 1, altı = 0)

Modele verdiğim bilgiler:
- `Container Type`
- `Recyclable fraction`
- `FL_A`
- `VS`

Kullanılan algoritmalar:
- Logistic Regression
- KNN
- Random Forest


### Koddan bir parça
![Makine Öğrenmesi - Veri Seçimi](resimler/resim3.png)

### Model Eğitimi ve Seçim

Modelleri eğitmeden önce veriyi uygun formata getirmek için bir pipeline kullandım.  
`Container Type` ve `Recyclable fraction` gibi kategorik sütunlar **One-Hot Encoding** ile sayısal hale getirildi.  
`FL_A` ve `VS` ise **StandardScaler** ile ölçeklendirildi.

Daha sonra Logistic Regression, KNN ve Random Forest modelleri aynı eğitim/test ayrımı üzerinde çalıştırıldı.  
Her model için **Accuracy** ve **F1 Score** hesaplandı ve **F1 Score en yüksek olan model “kazanan”** olarak seçildi.

![Model Sonuçları ve Kazanan](resimler/resim4.png)



## Grafikler

Projede iki grafik oluşturdum:

- **Isı haritası (Heatmap):** Pivot tablosundaki ortalama `FL_B` değerlerini konteyner türü ve atık türüne göre görselleştirir. Renk koyulaştıkça ortalama doluluk seviyesi artar.
- **Model karşılaştırma grafiği:** Logistic Regression, KNN ve Random Forest modellerinin `Accuracy` ve `F1 Score` değerlerini yan yana karşılaştırır.

### Heatmap (Pivot Görselleştirme)
![Pivot Heatmap](resimler/resim5.png)

### Model Performans Grafiği
![Model Performans Karsilastirmasi](resimler/resim6.png)


## Sertifikalar

![Sertifika 1](resimler/resim7.png)

![Sertifika 2](resimler/resim8.png)

