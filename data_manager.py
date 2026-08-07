from qdrant_client import QdrantClient, models


# "Chunking" yani parçalara bölme işlemi yapar. Return olarak parça döner.
def metni_parcalara_bol(metin, parca_buyuklugu=400, kesisim_payi=30):
    print("Chunking yapılıyor...")

    # Metni kelimelere böler 'split()' fonksiyonu ile.
    kelimeler = metin.split()
    parcalar = []

    # Buradan adım miktarını elde ederiz.
    adim_miktari = parca_buyuklugu - kesisim_payi

    # Her seferde adım_miktari kadar ileri gider. Buradaki sayımız 370'tir. Yani her seferde 370 kelime iterate eder.
    for i in range(0, len(kelimeler), adim_miktari):
        parca_metni = " ".join(kelimeler[i : i + parca_buyuklugu])
        parcalar.append(parca_metni)
    print(f"Toplam {len(parcalar)} adet chunk oluşturuldu.\n")
    return parcalar

# Chunk'ları vektöre çevirir ve return olarak vektörleri döndürür.
def metinleri_vektore_cevir(chunk_listesi, embedding_modeli):
    print("Metin parçaları vektörlere çevriliyor...")

    #Tüm chunk'ları vektörlere dönüştürdük encode() fonksiyonu ile.
    vektorler = embedding_modeli.encode(chunk_listesi, show_progress_bar=True)
    return vektorler

# Vector veritabınını oluşturur ve return olarak client ve collection_name'i döndürür.
def vektor_veritabani_olustur(vektorler, chunk_list):
    print("\n Q-Drant vektör veritabanı oluşturuluyor...")

    # Vektörün boyut sayısını hesaplarız
    vektor_boyutu = vektorler.shape[1]

    # Q-Drant vektör data base'inin başlatılması kısmı

    #Bizim server ile alakalımız olmadığı için sadece memory'de kullanmak istiyoruz (bkz:faiss gibi).
    client = QdrantClient(":memory:")

    # Q-Drant'taki collection, ilişkisel veritabanlarıdaki tablolar gibi düşünebiliriz. Vektörler ve metadataları depolar.
    # Collection'ımıza unique bir isim veriyoruz.
    collection_isim = "first_drant_collection"

    # İsmi verdikten sonra bir collection oluşturuyoruz. İsimi yukarıda verdiğimiz ismi verdik, vektör konfigirasyonunda ise size'ımızı vektör boyutu olarak verdik
    # Vektörler arası Similarity Search metodu olarak da COSINE kullandık (bkz: Cosine-Similarity)
    client.create_collection(
    collection_name=collection_isim,
    vectors_config=models.VectorParams(
        size=vektor_boyutu,  
        distance=models.Distance.COSINE  
    )
    )


    """ 
     Q-Drant'ta temel veri varlıkları noktalardır. Her nokta ID, Vector Data, Payload(Opsiyonel) vardır. Payload'a ek meta-data da diyebiliriz.
     Ayrıca vektörleri Q-Drant'ın anlayabileceği tarz olan points'lere çevirmemiz gerekmektedir.

     Önemli olan diğer bir konu ise FAISS'te sadece vektörleri vermek yeterli olur iken Q-Drant'ta hem chunk'ı hem vektörü vermemizin temel sebebi
     FAISS'in Q-Drant gibi bir vektör veritabanı değildir, sadece vektör arama kütüphanesidir. Q-Drant ise hem vektör arar + bir vektör veritabanıdır + server vardır.

     Aşağıdaki for kodunda olan şey tam olarak şudur:

     Diyelimki elimizde 2 liste var:

     vektorler = [ [0.1, 0.5], [0.8, 0.2] ]
     chunk_list = [ "Elma tatlıdır", "Limon ekşidir" ]

     kod kısmında zip(vektorler, chunk_list) dendiğinde eşleştirir => ( [0.1, 0.5], "Elma tatlıdır" ), ( [0.8, 0.2], "Limon ekşidir" )

     enumerate dediğinizde ise indeks verir => 0, ( [0.1, 0.5], "Elma tatlıdır" ) /// 1, ( [0.8, 0.2], "Limon ekşidir" ) 

     for'un içine geçtiğimizde ise id kısmında indeks (yani yukarıdaki örnekte 0 ya da 1), vector kısmına vektör ([0.1, 0.5]),
     payload kısmına ise örnekteki ("limon ekşidir") kısmı gelir. 

     En sonda da bu her 'nokta'yı boş bir array olan 'noktalar'a ekleriz

    """
    noktalar = []
    for indeks, (vektor, metin_parcasi) in enumerate(zip(vektorler, chunk_list)):
        nokta = models.PointStruct(
            id=indeks, 
            vector=vektor.tolist(), 
            payload={"metin": metin_parcasi} 
        )
        noktalar.append(nokta)

    # Noktaları veritabanına yüklüyoruz.
    client.upsert(
        collection_name=collection_isim,
        points=noktalar
    )
   
    print(f"Toplam {len(noktalar)} adet vektör eklendi.\n")

    return client, collection_isim

# Soruyla ilgili metinleri bulup bağlamı döndürür.
def cevap_icin_baglam_bul(soru, qdrant_client, koleksiyon_isim, model, k=3):

    # Sorulan soru, vektöre çevrilir.
    soru_vektoru = model.encode(soru)

    # Qdrant üzerinde arama işlemi yapıyoruz.
    arama_sonuclari = qdrant_client.query_points(
        collection_name = koleksiyon_isim,
        query = soru_vektoru.tolist(),
        limit=k
    )

    # Gelen sonuçların içindeki 'payload'dan metinleri çıkartıp bağlamı buluyoruz
    bulunan_metinler = []
    for sonuc in arama_sonuclari.points:
        bulunan_metinler.append(sonuc.payload["metin"])
        
    return "\n\n".join(bulunan_metinler)