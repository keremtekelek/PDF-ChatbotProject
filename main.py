import os
import time
import subprocess
import requests
import numpy as np
import atexit  
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Boş bir variable oluşturuyoruz. Ollama process'iin temsil amaçlı.
ollama_process = None


def llm_kapat():
    global ollama_process

    #Eğer ollama process'i boş değil ise dolu demektir yani bu process vardır demektir. Var ise durdurmak istiyoruz.
    if ollama_process is not None:
        print("\n Kapanış işlemleri: Local LLM durduruluyor...")

        #Process'i terminate edip nullptr yapıyoruz.
        ollama_process.terminate()  
        ollama_process = None
        print("Local LLM başarıyla kapatıldı. RAM temizlendi!")

# at-exit yani çıkışta yani program sona erdirildiğinde çalışacak olan fonksiyonu göstermektedir.
# Program çalışmayı bitirdiğinde llm_kapat adlı fonksiyonu çalıştırıp llm'i kapatacağız.
atexit.register(llm_kapat)



def llm_otomatik_baslat():
    global ollama_process
    print("Local LLM durumu kontrol ediliyor...")
    url = "http://localhost:11434/"

    # İlgili url'e bir request atıyoruz eğer 200 cevabı geliyor ise her şey yolunda demektir
    try:
        cevap = requests.get(url)
        if cevap.status_code == 200:
            print("Local LLM zaten arka planda çalışıyor...\n")
            return True
        if cevap.status_code == 500:
            print("Local LLM sunucusu çökmüş durumdadır.")
            return False
        
    # Eğer ConnectionError yer isek bu LLM kapalı demektir.
    except requests.exceptions.ConnectionError:
        print("Local LLM şu an kapalı. Sizin için otomatik olarak başlatılıyor...")
        try:
            # LLM kapalı olduğunda ollama process'i için bir subprocess açarız.
            ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            #5 saniye içinde LLM'in çalışması bekleniyor.
            for i in range(5, 0, -1):
                print(f"LLM'in uyanması bekleniyor... {i} saniye")
                time.sleep(1)
            print("Local LLM başarıyla başlatıldı!\n")
            return True

        # Ollama hiç yüklü değil ise bu kısım çalışır.
        except FileNotFoundError:
            print("Bilgisayarınızda Ollama yüklü değil veya bulunamadı!")
            return False

# Masaüstünde istenen pdf'i arar, return olarak dosya yolunu verir.
def masaustunde_pdf_ara(dosya_adi):

    # Masaüstünde pdf ararken "mutlaka .pdf biterek yaz" demediğimiz için, sadece pdf'in ismi de yeterli olduğu için sonu .pdf ile bitmiyorsa
    # Sonuna .pdf ekliyoruz.
    if not dosya_adi.lower().endswith('.pdf'):
        dosya_adi += '.pdf'

    # os.path.expanduser("~") kodu, kullanıcının ana klasörünü getirmektedir. Örneğin Windows ise "C:\Users\Username" döndürür.
    kullanici_dizini = os.path.expanduser("~")


    """
    PDF'i masaüstünde arayacağımız için kullanıcının masaüstüne erişmemiz gerekiyor.
    Ondan ötürü kullanıcının masaüstüne erişebilen potansiyel yolları arıyoruz.
    """
    olasi_yollar = [
        os.path.join(kullanici_dizini, "Desktop"), # C:\Users\Username\Desktop dizinini döndürür.
        os.path.join(kullanici_dizini, "Masaüstü"), # C:\Users\Username\Masaüstü dizinini döndürür.
        os.path.join(kullanici_dizini, "OneDrive", "Masaüstü"), # C:\Users\Username\OneDrive\Masaüstü dizinini döndürür.
        os.path.join(kullanici_dizini, "OneDrive", "Desktop") # C:\Users\Username\OneDrive\Desktop dizinini döndürür.
    ]


    # Olası yollardan mevcut olanı gecerli_masaustu adlı variable'a atıp valid olan desktop'u buluyoruz.
    gecerli_masaustu = None
    for yol in olasi_yollar:
        if os.path.exists(yol):
            gecerli_masaustu = yol
            break

    # Olası yollardan herhangi biri path olarak yok ise bilgisayarda, bu masaüstü bulunamamış demektir...
    if not gecerli_masaustu:
        print("Bilgisayarınızda Masaüstü bulunamadı.")
        return None
        
    print(f"\n'{dosya_adi}' dosyası Masaüstünde aranıyor...")

    """
    Bulduğumuz masaüstü path'indeki tüm klasör ve dosyaları buluyoruz. Sonra dosyaları geziyoruz. Ardından aradığımız pdf'i buluyoruz.
    """
    for root, dirs, files in os.walk(gecerli_masaustu):
        for file in files:
            if file.lower() == dosya_adi.lower():
                bulunan_yol = os.path.join(root, file)
                print(f"Dosya bulundu: {bulunan_yol}\n")
                return bulunan_yol
                
    print(f"'{dosya_adi}' Masaüstünde bulunamadı.")
    return None

# Verilen PDF dosyasını okur ve return olarak PDF'in tüm metnini döner
def pdf_oku(dosya_yolu):
    print("PDF dosyası okunuyor...")

    # Verilen dosya_yolu'nda bulunan PDF'i PyPDF2 kütüphanesi ile birlikte dosyayı okunmaya hazır hale getirir.
    okuyucu = PdfReader(dosya_yolu)

    # PDF'teki tüm metni çıkardık.
    tum_metin = ""
    for sayfa in okuyucu.pages:
        sayfa_metni = sayfa.extract_text()
        if sayfa_metni:
            tum_metin += sayfa_metni + "\n"
    print(f"Okuma tamamlandı! Toplam {len(okuyucu.pages)} sayfa işlendi.\n")
    return tum_metin

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

    

# Cevabı döndürür.
def llm_ile_cevap_uret(soru, baglam, model_adi="llama3.1"):
    prompt = f"""Sen zeki ve yardımcı bir asistansın. Sana verilen 'Bağlam' metnini okuyarak kullanıcının sorusunu cevapla. 
Sadece bağlamda geçen bilgileri kullan. Eğer sorunun cevabı bağlamda yoksa, kesinlikle kendi bilginden uydurma yapma ve sadece "Bu bilgi PDF dosyasında bulunmuyor." de kusursuz, akıcı ve dilbilgisi kurallarına uygun bir Türkçe ile cevapla. Çeviri kokan veya anlamsız cümleler kurmaktan kesinlikle kaçın.

Bağlam:
{baglam}

Soru: {soru}

Cevap:"""

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_adi, 
        "prompt": prompt,

        # Buradaki "stream" variable'ı, AI'ın oluşturduğu cevabı tekte mi yoksa daktilo gibi ürettikçe ekrana mı yansıtmalı onu temsil eder
        # "True" denirse daktilo efekti aratır, "false" denirse tüm cevabı tekte verir.
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Hata oluştu! Hata Kodu: {response.status_code}"
    except Exception as e:
        return f"LLM ile bağlantı kurulamadı: {e}"


if __name__ == "__main__":
    print("="*50)
    print("   PDF AI CHATBOT")
    print("="*50)

    # Eğer "llm_otomatik_baslat" fonksiyonu başarıyla çalışmazsa false gönderecektir ve programı kapatmamız gerekir çünkü LLM hazır değil.
    if not llm_otomatik_baslat():
        exit() 

    # Uygulama, User'dan masaüstünde bulunan bir PDF dosyanın adını ister. Kullanıcı valid bir pdf dosyası ismi verene kadar sormaya devam eder.
    pdf_yolu = None
    while True:
        istenen_pdf = input("Lütfen Masaüstündeki PDF dosyasının adını girin (Örn: kitap veya kitap.pdf) [Çıkmak için 'q']: ")
        
        if istenen_pdf.lower() in ['q', 'çıkış']:
            print("Sistem kapatılıyor...")
            exit()
            
        pdf_yolu = masaustunde_pdf_ara(istenen_pdf)
        
        if pdf_yolu:
            break 
        else:
            print("Lütfen dosya adını kontrol edip tekrar deneyin.\n")


    model_adi = "paraphrase-multilingual-MiniLM-L12-v2"
    print(f"'{model_adi}' yükleniyor...")

    # SentenceTransformer ile verilen modelin neural network altyapısını kurar, hazırlar, model inmediyse modeli indirir.
    # Ve model objesini "embedding_modeli" adlı variable'a atar.
    embedding_modeli = SentenceTransformer(model_adi)

    # Önce bulunan PDF okunur ve PDF'teki tüm metin bulunur.
    tum_metin = pdf_oku(pdf_yolu)

    # Tüm metin chunk'lara dönüştürülür.
    chunklar = metni_parcalara_bol(tum_metin)

    # Bu chunk'lar vektörlere çevrilir.
    vektorler = metinleri_vektore_cevir(chunklar, embedding_modeli)

    # Bu vektörler ile de Q-drant kullanılarak vektör veri tabanı oluşturulur
    qdrant_client, qdrant_collection = vektor_veritabani_olustur(vektorler, chunklar)
    
    print("="*50)
    print("   SİSTEM HAZIR! SOHBETE BAŞLAYABİLİRSİNİZ")
    print("="*50)
    print("PDF hakkında sorular sorabilirsiniz. Çıkmak için 'q' veya 'cikis' yazın.\n")
    
    while True:
        kullanici_sorusu = input("\nSiz: ")

        # Çıkış ifadelernden herhangi biri yazılır ise program kapatılır.
        if kullanici_sorusu.lower() in ['q', 'cikis', 'çıkış', 'exit']:
            print("Görüşmek üzere!")
            break
            
        if kullanici_sorusu.strip() == "":
            continue

        # Bağlam bulunur
        ilgili_baglam = cevap_icin_baglam_bul(
            soru=kullanici_sorusu, 
            qdrant_client=qdrant_client, 
            koleksiyon_isim=qdrant_collection, 
            model=embedding_modeli, 
            k=3
        )

        # Cevap üretilir.
        print("Chatbot düşünüyor...")
        ai_cevabi = llm_ile_cevap_uret(
            soru=kullanici_sorusu, 
            baglam=ilgili_baglam, 
            model_adi="llama3.1" 
        )

        # Cevap yazdırılır.
        print(f"\nChatbot: {ai_cevabi}")
        print("-" * 50)