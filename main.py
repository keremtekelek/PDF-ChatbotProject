# Dışarıdan Gelen Kütüphaneler
import numpy as np
import atexit

#Bizim Manager Kütüphanelerimiz
import llm_manager
import pdf_manager
import data_manager 

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document



# at-exit yani çıkışta yani program sona erdirildiğinde çalışacak olan fonksiyonu göstermektedir.
# Program çalışmayı bitirdiğinde llm_kapat adlı fonksiyonu çalıştırıp llm'i kapatacağız.
atexit.register(llm_manager.llm_kapat)




    
if __name__ == "__main__":
    print("="*50)
    print("   PDF AI CHATBOT")
    print("="*50)

    # Eğer "llm_otomatik_baslat" fonksiyonu başarıyla çalışmazsa false gönderecektir ve programı kapatmamız gerekir çünkü LLM hazır değil.
    if not llm_manager.llm_otomatik_baslat():
        exit() 

    # Uygulama, User'dan masaüstünde bulunan bir PDF dosyanın adını ister. Kullanıcı valid bir pdf dosyası ismi verene kadar sormaya devam eder.
    pdf_yolu = None
    while True:
        istenen_pdf = input("Lütfen Masaüstündeki PDF dosyasının adını girin (Örn: kitap veya kitap.pdf) [Çıkmak için 'q']: ")
        
        if istenen_pdf.lower() in ['q', 'çıkış']:
            print("Sistem kapatılıyor...")
            exit()
            
        pdf_yolu = pdf_manager.masaustunde_pdf_ara(istenen_pdf)
        
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
    tum_metin = pdf_manager.pdf_oku(pdf_yolu)

    # Tüm metin chunk'lara dönüştürülür.
    chunklar = data_manager.metni_parcalara_bol(tum_metin)

    # Bu chunk'lar vektörlere çevrilir.
    vektorler = data_manager.metinleri_vektore_cevir(chunklar, embedding_modeli)

    # Bu vektörler ile de Q-drant kullanılarak vektör veri tabanı oluşturulur
    qdrant_client, qdrant_collection = data_manager.vektor_veritabani_olustur(vektorler, chunklar)
    
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
        ilgili_baglam = data_manager.cevap_icin_baglam_bul(
            soru=kullanici_sorusu, 
            qdrant_client=qdrant_client, 
            koleksiyon_isim=qdrant_collection, 
            model=embedding_modeli, 
            k=3
        )

        # Cevap üretilir.
        print("Chatbot düşünüyor...")
        ai_cevabi = llm_manager.llm_ile_cevap_uret(
            soru=kullanici_sorusu, 
            baglam=ilgili_baglam, 
            model_adi="llama3.1" 
        )

        # Cevap yazdırılır.
        print(f"\nChatbot: {ai_cevabi}")
        print("-" * 50)