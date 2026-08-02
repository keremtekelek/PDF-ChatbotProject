import os
import time
import subprocess
import requests
import numpy as np
import faiss
import atexit  
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


ollama_sureci = None


def llm_kapat():
    global ollama_sureci
    if ollama_sureci is not None:
        print("\n[SİSTEM] Kapanış işlemleri: Local LLM (Ollama) durduruluyor...")
        ollama_sureci.terminate()  
        ollama_sureci = None
        print("[SİSTEM] Local LLM başarıyla kapatıldı. RAM temizlendi!")

atexit.register(llm_kapat)



def llm_otomatik_baslat():
    global ollama_sureci
    print("Local LLM durumu kontrol ediliyor...")
    url = "http://localhost:11434/"
    
    try:
        cevap = requests.get(url)
        if cevap.status_code == 200:
            print("Local LLM zaten arka planda çalışıyor...\n")
            return True
    except requests.exceptions.ConnectionError:
        print("Local LLM şu an kapalı. Sizin için otomatik olarak başlatılıyor...")
        try:
            ollama_sureci = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            for i in range(5, 0, -1):
                print(f"LLM'in uyanması bekleniyor... {i} saniye")
                time.sleep(1)
            print("Local LLM başarıyla başlatıldı!\n")
            return True
        except FileNotFoundError:
            print("Bilgisayarınızda Ollama yüklü değil veya bulunamadı!")
            return False

def masaustunde_pdf_ara(dosya_adi):
    if not dosya_adi.lower().endswith('.pdf'):
        dosya_adi += '.pdf'
        
    kullanici_dizini = os.path.expanduser("~")
    
    olasi_yollar = [
        os.path.join(kullanici_dizini, "Desktop"),
        os.path.join(kullanici_dizini, "Masaüstü"),
        os.path.join(kullanici_dizini, "OneDrive", "Masaüstü"),
        os.path.join(kullanici_dizini, "OneDrive", "Desktop")
    ]
    
    gecerli_masaustu = None
    for yol in olasi_yollar:
        if os.path.exists(yol):
            gecerli_masaustu = yol
            break
            
    if not gecerli_masaustu:
        print("Bilgisayarınızda Masaüstü bulunamadı.")
        return None
        
    print(f"\n'{dosya_adi}' dosyası Masaüstünde aranıyor...")
    
    for root, dirs, files in os.walk(gecerli_masaustu):
        for file in files:
            if file.lower() == dosya_adi.lower():
                bulunan_yol = os.path.join(root, file)
                print(f"Dosya bulundu: {bulunan_yol}\n")
                return bulunan_yol
                
    print(f"'{dosya_adi}' Masaüstünde bulunamadı.")
    return None

def pdf_oku(dosya_yolu):
    print("PDF dosyası okunuyor...")
    okuyucu = PdfReader(dosya_yolu)
    tum_metin = ""
    for sayfa in okuyucu.pages:
        sayfa_metni = sayfa.extract_text()
        if sayfa_metni:
            tum_metin += sayfa_metni + "\n"
    print(f"Okuma tamamlandı! Toplam {len(okuyucu.pages)} sayfa işlendi.\n")
    return tum_metin

def metni_parcalara_bol(metin, parca_buyuklugu=400, kesisim_payi=30):
    print("Chunking yapılıyor...")
    kelimeler = metin.split()
    parcalar = []
    adim_miktari = parca_buyuklugu - kesisim_payi
    for i in range(0, len(kelimeler), adim_miktari):
        parca_metni = " ".join(kelimeler[i : i + parca_buyuklugu])
        parcalar.append(parca_metni)
    print(f"Toplam {len(parcalar)} adet metin parçası (chunk) oluşturuldu.\n")
    return parcalar

def metinleri_vektore_cevir(chunk_listesi, embedding_modeli):
    print("Metin parçaları vektörlere çevriliyor...")
    vektorler = embedding_modeli.encode(chunk_listesi, show_progress_bar=True)
    return vektorler

def vektor_veritabani_olustur(vektorler):
    print("\n Faiss vektör veritabanı oluşturuluyor...")
    vektor_boyutu = vektorler.shape[1]
    faiss_indeksi = faiss.IndexFlatL2(vektor_boyutu)
    vektorler_float32 = np.array(vektorler).astype('float32')
    faiss_indeksi.add(vektorler_float32)
    print(f"Veritabanı hazır! Toplam {faiss_indeksi.ntotal} adet vektör eklendi.\n")
    return faiss_indeksi

def cevap_icin_baglam_bul(soru, faiss_indeksi, chunk_listesi, model, k=3):
    soru_vektoru = model.encode([soru])
    soru_vektoru_float32 = np.array(soru_vektoru).astype('float32')
    mesafeler, indeksler = faiss_indeksi.search(soru_vektoru_float32, k)
    
    bulunan_metinler = [chunk_listesi[sirasi] for sirasi in indeksler[0]]
    return "\n\n".join(bulunan_metinler)

def llm_ile_cevap_uret(soru, baglam, model_adi="phi3"):
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
    
    if not llm_otomatik_baslat():
        exit() 

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
    embedding_modeli = SentenceTransformer(model_adi)

    tum_metin = pdf_oku(pdf_yolu)
    chunklar = metni_parcalara_bol(tum_metin)
    vektorler = metinleri_vektore_cevir(chunklar, embedding_modeli)
    faiss_db = vektor_veritabani_olustur(vektorler)
    
    print("="*50)
    print("   SİSTEM HAZIR! SOHBETE BAŞLAYABİLİRSİNİZ")
    print("="*50)
    print("PDF hakkında sorular sorabilirsiniz. Çıkmak için 'q' veya 'cikis' yazın.\n")
    
    while True:
        kullanici_sorusu = input("\nSiz: ")
        
        if kullanici_sorusu.lower() in ['q', 'cikis', 'çıkış', 'exit']:
            print("Görüşmek üzere!")
            break
            
        if kullanici_sorusu.strip() == "":
            continue
            
        ilgili_baglam = cevap_icin_baglam_bul(
            soru=kullanici_sorusu, 
            faiss_indeksi=faiss_db, 
            chunk_listesi=chunklar, 
            model=embedding_modeli, 
            k=3
        )
        
        print("Chatbot düşünüyor...")
        ai_cevabi = llm_ile_cevap_uret(
            soru=kullanici_sorusu, 
            baglam=ilgili_baglam, 
            model_adi="llama3.1" 
        )
        
        print(f"\nChatbot: {ai_cevabi}")
        print("-" * 50)