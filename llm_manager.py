import time
import subprocess
import requests


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