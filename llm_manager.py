import time
import subprocess
import requests

from langchain_ollama import ChatOllama


# Boş bir variable oluşturuyoruz. Ollama process'iin temsil amaçlı.
ollama_process = None


def close_llm():
    global ollama_process

    #Eğer ollama process'i boş değil ise dolu demektir yani bu process vardır demektir. Var ise durdurmak istiyoruz.
    if ollama_process is not None:
        print("\n Kapanış işlemleri: Local LLM durduruluyor...")

        #Process'i terminate edip nullptr yapıyoruz.
        ollama_process.terminate()  
        ollama_process = None
        print("Local LLM başarıyla kapatıldı. RAM temizlendi!")

def start_llm_automatically():

    global ollama_process
    print("Local LLM durumu kontrol ediliyor...")
    url = "http://localhost:11434/"

    # İlgili url'e bir request atıyoruz eğer 200 cevabı geliyor ise her şey yolunda demektir
    try:
        answer = requests.get(url)
        if answer.status_code == 200:
            print("Local LLM zaten arka planda çalışıyor...\n")
            return True
        if answer.status_code == 500:
            print("Local LLM sunucusu çökmüş durumdadır.")
            print(f"Local LLM beklenmedik bir cevap verdi. Kod: {answer.status_code}")
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

# Ollama'daki modele bağlanan, LangChain uyumlu bir LLM nesnesi üretir.
def create_llm(model_name="llama3.1"):

    llm = ChatOllama(
        model=model_name,

        # Temperature, kısaca risk seviyesidir. 0 değeri en olasını tercih ederken 1 değeri ya da daha yüksek değerler risk alır, yaratıcı olmaya çalışır. Buna gerek yok.
        temperature=0
    )

    return llm