import os
from PyPDF2 import PdfReader


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