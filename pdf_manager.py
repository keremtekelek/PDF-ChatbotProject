import os
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFDirectoryLoader


# Masaüstünde istenen pdf klasörünü bulur ve yolu döndürür
def search_for_pdf(requested_pdf_folder):

    # os.path.expanduser("~") kodu, kullanıcının ana klasörünü getirmektedir. Örneğin Windows ise "C:\Users\Username" döndürür.
    user_directory = os.path.expanduser("~")


    """
    PDF klasörünü masaüstünde arayacağımız için kullanıcının masaüstüne erişmemiz gerekiyor.
    Ondan ötürü kullanıcının masaüstüne erişebilen potansiyel yolları arıyoruz.
    """
    potential_paths = [
        os.path.join(user_directory, "Desktop"), # C:\Users\Username\Desktop dizinini döndürür.
        os.path.join(user_directory, "Masaüstü"), # C:\Users\Username\Masaüstü dizinini döndürür.
        os.path.join(user_directory, "OneDrive", "Masaüstü"), # C:\Users\Username\OneDrive\Masaüstü dizinini döndürür.
        os.path.join(user_directory, "OneDrive", "Desktop") # C:\Users\Username\OneDrive\Desktop dizinini döndürür.
    ]


    # Olası yollardan mevcut olanı gecerli_masaustu adlı variable'a atıp valid olan desktop'u buluyoruz.
    current_desktop = None
    for path in potential_paths:
        if os.path.exists(path):
            current_desktop = path
            break

    # Olası yollardan herhangi biri path olarak yok ise bilgisayarda, bu masaüstü bulunamamış demektir...
    if not current_desktop:
        print("Bilgisayarınızda Masaüstü bulunamadı.")
        return None
        
    print(f"\n'{requested_pdf_folder}' klasörü Masaüstünde aranıyor...")

    
    # Bulduğumuz masaüstü path'indeki tüm klasörleri ilgili klasörü bulmaya çalışıyoruz ve bunu döndürüyoruz.
    
    for root, dirs, files in os.walk(current_desktop):
        for dir_name in dirs: 
            if dir_name.lower() == requested_pdf_folder.lower():
                found_pdf_folder = os.path.join(root, dir_name)
                print(f"Klasör bulundu: {found_pdf_folder}\n")
                return found_pdf_folder
                
    print(f"'{requested_pdf_folder}' Masaüstünde bulunamadı.")
    return None


# Verilen PDF dosyasını okur ve return olarak PDF'in tüm metnini döner
def read_pdf_file(file_path):

    print("Klasör okunuyor...")

    pdf_storage = PyPDFDirectoryLoader(
        path=file_path,
        glob="**/*.pdf",   # alt klasörlere de bakar
    )

    pdfs = pdf_storage.load()

    for doc in pdfs:
        old_path = doc.metadata.get("source", "")
        pdf_name = os.path.basename(old_path)
        doc.metadata["source"] = pdf_name

    print(f"Toplam {len(pdfs)} sayfa yüklendi.\n")
    return pdfs