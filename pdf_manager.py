import os
import tempfile
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

def search_for_pdf(requested_pdf_folder):
    user_directory = os.path.expanduser("~")
    potential_paths = [
        os.path.join(user_directory, "Desktop"),
        os.path.join(user_directory, "Masaüstü"),
        os.path.join(user_directory, "OneDrive", "Masaüstü"),
        os.path.join(user_directory, "OneDrive", "Desktop")
    ]

    current_desktop = None
    for path in potential_paths:
        if os.path.exists(path):
            current_desktop = path
            break

    if not current_desktop:
        return None
        
    for root, dirs, files in os.walk(current_desktop):
        for dir_name in dirs: 
            if dir_name.lower() == requested_pdf_folder.lower():
                return os.path.join(root, dir_name)
    return None

def read_pdf_file(folder_path):
    pdf_storage = PyPDFDirectoryLoader(path=folder_path, glob="**/*.pdf")
    pdfs = pdf_storage.load()

    for doc in pdfs:
        pdf_name = os.path.basename(doc.metadata.get("source", ""))
        page_index = doc.metadata.get("page", 0)
        doc.metadata = {
            "source": pdf_name,
            "page": page_index
        }
    return pdfs

# --- Sürükle-Bırak / Arayüz Yüklemeleri İçin Fonksiyon ---
def read_uploaded_pdfs(uploaded_files):
    """
    Arayüzden sürükle-bırak veya dosya seçici ile yüklenen PDF dosyalarını okur
    ve standart metadata formatı ile Document nesneleri listesi döner.
    """
    all_documents = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata = {
                    "source": uploaded_file.name,
                    "page": doc.metadata.get("page", 0)
                }
                all_documents.append(doc)
                
    return all_documents