import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader

def read_uploaded_pdfs(uploaded_files):
    """
    Sürükle-bırak veya dosya seçici ile yüklenen PDF'leri okur
    ve standart metadata ile Document listesi döner.
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