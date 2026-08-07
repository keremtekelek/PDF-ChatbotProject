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
atexit.register(llm_manager.close_llm)




    
if __name__ == "__main__":
    print("="*50)
    print("   PDF AI CHATBOT")
    print("="*50)

    # Eğer "llm_otomatik_baslat" fonksiyonu başarıyla çalışmazsa false gönderecektir ve programı kapatmamız gerekir çünkü LLM hazır değil.
    if not llm_manager.start_llm_automatically():
        exit() 

    # Uygulama, User'dan masaüstünde bulunan bir PDF klasörün ismini ister...
    while True:
        requested_pdf_folder = input("Lütfen Masaüstündeki okunmasını istediğiniz klasörünün adını girin (Örn: kitap veya kitap.pdf) [Çıkmak için 'q']: ")
        
        if requested_pdf_folder.lower() in ['q', 'çıkış']:
            print("Sistem kapatılıyor...")
            exit()
            
        pdf_folder_path = pdf_manager.search_for_pdf(requested_pdf_folder)
        
        if pdf_folder_path:
            break 
        else:
            print("Lütfen dosya adını kontrol edip tekrar deneyin.\n")


    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    print(f"'{model_name}' yükleniyor...")

    # SentenceTransformer ile verilen modelin neural network altyapısını kurar, hazırlar, model inmediyse modeli indirir.
    # Ve model objesini "embedding_modeli" adlı variable'a atar.
    embedding_model = SentenceTransformer(model_name)

    # Önce bulunan PDF okunur ve PDF'teki tüm metin bulunur.
    full_text = pdf_manager.read_pdf_file(pdf_folder_path)

    
    # Tüm metin chunk'lara dönüştürülür.
    chunks = data_manager.chunk(full_text)

    # Bu chunk'lar vektörlere çevrilir.
    vectors = data_manager.embedding(chunks, embedding_model)

    # Bu vektörler ile de Q-drant kullanılarak vektör veri tabanı oluşturulur
    qdrant_client, qdrant_collection = data_manager.create_vector_database(vectors, chunks)
    
    print("="*50)
    print("   SİSTEM HAZIR! SOHBETE BAŞLAYABİLİRSİNİZ")
    print("="*50)
    print("PDF hakkında sorular sorabilirsiniz. Çıkmak için 'q' veya 'cikis' yazın.\n")
    
    while True:
        user_question = input("\nSiz: ")

        # Çıkış ifadelernden herhangi biri yazılır ise program kapatılır.
        if user_question.lower() in ['q', 'cikis', 'çıkış', 'exit']:
            print("Görüşmek üzere!")
            break
            
        if user_question.strip() == "":
            continue

        # Bağlam bulunur
        related_context = data_manager.find_context(
            question=user_question, 
            qdrant_client=qdrant_client, 
            collection_name=qdrant_collection, 
            model=embedding_model, 
            k=3
        )

        # Cevap üretilir.
        print("Chatbot düşünüyor...")
        ai_answer = llm_manager.generate_answer(
            question=user_question, 
            context=related_context, 
            model_name="llama3.1" 
        )

        # Cevap yazdırılır.
        print(f"\nChatbot: {ai_answer}")
        print("-" * 50)