# Dışarıdan Gelen Kütüphaneler
import atexit

#Bizim Manager Kütüphanelerimiz
import llm_manager
import pdf_manager
import data_manager 

from langchain_huggingface import HuggingFaceEmbeddings




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
    # HuggingFaceEmbeddings, arka planda SentenceTransformer'ı kurar
    embedding_model = HuggingFaceEmbeddings(model_name = model_name, encode_kwargs={"show_progress_bar": True})

    # Verilen path'teki klasör içindeki tüm PDF'lerin sayfalarını 'pages' adlı variable'a atar.
    # Tüm bir PDF'i atmıyoruz, her pdf'in sayfasını ayrı ayrı atıyoruz ancak her sayfanın metadatası ile hangi PDF'e ait olduğunu biliyoruz.
    pdf_pages = pdf_manager.read_pdf_file(pdf_folder_path)

    
    # PDF sayfaları chunk'lara dönüştürülür
    chunks = data_manager.chunk(pdf_pages)

    
    # Chunk'lar vektöre çevrilir ardından sorgulanabilir bir vector veritabanina çevrilir.
    # Tek seferded embedding de burada yapılıyor vektör database'i de burada yapılıyor.
    # Aradan chunk'ları manuel olarak embedding ile vektörlere çevirme işini yapmıyoruz langchain sayesinde.
    vector_db = data_manager.create_vector_database(chunks, embedding_model)


    # Chunk'ların içinde geçen PDF adlarını alfabetik olarak döner
    available_sources = data_manager.list_sources(chunks)

    # Sorgu, belirli bir PDF'te mi aranacak yoksa tüm PDF'lerde mi onun ayrımı yapılır.
    print("\nMevcut PDF'ler:")
    for order, source in enumerate(available_sources, start=1):
        print(f"   {order}. {source}")

    selection = input("\nHangi PDF'te arama yapılsın? (Numara girin, tümü için boş bırakın): ").strip()

    selected_source = None

    if selection == "":
        print("Filtre yok: tüm PDF'lerde aranacak.\n")

    elif selection.isdigit() and 1 <= int(selection) <= len(available_sources):
        selected_source = available_sources[int(selection) - 1]
        print(f"Filtre aktif: sadece '{selected_source}' içinde aranacak.\n")

    else:
        print("Geçersiz seçim. Tüm PDF'lerde aranacak.\n")

    # VectorStore'dan, chain'in kullanabileceği standart bir arama arayüzü üretilir.
    retriever = data_manager.create_retriever(vector_db, k=3, source_filter=selected_source)

    # LLM nesnesi oluşturulur langchaine uygun.
    llm_model_name = "llama3.1"
    llm = llm_manager.create_llm(model_name=llm_model_name)




    #-----------------------------------------------------------------------------------------#

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