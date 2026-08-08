from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore


# "Chunking" yani parçalara bölme işlemi yapar. Return olarak parça döner.
def chunk(documents, chunk_size=1000, overlap_margin=150):
    print("Chunking yapılıyor...")

    """ 
     Önceden olduğu gibi bir PDF'i tamamen bir devasa büyük bir stringe dönüştürürsek AI'ın verdiği cevap hangi pdf'ten bunu çözmemiz mümkün olmazdı.
     Bunun önüne geçmek için Langchain'in 'Document' nesnesini kullanmamız gerekmektedir. Text_Splitter ile bu document nesnesinin metadata'sını
     kullanabiliyoruz.
     """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_margin,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)

    print(f"Toplam {len(documents)} sayfadan {len(chunks)} adet chunk oluşturuldu.\n")

    return chunks

# Tek seferded embedding de burada yapılıyor vektör database'i de burada yapılıyor.
# Ardından chunk'ları manuel olarak embedding ile vektörlere çevirme işini yapmıyoruz langchain sayesinde. Sorgulanabilir VectorStore dönüdüyoruz.
def create_vector_database(chunks, embedding_model, collection_name="pdf_chatbot"):

    print("\nQ-Drant vektör veritabanı oluşturuluyor...")

    """
    from_documents tek başına;
    Gider önce Q-Drant Client'ı açar => Vektör boyutunu embedding modeline sorup alır => Chunları embedding yapıp vektörlere çevirir
    => Her chunk için point üretir, payload'u metadata ve metinle doldurur => Hepsini koleksiyona yükler.

    Biz bunların hepsini elle yazıyorduk ancak bu artık otomatize olmuş durumda.
    """

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        location=":memory:",
        collection_name=collection_name
    )

    print(f"Toplam {len(chunks)} adet chunk veritabanına eklendi.\n")
    return vector_store

# Soruyla ilgili metinleri bulup bağlamı döndürür.
def find_context(question, qdrant_client, collection_name, model, k=3):

    # Sorulan soru, vektöre çevrilir.
    question_vector = model.encode(question)

    # Qdrant üzerinde arama işlemi yapıyoruz.
    search_results = qdrant_client.query_points(
        collection_name = collection_name,
        query = question_vector.tolist(),
        limit=k
    )

    # Gelen sonuçların içindeki 'payload'dan metinleri çıkartıp bağlamı buluyoruz
    found_texts = []
    for result in search_results.points:
        found_texts.append(result.payload["text"])
        
    return "\n\n".join(found_texts)