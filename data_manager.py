from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models


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

# VectorStore'u; soruya en yakın chunk'ları getiren bir Retriever'a dönüştürür.
# Ayrıca retriever önceki find_context fonksiyonu gibi tek bir string değil, bir document listesi döndürür.
# source_filter verilirse arama sadece o PDF'in chunk'ları içinde yapılır.
def create_retriever(vector_store, k=3, source_filter=None):

    search_kwargs = {"k": k}

    if source_filter:
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=source_filter)
                )
            ]
        )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    return retriever


# Chunk'ların içinde geçen PDF adlarını alfabetik olarak döner.
def list_sources(chunks):
    return sorted(set(c.metadata["source"] for c in chunks))