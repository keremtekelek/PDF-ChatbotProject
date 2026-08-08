from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)

    print(f"Toplam {len(documents)} sayfadan {len(chunks)} adet chunk oluşturuldu.\n")

    return chunks

# Chunk'ları vektöre çevirir ve return olarak vektörleri döndürür.
def embedding(chunk_list, embedding_model):
    print("Metin parçaları vektörlere çevriliyor...")

    #Tüm chunk'ları vektörlere dönüştürdük encode() fonksiyonu ile.
    vectors = embedding_model.encode(chunk_list, show_progress_bar=True)
    return vectors

# Vector veritabınını oluşturur ve return olarak client ve collection_name'i döndürür.
def create_vector_database(vectors, chunk_list):
    print("\n Q-Drant vektör veritabanı oluşturuluyor...")

    # Vektörün boyut sayısını hesaplarız
    vector_size = vectors.shape[1]

    # Q-Drant vektör data base'inin başlatılması kısmı

    #Bizim server ile alakalımız olmadığı için sadece memory'de kullanmak istiyoruz (bkz:faiss gibi).
    client = QdrantClient(":memory:")

    # Q-Drant'taki collection, ilişkisel veritabanlarıdaki tablolar gibi düşünebiliriz. Vektörler ve metadataları depolar.
    # Collection'ımıza unique bir isim veriyoruz.
    collection_name = "first_drant_collection"

    # İsmi verdikten sonra bir collection oluşturuyoruz. İsimi yukarıda verdiğimiz ismi verdik, vektör konfigirasyonunda ise size'ımızı vektör boyutu olarak verdik
    # Vektörler arası Similarity Search metodu olarak da COSINE kullandık (bkz: Cosine-Similarity)
    client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size,  
        distance=models.Distance.COSINE  
    )
    )


    """ 
     Q-Drant'ta temel veri varlıkları noktalardır. Her nokta ID, Vector Data, Payload(Opsiyonel) vardır. Payload'a ek meta-data da diyebiliriz.
     Ayrıca vektörleri Q-Drant'ın anlayabileceği tarz olan points'lere çevirmemiz gerekmektedir.

     Önemli olan diğer bir konu ise FAISS'te sadece vektörleri vermek yeterli olur iken Q-Drant'ta hem chunk'ı hem vektörü vermemizin temel sebebi
     FAISS'in Q-Drant gibi bir vektör veritabanı değildir, sadece vektör arama kütüphanesidir. Q-Drant ise hem vektör arar + bir vektör veritabanıdır + server vardır.

     Aşağıdaki for kodunda olan şey tam olarak şudur:

     Diyelimki elimizde 2 liste var:

     vektorler = [ [0.1, 0.5], [0.8, 0.2] ]
     chunk_list = [ "Elma tatlıdır", "Limon ekşidir" ]

     kod kısmında zip(vektorler, chunk_list) dendiğinde eşleştirir => ( [0.1, 0.5], "Elma tatlıdır" ), ( [0.8, 0.2], "Limon ekşidir" )

     enumerate dediğinizde ise indeks verir => 0, ( [0.1, 0.5], "Elma tatlıdır" ) /// 1, ( [0.8, 0.2], "Limon ekşidir" ) 

     for'un içine geçtiğimizde ise id kısmında indeks (yani yukarıdaki örnekte 0 ya da 1), vector kısmına vektör ([0.1, 0.5]),
     payload kısmına ise örnekteki ("limon ekşidir") kısmı gelir. 

     En sonda da bu her 'nokta'yı boş bir array olan 'noktalar'a ekleriz

    """
    points = []
    for index, (vector, text) in enumerate(zip(vectors, chunk_list)):
        point = models.PointStruct(
            id=index, 
            vector=vector.tolist(), 
            payload={"text": text} 
        )
        points.append(point)

    # Noktaları veritabanına yüklüyoruz.
    client.upsert(
        collection_name=collection_name,
        points=points
    )
   
    print(f"Toplam {len(points)} adet vektör eklendi.\n")

    return client, collection_name

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