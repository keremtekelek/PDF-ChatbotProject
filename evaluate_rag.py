import os
import sys
from unittest.mock import MagicMock

sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

from dotenv import load_dotenv
from datasets import Dataset


# ragas kütüphaneleri
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.run_config import RunConfig

# Kendi kütüphanelerimiz
import llm_manager
import pdf_manager
import data_manager
import chain_manager

# Langchain kütüphaneleri
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

# Test soruları 
test_questions = [
    {
        "question": "Self Attention nedir?",
        "ground_truth": "Self-Attention yani Öz-Dikkat mekanizması, en basit olarak verilen metinde kelimelerin ve belirteçlerin önemini tartmak ve aralarındaki ilişkileri daha iyi anlamak için kullanılır"
    },
    {
        "question": "Logits nedir?",
        "ground_truth": " Sinir ağının son katmanında üretilen ve olasılıklara dönüşmeden önce context vektörlerinden edilen ham, normalleştirilmemiş skorları temsil eder. "
    }
]

def main():
    print("="*50)
    print("  RAGAS DEĞERLENDİRME ")
    print("="*50)

   
    if not llm_manager.start_llm_automatically():
        print("LLM başlatılamadı, çıkılıyor.")
        return

    # PDF klasörünü alıyoruz
    requested_pdf_folder = input("Lütfen testin yapılacağı masaüstündeki PDF klasörünün adını girin (Örn: pdfler): ")
    pdf_folder_path = pdf_manager.search_for_pdf(requested_pdf_folder)
    
    if not pdf_folder_path:
        return

    print("\n RAG çalıştırılıyor.")
    
    # Rag-chain kuruluyor
    embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    pdf_pages = pdf_manager.read_pdf_file(pdf_folder_path)
    chunks = data_manager.chunk(pdf_pages)
    vector_db = data_manager.create_vector_database(chunks, embedding_model)
    retriever = data_manager.create_retriever(vector_db, k=3)
    llm = llm_manager.create_llm(model_name="llama3.1")
    rag_chain = chain_manager.create_rag_chain(retriever, llm)

    # Veri setini dolduruyoruz
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print("\n Test soruları soruluyor.")
    
    for item in test_questions:
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"Soru: {question}")
        
        # Soruyu RAG zincirine gönderiyoruz
        result = rag_chain.invoke({"input": question})
        
        # Sistemin verdiği cevabı alıyoruz
        answer = result["answer"]
        
        # Sistemin o cevabı vermek için okuduğu PDF parçalarını metne çeviriyoruz
        contexts = [doc.page_content for doc in result["context"]]
        
        # Verileri listelere ekliyoruz
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(ground_truth)
        
    
    print("\n Cevaplar Değerlendiriliyor.")
    
    dataset = Dataset.from_dict(data)
    
   
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
    
     
    # Normalde RAGAS, LLM ile kullanılmadığı için hemen cevap alabiliyor. Biz LLM kullandığımız için TimeoutError yiyebiliyoruz.
    # Bu hatayı yememek için genişlik getiriyoruz.
    settings = RunConfig(timeout=600, max_workers=1)

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embedding_model,
        run_config=settings
    )
    
    print("\n" + "="*50)
    print("  DEĞERLENDİRME SONUÇLARI ")
    print("="*50)
    print(results)

if __name__ == "__main__":
    main()