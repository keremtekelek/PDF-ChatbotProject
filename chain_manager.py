from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# Bağlam yerine retriever'ın bulduğu chunk'lar yerleşecek.
SYSTEM_PROMPT = """Sen zeki ve yardımcı bir asistansın. Sana verilen 'Bağlam' metnini okuyarak kullanıcının sorusunu cevapla.

Kurallar:
- Sadece bağlamda geçen bilgileri kullan.
- Cevabın bağlamda yoksa kesinlikle kendi bilginden uydurma yapma ve sadece "Bu bilgi PDF dosyasında bulunmuyor." de.
- Kusursuz, akıcı ve dilbilgisi kurallarına uygun bir Türkçe ile cevapla.
- Çeviri kokan veya anlamsız cümleler kurmaktan kaçın.
- Soru birden fazla alt soru içeriyorsa, her birini ayrı ayrı ele al ve hiçbirini atlama.
- Alt sorulardan birinin cevabı bağlamda yoksa, o alt soru için "Bu bilgi PDF dosyasında bulunmuyor." de; diğerlerini normal cevapla.
- "Bağlamda geçen bilgilerden hareketle cevaplayacağım" gibi giriş cümleleri kurma, doğrudan cevaba geç.

Bağlam:
{context}"""


# Retriever ve LLM'i tek bir RAG zincirine bağlar.
def create_rag_chain(retriever, llm):

    # Prompt'u rollere ayırdık.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    # Document listesini tek metne birleştirip prompt'a yerleştiren ve LLM'e gönderen zincir.
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Önce retriever'ı çalıştırıp bulduğu dökümanları document_chain'e besleyen zincir.
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain