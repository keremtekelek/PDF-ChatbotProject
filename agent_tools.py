from langchain_core.tools import tool
from typing import Union

@tool
def multiply_operation(a: float, b: float) -> float:
    """İki sayıyı çarpmak için bu aracı kullan."""
    return a * b

@tool
def addition_operation(a: float, b: float) -> float:
    """İki sayıyı toplamak için bu aracı kullan."""
    return a + b

@tool
def subtract_operation(a: float, b: float) -> float:
    """İki sayının farkını almak (çıkarma) için bu aracı kullan."""
    return a - b

@tool
def division_operation(a: float, b: float) -> Union[float, str]:
    """İki sayıyı bölmek için bu aracı kullan."""
    if b == 0:
        return "Hata: Sıfıra bölme yapılamaz."
    return a / b

# Temel matematik araçları
base_math_tools = [multiply_operation, addition_operation, subtract_operation, division_operation]

def build_pdf_search_tool(retriever):
    """
    Oluşturulan vektör veritabanı retriever'ını LangChain tool'una dönüştürür.
    """
    @tool
    def search_pdf(query: str) -> str:
        """
        Kullanıcı yüklenen PDF belgeleriyle ilgili bir bilgi sorduğunda veya 
        belgeler içinde veri aramak gerektiğinde bu aracı kullan.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "PDF belgelerinde bu konuyla ilgili eşleşen bilgi bulunamadı."
        
        results = []
        for doc in docs:
            source = doc.metadata.get("source", "Bilinmeyen Belge")
            page = doc.metadata.get("page", 0) + 1
            content = doc.page_content.strip()
            results.append(f"[Kaynak: {source} | Sayfa: {page}]\n{content}")
            
        return "\n\n".join(results)
        
    return search_pdf