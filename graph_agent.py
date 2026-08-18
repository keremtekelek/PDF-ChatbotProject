from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

SYSTEM_PROMPT = """Sen zeki, yetenekli bir Ajan (Agentic AI) ve PDF araştırma asistanısın.
Kullanıcının sorularını cevaplamak için elindeki araçları (tools) kullanabilirsin:

Kurallar:
1. PDF belgesiyle ilgili sorularda veya belgedeki bilgileri aramak gerektiğinde 'search_pdf' aracını kullan.
2. Matematiksel hesaplamalarda veya sayısal işlemlerde kesinlikle tahmin yapma; 'multiply_operation', 'addition_operation', 'subtract_operation', 'division_operation' araçlarını kullan.
3. Soruda hem belge taraması hem hesaplama gerekiyorsa, önce 'search_pdf' ile veriyi bul, ardından çıkan verilerle matematik araçlarını çağır.
4. Bilgi PDF'te yoksa ve genel bir soru değilse "Bu bilgi PDF dosyasında bulunmuyor." de.
5. Akıcı, net ve dilbilgisi kurallarına uygun bir Türkçe ile doğrudan cevap ver.
"""

def create_agent_graph(llm, tools):
    """
    LangGraph StateGraph döngülü ajan grafiğini oluşturur ve derler.
    """
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: MessagesState):
        messages = state["messages"]
        # Sistem promptu yoksa en başa ekle
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools=tools)

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("asistan", chatbot)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "asistan")
    graph_builder.add_conditional_edges("asistan", tools_condition)
    graph_builder.add_edge("tools", "asistan")  # Döngü (Loop): Tool çıktısı asistana geri beslenir

    return graph_builder.compile()