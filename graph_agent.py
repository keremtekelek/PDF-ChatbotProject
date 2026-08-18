import llm_manager
from agent_tools import usable_tools
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

def main():
    print("="*50)
    print("   LANGGRAPH AGENT TESTİ (LOOPS)")
    print("="*50)

    if not llm_manager.start_llm_automatically():
        print("LLM başlatılamadı.")
        return


    # LLM'i oluşturup tool'u bağladık.
    llm = llm_manager.create_llm(model_name="llama3.1")
    llm_with_tools = llm.bind_tools(usable_tools)

    
    graph_builder = StateGraph(MessagesState)

    def chatbot(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    tool_node = ToolNode(tools=usable_tools)

    
    graph_builder.add_node("asistan", chatbot)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "asistan")

    graph_builder.add_conditional_edges(
        "asistan",
        tools_condition,
    )
    
    graph_builder.add_edge("tools", "asistan")

    agent_graph = graph_builder.compile()

    question = "Önce 125 ile 4'ü çarp, sonra çıkan sonuçtan 100 çıkar. Sonuç nedir?"
    print(f"\nKullanıcı Sorusu: {question}\n")
    print("Agent Düşünüyor...\n")

    initial_state = {"messages": [HumanMessage(content=question)]}
    
    for event in agent_graph.stream(initial_state, stream_mode="values"):
        mesaj = event["messages"][-1]
        mesaj.pretty_print() 

if __name__ == "__main__":
    main()