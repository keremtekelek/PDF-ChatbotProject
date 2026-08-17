import llm_manager
from agent_tools import usable_tools
from langchain_core.messages import HumanMessage

def main():
    print("="*50)
    print("   FUNCTION CALLING TESTİ")
    print("="*50)

    if not llm_manager.start_llm_automatically():
        print("LLM başlatılamadı.")
        return
        
    llm = llm_manager.create_llm(model_name="llama3.1")

    # bind_tools ile birlikte Tools'u LLM'e bağlıyoruz.
    llm_with_tools = llm.bind_tools(usable_tools)

    tool_dictionary = {tool.name: tool for tool in usable_tools}
    
    question = "125 ile 4'ü çarpar mısın?"
    print(f"\n Kullanıcı Sorusu: {question}")
    print("Agent Düşünüyor...")

    message = HumanMessage(content=question)
    
    answer = llm_with_tools.invoke([message])

    print("\n--- FUNCTION CALLING ANALIZ SONUCU ---")
    
    if answer.tool_calls:
        print("💡 Agent, fonksiyon çağırdı.")
        for tool_calling in answer.tool_calls:

            tool_name = tool_calling['name']
            tool_args = tool_calling['args']
            
            print(f"- Seçtiği Araç: {tool_name}")
            print(f"- Gönderdiği Parametreler: {tool_args}")
            
            selected_tool = tool_dictionary[tool_name]
            islem_sonucu = selected_tool.invoke(tool_args)
            
            print("-" * 30)
            print(f" İŞLEM SONUCU: {islem_sonucu}")
            print("-" * 30)
            
    else:
        print("Agent araç kullanmaya gerek duymadı. Normal cevap verdi:")
        print(answer.content)

if __name__ == "__main__":
    main()