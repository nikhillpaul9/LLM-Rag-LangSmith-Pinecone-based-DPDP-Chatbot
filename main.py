from generation import build_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

def run_pipeline():
    print("Initializing pipeline components...")
    
    rag_chain = build_rag_chain()
    chat_history = []
    
    print("\nStateful DPDP Framework Pipeline ready. LangSmith tracing is active.\n")
    print("-" * 50)
    
    while True:
        # Adjusted the input text to reflect both documents
        query = input("Ask a question about the DPDP Act or Rules (or type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break
            
        print("\nProcessing...\n")
        
        response = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        answer = response["answer"]
        
        print("--- Answer ---")
        print(answer)
        print("-" * 50)
        
        chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=answer)
        ])

if __name__ == "__main__":
    run_pipeline()