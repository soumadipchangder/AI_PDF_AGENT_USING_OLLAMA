from langchain_core.messages import HumanMessage

# Preferred Ollama integration in newer LangChain versions
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:latest", temperature=0)

messages = [
    HumanMessage(content="What is the capital of France?")
]

response = llm.invoke(messages)
print(response.content)