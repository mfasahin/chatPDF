# prompt template 

from langchain_core.prompts import ChatPromptTemplate

QA_PROMPT = ChatPromptTemplate.from_template("""
Sadece PDF içeriğine dayanarak cevap ver.

SOHBET:
{chat_history}

PDF BAĞLAMI:
{context}

SORU:
{question}
""")
