from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from typing import List

app = FastAPI()

# Load environment variables
load_dotenv()


# Helper functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Global storage for the conversation chain
conversation_chain = None


# FastAPI Routes
@app.post("/process-pdfs/")
async def process_pdfs(files: List[UploadFile] = File(...)):
    global conversation_chain
    pdf_text = get_pdf_text(files)
    text_chunks = get_text_chunks(pdf_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    return JSONResponse(content={"message": "PDFs processed successfully."})


@app.post("/ask-question/")
async def ask_question(question: str = Form(...)):
    global conversation_chain
    if conversation_chain is None:
        return JSONResponse(content={"error": "Please process the PDFs first."}, status_code=400)

    response = conversation_chain({'question': question})
    chat_history = response['chat_history']

    messages = []
    for i, message in enumerate(chat_history):
        role = "user" if i % 2 == 0 else "bot"
        messages.append({"role": role, "message": message.content})

    return JSONResponse(content={"chat_history": messages})


# Start the server using uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
