# Personalized Educational Chatbot with PDF Chat Support and Image Generation

import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from deep_translator import GoogleTranslator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
import speech_recognition as sr
# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Sidebar - Student Profile
st.sidebar.title("👤 Student Profile")
name = st.sidebar.text_input("Your Name:", value="Student")
grade = st.sidebar.selectbox("Your Grade Level:", ["Grade 6", "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "Grade 12"])
subjects = st.sidebar.multiselect("Subjects of Interest:", ["Math", "Science", "History", "Geography", "Literature", "Programming"])
goal = st.sidebar.text_input("Your Learning Goal:", value="Improve understanding of basic concepts")
style = st.sidebar.selectbox("Preferred Learning Style:", ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"])
selected_language = st.sidebar.selectbox("Preferred Language:", ["English", "Hindi", "Spanish", "French"])
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
engine = st.sidebar.selectbox("OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# Upload PDFs
st.sidebar.markdown("---")
uploaded_pdfs = st.sidebar.file_uploader("📄 Upload PDF(s) for Q&A", type="pdf", accept_multiple_files=True)

# Title
st.title("📚 Personalized Educational AI Chatbot")
st.write(f"Hello {name}! Ask me anything about your subjects or learning goals.")

# Handle Profile
profile = {
    "name": name,
    "grade": grade,
    "subjects": subjects,
    "goal": goal,
    "style": style
}

# Initialize memory store
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
client = OpenAI(api_key=api_key)
def generate_image_from_prompt(prompt):
    response = client.images.generate(
        model="dall-e-3",  # Or use "dall-e-2" if needed
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    return response.data[0].url
# Setup base prompt and model
llm = ChatOpenAI(model=engine, temperature=temperature)
output_parser = StrOutputParser()
session_id = "student-session-001"

# PDF processing setup
docs = []
if uploaded_pdfs:
    for pdf in uploaded_pdfs:
        temp_path = f"./{pdf.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf.read())
        loader = PyPDFLoader(temp_path)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # History-aware retriever setup
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a user query, rewrite it as a standalone question if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA chain setup
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the context below to answer the user's question in no more than three sentences. Say 'I don't know' if unsure.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    chat_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
else:
    # Default prompt chain when no PDF is uploaded
    system_prompt = """
    You are a personalized educational AI tutor.
    Your job is to help the student named {name}, who is in {grade}, and is interested in {subjects}.
    Their goal is: {goal}, and they prefer a {style} learning style.
    Respond clearly, informatively, and supportively to help them reach their learning objectives.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    chain = prompt | llm | output_parser
    chat_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

# # Image generation function
# def generate_image_from_prompt(prompt):
#     response = openai.Image.create(
#         prompt=prompt,
#         n=1,
#         size="512x512"
#     )
#     return response['data'][0]['url']
st.markdown("🎙️ **Optional Voice Input**")
if st.button("🎤 Start Voice Input"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        voice_input = recognizer.recognize_google(audio)
        st.success(f"You said: {voice_input}")
        user_input = voice_input
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        user_input = ""
    except sr.RequestError:
        st.error("Google Speech Recognition service is unavailable.")
        user_input = ""
else:
    user_input = st.text_input("You:")
# Chat interface
#user_input = st.text_input("You:")

# Image generation option
st.markdown("---")
st.subheader("🖼️ Generate Educational Image")
generate_image = st.checkbox("Generate an image from your prompt")
image_prompt = ""
if generate_image:
    image_prompt = st.text_input("Describe the image you want to generate (e.g. 'photosynthesis diagram'):")

if user_input and api_key:
    translated_input = user_input
    if selected_language != "English":
        translated_input = GoogleTranslator(source=selected_language.lower(), target="en").translate(user_input)

    response = chat_with_memory.invoke(
        {
            "input": translated_input,
            "name": profile["name"],
            "grade": profile["grade"],
            "subjects": ", ".join(profile["subjects"]),
            "goal": profile["goal"],
            "style": profile["style"]
        },
        config={"configurable": {"session_id": session_id}}
    )

    if selected_language != "English":
        final_response = GoogleTranslator(source="en", target=selected_language.lower()).translate(
            response["answer"] if isinstance(response, dict) else response
        )
    else:
        final_response = response["answer"] if isinstance(response, dict) else response

    st.markdown(f"**Bot:** {final_response}")

    if generate_image:
        image_prompt = st.text_input("Enter an image prompt:")
        if st.button("Generate Image") and image_prompt:
            try:
                image_url = generate_image_from_prompt(image_prompt)
                st.image(image_url, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

elif user_input:
    st.warning("Please enter your OpenAI API Key in the sidebar.")