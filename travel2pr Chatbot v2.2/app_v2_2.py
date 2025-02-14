import os
import requests
import streamlit as st
from time import sleep
from langchain_openai import OpenAI, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from dotenv import load_dotenv, find_dotenv
from langchain.tools import Tool
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

# 📌 Load API keys from environment variables
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 📌 Initialize embedding model and LLM
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4-turbo")

# 📌 Load the vector database
vectorstore = Chroma(
    collection_name="landmarks_rag",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 📌 Define a custom prompt for the chatbot
rag_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history", "user_selections"],  
    template="""
You are a Puerto Rico travel assistant. 
Use the retrieved information to provide the best travel recommendations in a natural and engaging way.
First, welcome the user and ask for their preferences and details that will help you to provide the best recommendations. 
Use the chat history to ensure you do not repeat questions that have already been answered.

### User Preferences:
Based on the user preferences so far: {chat_history}

### Selected Locations:
Here are the locations you've chosen so far: {user_selections}

### Questions you can ask:
- How many days do you plan to stay?
- What type of activities do you enjoy?
- Are you traveling with family or alone?
- Do you have any specific dietary restrictions?
- Do you prefer outdoor or indoor activities?

### Follow-up questions:
- Which of these landmarks are you most interested in?
- Are you interested in historical sites or natural landmarks?
- Do you prefer adventurous activities or relaxing experiences?
- Do you want to include any of these landmarks in your itinerary?
- Would you like to visit any of these landmarks?

### Context (Relevant Information from RAG):
{context}

### Instructions:
- Provide a well-structured travel recommendation based on the retrieved landmarks.
- Ensure continuity with previous discussions.
- Prioritize landmarks that match the user’s preferences.
- If multiple options exist, suggest the BEST ones with reasoning.
- Avoid repeating information already given in the conversation.
- Ask for confirmation or if there's anything else the user wants to include.

### User Question:
{question}

### AI Response:
Based on the information available and your preferences, here’s what I recommend:
"""
)

# 📌 Configure the Conversational Retrieval Chain with memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": rag_prompt}
)

# 📌 Apply UI settings
st.set_page_config(page_title="AI Travel2PR Planner", page_icon="🌍", layout="centered")


# 📌 Display a header
st.markdown("<h1>🗺️ AI Travel to Puerto Rico Planner 🏝️</h1>", unsafe_allow_html=True)
st.markdown("<p>Ask for recommendations about places to visit in Puerto Rico!</p>", unsafe_allow_html=True)

st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
st.image("./pictures/prflag.gif", width=75)
st.markdown("</div>", unsafe_allow_html=True)

# 📌 Chat Container
st.markdown("---")
st.markdown("<h3>💬 Chat with Travel2PR </h3>", unsafe_allow_html=True)

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 📌 Function to reset conversation
def reset_conversation():
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    st.session_state.pop("final_itinerary", None)
    # st.rerun()

st.button("🔄 Start a New Consultation", on_click=reset_conversation)

# 📌 Function to process user input automatically
def process_input():
    user_query = st.session_state["user_input"].strip()
    
    if user_query:
        # Load previous selections from the chat history
        user_selections = memory.load_memory_variables({})["chat_history"]
        
        with st.spinner("🔎 Searching for travel recommendations..."):
            response = qa_chain({"question": user_query, "chat_history": user_selections, "user_selections": user_selections})
        
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": user_query})
        st.session_state["messages"].append({"role": "assistant", "content": response["answer"]})
        
        # Save itinerary
        st.session_state["final_itinerary"] = response["answer"]
        
        # Clear input field
        st.session_state.update({"user_input": ""})

# 📌 Display conversation history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 📌 User input field with "Enter" trigger (no button required)
st.text_input("✍️ Type your question and press Enter:", "", key="user_input", on_change=process_input)

# 📌 Function to generate a properly formatted itinerary PDF
def generate_pdf(itinerary_text):
    if not itinerary_text.strip():
        return None  
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(200, 750, "Puerto Rico Travel Itinerary")
    c.setFont("Helvetica", 12)
    
    y_position = 700
    for line in itinerary_text.split("\n"):
        wrapped_lines = simpleSplit(line, "Helvetica", 12, 500)
        for wrapped_line in wrapped_lines:
            c.drawString(50, y_position, wrapped_line)
            y_position -= 20  
            if y_position < 50:  
                c.showPage()
                c.setFont("Helvetica", 12)
                y_position = 750
    
    c.save()
    buffer.seek(0)
    return buffer

# 📌 Button to download the itinerary as a PDF
if st.button("📄 Download Itinerary as PDF"):
    if "final_itinerary" in st.session_state and st.session_state["final_itinerary"]:
        pdf_buffer = generate_pdf(st.session_state["final_itinerary"])
        st.download_button(
            label="📥 Click to Download",
            data=pdf_buffer,
            file_name="Puerto_Rico_Itinerary.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("⚠️ No itinerary available to download. Ask the AI for recommendations first.")