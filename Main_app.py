import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings  # Fixed import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

import bs4
import time


from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🍔 BurgerClub AI Assistant",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
        margin: 1rem 0;
    }
    .chat-input {
        font-size: 1.1rem;
    }
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .loading-text {
        color: #FF6B35;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

## load the Chatgpt/groq API key and setting langsmith tracking.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
groq_api_key=os.environ['GROQ_API_KEY']

# Header Section
st.markdown('<h1 class="main-header">🍔 BurgerClub AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent companion for all things BurgerClub India</p>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 🤖 What I Can Help You With:")
    st.markdown("""
    - 🏪 **Store Locations** - Find BurgerClub outlets near you
    - 🍔 **Menu Items** - Explore our delicious burger varieties
    - 📋 **Company Info** - Learn about BurgerClub's story
    - 💰 **Pricing** - Get information about food prices
    - 🕒 **Operating Hours** - Check store timings
    - 🎯 **Special Offers** - Discover current promotions
    """)
    
    st.markdown("### 💡 Sample Questions:")
    sample_questions = [
        "What types of burgers do you serve?",
        "Where can I find BurgerClub stores?",
        "Tell me about BurgerClub's history",
        "What are your popular menu items?",
        "Do you have vegetarian options?"
    ]
    
    for i, question in enumerate(sample_questions, 1):
        if st.button(f"💬 {question}", key=f"sample_{i}"):
            st.session_state.sample_question = question
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Status section
    if "vectors" in st.session_state:
        st.success("🟢 AI Assistant Ready!")

# Initialize vector store with enhanced UI
if "vector" not in st.session_state:
    # Show loading message
    loading_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    loading_placeholder.markdown('<p class="loading-text">🔄 Initializing BurgerClub AI Assistant...</p>', unsafe_allow_html=True)
    
    st.session_state.embeddings=OllamaEmbeddings()
    #st.session_state.embeddings= OpenAIEmbeddings()

    # Load multiple pages and combine them
    all_docs = []
    progress_bar.progress(10)
    status_text.text("📖 Loading BurgerClub website content...")
    
    # Load about page
    loader1 = WebBaseLoader(web_paths=("https://theburgerclub.in/about",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("common-para")
                     )))
    try:
        docs1 = loader1.load()
        all_docs.extend(docs1)
        progress_bar.progress(30)
        status_text.text(f"✅ Loaded company information ({len(docs1)} documents)")
    except Exception as e:
        st.warning(f"⚠️ Could not load about page: {e}")
    
    # Load store locator page
    loader2 = WebBaseLoader(web_paths=("https://theburgerclub.in/store-locator",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("font-weight-light","mt-2")
                     )))
    try:
        docs2 = loader2.load()
        all_docs.extend(docs2)
        progress_bar.progress(50)
        status_text.text(f"✅ Loaded store locations ({len(docs2)} documents)")
    except Exception as e:
        st.warning(f"⚠️ Could not load store locator: {e}")
    
    # Load menu page
    loader3 = WebBaseLoader(web_paths=("https://theburgerclub.in/order/the-burger-club-rani-bagh-delhi",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("wla-outlet-name-md","item-title","heading-customize more30857606","price-p")
                     )))
    try:
        docs3 = loader3.load()
        all_docs.extend(docs3)
        progress_bar.progress(70)
        status_text.text(f"✅ Loaded menu information ({len(docs3)} documents)")
    except Exception as e:
        st.warning(f"⚠️ Could not load menu page: {e}")
    
    # If no documents were loaded, try a simpler approach
    if not all_docs:
        status_text.text("🔄 Trying alternative loading method...")
        try:
            simple_loader = WebBaseLoader(web_paths=("https://theburgerclub.in/about",))
            all_docs = simple_loader.load()
            progress_bar.progress(80)
            status_text.text(f"✅ Loaded website content ({len(all_docs)} documents)")
        except Exception as e:
            st.error(f"❌ Failed to load any documents: {e}")
            st.stop()
    
    st.session_state.docs = all_docs
    
    # Check if we have documents before proceeding
    if not st.session_state.docs:
        st.error("❌ No documents were loaded. Please check the website URLs and try again.")
        st.stop()
    
    # Check if documents have content
    content_docs = [doc for doc in st.session_state.docs if doc.page_content.strip()]
    if not content_docs:
        st.error("❌ All loaded documents are empty. Please check the CSS selectors.")
        st.stop()
    
    progress_bar.progress(85)
    status_text.text("🔄 Processing and splitting documents...")
    
    # Optimized text splitting for faster processing
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,    # Smaller chunks for faster processing
        chunk_overlap=100  # Reduced overlap for speed
    )
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(content_docs[:30])  # Process fewer documents
    
    # Check if we have final documents
    if not st.session_state.final_documents:
        st.error("❌ No documents after text splitting. Check document content.")
        st.stop()
    
    progress_bar.progress(90)
    status_text.text(f"🧠 Creating AI knowledge base with {len(st.session_state.final_documents)} chunks...")
    
    try:
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
        progress_bar.progress(100)
        status_text.text("🎉 BurgerClub AI Assistant is ready!")
        
        # Clear loading elements and show success
        time.sleep(1)
        loading_placeholder.empty()
        progress_bar.empty()
        status_text.empty()
        
        # Show balloons only once when vector store is created
        if "balloons_shown" not in st.session_state:
            st.balloons()
            st.session_state.balloons_shown = True
        
        st.success("🎯 AI Assistant ready! Ask me anything about BurgerClub.")
        
    except Exception as e:
        st.error(f"❌ Error creating vector store: {e}")
        st.info("💡 Make sure Ollama is installed and running for embedding generation.")
        st.stop()

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🍔 **Welcome to BurgerClub!**
    
    I'm your friendly AI assistant, here to help you discover everything about BurgerClub India.
    
    **I can help you with:**
    - 🏪 Store locations and contact details
    - 🍔 Menu items and pricing information  
    - 📖 Company history and brand story
    - 🕒 Operating hours and special offers
    - 🌱 Nutritional information and ingredients
    
    *Ask me anything about our delicious burgers and services!*
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize the LLM with better configuration
llm=ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.1,  # Lower temperature for faster, more consistent responses
    max_tokens=300    # Reduced token limit for faster response
)

# Enhanced prompt template
prompt=ChatPromptTemplate.from_template(
"""
You are the official AI customer service representative for "The Burger Club" - India's premium burger restaurant chain. 

Your role is to provide helpful, accurate, and friendly assistance to customers based EXCLUSIVELY on the provided context from the company's official website.

IMPORTANT GUIDELINES:
- Always maintain a warm, professional, and enthusiastic tone
- Provide specific, actionable information when available
- If you're not certain about information, politely direct customers to official channels
- Never invent or assume details not present in the context
- Keep responses concise but comprehensive
- Always end with a helpful next step or website reference

RESPONSE STRUCTURE:
1. Acknowledge the customer's question warmly
2. Provide accurate information from the context
3. Offer additional relevant details if available
4. Direct to website/contact for further assistance

<context>
{context}
</context>

Customer Question: {input}

Response (as BurgerClub's official AI assistant):
"""
)

# Only create chains if vector store exists
if "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 2})  # Get top 2 relevant chunks for faster response
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Enhanced chat interface
    st.markdown("### 💬 Chat with BurgerClub AI Assistant")
    
    # Check if there's a sample question from sidebar or quick action
    preset_value = ""
    if "sample_question" in st.session_state:
        preset_value = st.session_state.sample_question
        del st.session_state.sample_question
    elif "quick_question" in st.session_state:
        preset_value = st.session_state.quick_question
        del st.session_state.quick_question
    
    if preset_value:
        prompt_input = st.text_input(
            "Ask me anything about BurgerClub:", 
            value=preset_value,
            key="main_input"
        )
    else:
        prompt_input = st.text_input(
            "Ask me anything about BurgerClub:", 
            placeholder="e.g., What types of burgers do you serve?",
            key="main_input"
        )

    if prompt_input:
        with st.spinner("🤔 Finding the best answer for you..."):
            try:
                response = retrieval_chain.invoke({"input": prompt_input})
                
                # Display response with clean formatting
                st.markdown("### 🤖 **BurgerClub Assistant:**")
                st.markdown(f"{response['answer']}")
                
                st.markdown("*Need more info? Visit [theburgerclub.in](https://theburgerclub.in)*")
                        
            except Exception as e:
                st.error(f"❌ Sorry, I encountered an error: {str(e)}")
                st.info("💡 Please try rephrasing your question or check if the knowledge base is properly loaded.")
                
    # Quick action buttons
    st.markdown("### 🔥 **Quick Actions**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏪 Find Stores"):
            st.session_state.quick_question = "Where are BurgerClub stores located?"
            st.rerun()
    
    with col2:
        if st.button("🍔 View Menu"):
            st.session_state.quick_question = "What food items are available?"
            st.rerun()
    
    with col3:
        if st.button("💰 Check Prices"):
            st.session_state.quick_question = "What are the prices of burgers?"
            st.rerun()
    
    with col4:
        if st.button("📞 Contact Info"):
            st.session_state.quick_question = "How can I contact BurgerClub?"
            st.rerun()

else:
    st.error("❌ Vector store not initialized. Please refresh the page to load the knowledge base.")
    if st.button("🔄 Refresh Page"):
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1.5rem 0;'>
    <h4>🍔 BurgerClub India</h4>
    <p>🔗 <a href="https://theburgerclub.in" target="_blank">Visit Official Website</a> | 
       � Contact Customer Support</p>
    <p><em>Your AI assistant for all BurgerClub information and queries.</em></p>
</div>
""", unsafe_allow_html=True)
