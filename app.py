import streamlit as st
from streamlit_option_menu import option_menu
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize the embeddings model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize the Qdrant client
url = "http://localhost:6333"
client = QdrantClient(
    url=url, prefer_grpc=False
)

# Initialize the Qdrant database
db = Qdrant(client=client, embeddings=embeddings, collection_name="books")

# Load the Hugging Face model and tokenizer
model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
hf = HuggingFacePipeline(pipeline=pipe)

# Streamlit app
st.title("BookFinder: Autonomous LLM-based Book Recommendation Agent")

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
    page = option_menu(
        "Navigation", 
        ["Home", "Generate Response", "Contact Us"],
        icons=["house", "search", "envelope"],
        menu_icon="cast",
        default_index=0,

    )

    st.sidebar.success("This app demonstrates Retrieval-Augmented Generation (RAG) using the Hugging Face Open Source Model.")
    st.sidebar.warning("Developed by [Harsh Kumawat](https://www.linkedin.com/in/harsh-k04/)")


if page == "Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("img.jpeg", width=600, caption="RAG with LLM", use_column_width=True)  # Replace with a valid image URL
    st.subheader("Home")
    st.success("Retrieval-Augmented Generation (RAG) using the Hugging Face Open Source Model.")
    st.write("Welcome to the Chat application. Select 'Generate Response' from the menu to get started.")
    st.write("Using cutting-edge AI technology, BookFinder is an independent LLM-based book recommendation agent. By utilizing cutting-edge models like the Open Source Model from Hugging Face and the BGE-Large-EN embeddings from BAAI, BookFinder provides an advanced retrieval-augmented generation (RAG) capacity. Through a conversational chat interface, users may interact with BookFinder and receive tailored book suggestions based on their inquiries. Semantic search and deep learning models are easily integrated by the application to deliver precise and contextually relevant results. BookFinder was created with the user in mind, combining the effectiveness of AI-driven recommendation algorithms with natural user interfaces to improve the book discovery experience.")
elif page == "Generate Response":
    st.subheader("Start Chatting with the Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]

    # Display all messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Say something"):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Generating response..."):
                # Perform a semantic search
                docs = db.similarity_search_with_score(query=prompt, k=5)

                # Prepare the context for the Hugging Face model
                context = "\n".join([doc.page_content for doc, score in docs])

                # Generate a response using the Hugging Face model
                response = hf.invoke(f"Using the following context, write about {prompt}:\n{context}")

                # Debug: Print response to understand its structure
                st.write("Debugging response structure:", response)

                # Extract the generated text from the response
                if isinstance(response, list) and len(response) > 0:
                    if 'generated_text' in response[0]:
                        full_response = response[0]['generated_text']
                    else:
                        full_response = str(response[0])  # Convert to string for safety
                else:
                    full_response = "Error: Unexpected response format."

                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    if st.button("End Chat"):
        st.session_state.messages.append({"role": "assistant", "content": "Thank you for using the BookFinder application. Have a great day!"})
        st.write("Thank you for using the BookFinder application. Have a great day!")
        st.stop()  # Stop the script to prevent further input

elif page == "Contact Us":
    st.markdown("***")

    st.header("Contact Me")
    st.write("Please fill out the form below to get in touch with me.")

    # Input fields for user's name, email, and message
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message", height=150)

    # Submit button
    if st.button("Submit"):
        if name.strip() == "" or email.strip() == "" or message.strip() == "":
            st.warning("Please fill out all the fields.")
        else:
            send_email_to = 'kumawatharsh2004@gmail.com'
            st.success("Your message has been sent successfully!")
