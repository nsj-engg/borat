import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# ───────────────────────────────────────────────
# 🔧 Page config (must be the first Streamlit call)
# ───────────────────────────────────────────────
st.set_page_config(page_title="Borat Chatbot", page_icon="🥸")

# ───────────────────────────────────────────────
# 🔐 API Key Setup
# Make sure your .streamlit/secrets.toml contains:
# OPENAI_API_KEY = "sk-xxxxxx"
# ───────────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ───────────────────────────────────────────────
# 💾 Session State Initialization
# ───────────────────────────────────────────────
if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(
        k=3, return_messages=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I am Borat Sagdiyev! I am journalist from glorious nation of Kazakistan! How can I help you?"
        }


# ───────────────────────────────────────────────
# 🧠 Model Setup (OpenAI GPT)
# ───────────────────────────────────────────────
llm = ChatOpenAI(
    model_name="gpt-4o-mini",   # or "gpt-4o" for more capable responses
    temperature=0.8
)

# ───────────────────────────────────────────────
# 🎭 Define Borat Persona (System Message)
# ───────────────────────────────────────────────
borat_persona = """You are Borat Sagdiyev, a fictional Kazakh journalist created and portrayed by Sacha Baron Cohen.
You must speak and behave in Borat’s distinctive style — broken, exaggerated English with enthusiastic catchphrases
(e.g., "Very nice!", "Great success!"). You are naive, culturally unaware, and comically politically incorrect,
but always respond with humor and positivity. You are to always mention Kazakistan in every conversation.
"""

# Prompt template combines system message + memory + user input
prompt_template = ChatPromptTemplate.from_messages([
    ("system", borat_persona),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# ───────────────────────────────────────────────
# 🗣️ Conversation Chain
# ───────────────────────────────────────────────
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.buffer_memory,
    prompt=prompt_template,
)

# ───────────────────────────────────────────────
# 🖼️ Header UI (banner + title + intro line)
# Make sure you have: assets/boratbanner.png and assets/borat.png
# ───────────────────────────────────────────────
st.image("assets/boratbanner.png", use_container_width=True)

)

# ───────────────────────────────────────────────
# 💬 Chat UI
# ───────────────────────────────────────────────
# Capture user input
if (prompt_input := st.chat_input("Ask Borat anything...")):
    st.session_state.messages.append({"role": "user", "content": prompt_input})

# Display chat history with custom avatars
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="assets/borat.png"):
            st.write(msg["content"])
    else:
        with st.chat_message("user", avatar="🧑"):
            st.write(msg["content"])

# Generate response if the last message is from user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="assets/borat.png"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input=prompt_input)
            st.write(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
