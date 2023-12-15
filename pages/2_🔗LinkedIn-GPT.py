
# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['GOOGLE_API_KEY'] = apikey
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)

# App framework
st.title('🦜🔗 👉 LinkedIn GPT')
st.subheader('Generate LinkedIn post title and description in just 20 sec. 😎 ')
prompt = st.text_input('Enter Your Prompt') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a LinkedIn title about {topic}'
)

description_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='write me a LinkedIn post Description based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)
hashtag_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='write me a LinkedIn hashtag based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)
# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
description_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
hashtag_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')



# Llms
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
description_chain = LLMChain(llm=llm, prompt=description_template, verbose=True, output_key='script', memory=description_memory)
hashtag_chain = LLMChain(llm=llm, prompt=hashtag_template, verbose=True, output_key='script', memory=hashtag_memory)


wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    description = description_chain.run(title=title, wikipedia_research=wiki_research)
    hashtag = hashtag_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(description)
    st.write(hashtag) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('description History'): 
        st.info(description_memory.buffer)

    with st.expander('hashtag History'): 
        st.info(hashtag_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
