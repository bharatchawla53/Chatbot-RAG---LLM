import time
import wikipedia

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login


# get the document from external source
topic = "SpaceX_Mars_colonization_program"
page = wikipedia.page(topic)
text = page.content
doc = Document(page_content=text)

# split the document into smaller chunks for indexing and model training 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents([doc])

# using hugging face open embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# store document in FAISS 
vector_store = FAISS.from_documents(
    documents=all_splits, 
    embedding=embeddings
)

ids = vector_store.add_documents(documents=all_splits)

# use the retrieve from FAISS based on top 3 most similar chunks as context for the LLM based on similarity
retriever = vector_store.as_retriever(search_kwargs={"k": 3, "search_type": "similarity"})

# prompt template help translate input and parameters into instructions for a language model
prompt_template = """
Answer the question using only the information from the context provided below. Do not make up information.
Context: {context}
Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template = prompt_template,
    input_variables = ["context", "question"]
)


login(HUGGING_FACE_TOKEN)

model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                pad_token_id = tokenizer.eos_token_id,
                do_sample=True,
                temperature = 0.1,
                top_p = 0.95,
                repetition_penalty = 1.15,
                max_new_tokens=100)


llm = HuggingFacePipeline(pipeline=pipe)