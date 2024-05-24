import os
from os import listdir
from os.path import isfile, join

mypath = 'training_material'
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

# Filters the files which are in the html format
onlyfiles = [x for x in onlyfiles if 'htm' in x]

# there are plenty of HTMLLoaders to choose from
from langchain_community.document_loaders import UnstructuredHTMLLoader

# the data dictionary will contain the documents
data = {}

i = 0
for file in onlyfiles:
    loader = UnstructuredHTMLLoader(file)
    data[i] = loader.load()
    i += 1

# importing the particular text splitter, in this case we would use TokenTextSplitter
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=25)
# A dictionary which contains all the post split docs.
texts = {}
for i in range(len(data)):
    # The key of texts dictionary is the title of each file
    texts[data[i][0].metadata['source']] = text_splitter.split_documents(data[i])

from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path("environment_variables.env")
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("API_KEY")

for i in range(len(texts)):
    vectordb = Chroma.from_documents(
        # Takes in a list of documents
        text_splitter.split_documents(data[i]),
        # Embedding function, we are using OpenAI default
        embedding=OpenAIEmbeddings(api_key=openai_api_key),
        # Specify the directory where you want these
        persist_directory='./LLM_train_embedding/Doc'
    )

# langchain ParentDocumentRetrieval Packages
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Tells how to split the children
child_splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=10)
# Tells how to split the parent
parent_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=50)

vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings(api_key=openai_api_key)
)
# Creates a document store in memory
store = InMemoryStore()
# This retriever takes vector store and document store
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Adds the child documents and splits them.
for i in range(len(data)):
    retriever.add_documents(data[i])

# Search over documents:
retriever.invoke("Mon endroit préféré")
retriever.invoke("Un ŒIL dans l'espace")
retriever.invoke("L'Incident")
retriever.invoke("La Gentiane de Victorin")
retriever.invoke("Questions CCQ:")

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI

# This is the prompt I used

# It takes in the documents as {context} and user provide {topic}
template = """Mimic the writing style in the context:
{context} and produce a blog on the topic

Topic: {topic}


"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(api_key=openai_api_key)

# Using LangCHain LCEL to supply the prompt and generate output
chain = (
    {
        "context": itemgetter("topic") | retriever,
        "topic": itemgetter("topic"),

    }
    | prompt
    | model
    | StrOutputParser()
)
# running the Chain
docs = chain.invoke({"topic": "Description de l'Expansion du monde industriel"})

print(docs)