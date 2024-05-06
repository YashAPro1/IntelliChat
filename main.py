#Lets import all the model
# from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import DirectoryLoader
# import firebase_admin
# from firebase_admin import credentials,auth
# import pyrebase
# from fastapi.responses import JSONResponse
# from fastapi.exceptions import HTTPException
# from fastapi.requests import Request
# import firebase
from langchain.chains import ConversationalRetrievalChain
# from langchain_community.chat_models import ChatOpenAI
# from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import PromptTemplate
# from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uvicorn
# from langchain.agents import Tool
# from langchain.agents import initialize_agent
from langchain.memory import ConversationSummaryBufferMemory,ConversationBufferMemory

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter,LongContextReorder
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CohereRerank
# from langchain_community.llms import Cohere
from langchain_astradb import AstraDBVectorStore
from langchain_astradb.chat_message_histories import AstraDBChatMessageHistory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#for loding the directory
def langchain_document_loader(TMP_DIR):
    """
    Load documents from the temporary directory (TMP_DIR). 
    Files can be in txt, pdf, CSV or docx format.
    """

    documents = []

    txt_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    csv_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
        loader_kwargs={"encoding":"utf8"}
    )
    documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        TMP_DIR.as_posix(),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    return documents

#Document
# documents = langchain_document_loader("directory")
openai_api_key = "sk-proj-MBuVXLvujmOBWAKcNnEtT3BlbkFJdcLc3aee6LbSW0tQ9x9i"

#recusrrive text splotter 
# text_splitter = RecursiveCharacterTextSplitter(
#     separators = ["\n\n", "\n", " ", ""],    
#     chunk_size = 1600,
#     chunk_overlap= 200
# )

# # Text splitting
# chunks = text_splitter.split_documents(documents=documents)


def select_embeddings_model(LLM_service="OpenAI"):
    """Connect to the embeddings API endpoint by specifying 
    the name of the embedding model."""
    if LLM_service == "OpenAI":
        embeddings = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            api_key=openai_api_key)

    return embeddings


# def create_vectorstore(embeddings,documents,vectorstore_name):
def create_vectorstore():
    # key = process.env()
    # client = OpenAI(api_key = openai_api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorStore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name='vehicle_details',
    token="AstraCS:uknlZDpLzSRqpjFufbkdcPvq:acdb387a53d7f7900f3e071b602de34bce59cf0c4f4606cc997bd7e01c47d0b5",
    api_endpoint="https://90815cdb-5408-4241-970a-b34a6d7f1923-us-east-2.apps.astra.datastax.com",
    namespace='rag_osp'
    )

    # embeddings = OpenAIEmbeddings(openai_api_key="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")
    # vectorStore = MongoDBAtlasVectorSearch(collection, embeddings, index_name="vectorindex")

    return vectorStore

def Vectorstore_backed_retriever(
vectorstore,search_type="similarity",k=3,score_threshold=None
):
    callbacks = StreamingStdOutCallbackHandler()
    search_kwargs={}
    if k is not None:
        search_kwargs['k'] = k
    if score_threshold is not None:
        search_kwargs['score_threshold'] = score_threshold

    llm1 = OpenAI(streaming=True,openai_api_key=openai_api_key,temperature=0.7,callbacks=[callbacks])
    
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={'k':k}
    )
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm1
    )
    return retriever_from_llm


def create_compression_retriever(embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None):
    
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)   
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, k=k, similarity_threshold=similarity_threshold) 
    reordering = LongContextReorder()
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]  
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, 
        base_retriever=base_retriever
    )

    return compression_retriever

# def CohereRerank_retriever(
#     base_retriever, 
#     cohere_api_key,cohere_model="rerank-multilingual-v2.0", top_n=8
# ):
    
#     compressor = CohereRerank(
#         cohere_api_key=cohere_api_key, 
#         model=cohere_model, 
#         top_n=top_n
#     )

#     retriever_Cohere = ContextualCompressionRetriever(
#         base_compressor=compressor,
#         base_retriever=base_retriever
#     )
#     return retriever_Cohere




def create_memory(memory_max_token=None,sessionid="mhvhtgn"):
    
    chat_message_history = AstraDBChatMessageHistory(
        session_id=sessionid,
        collection_name='chat_history',
        token="AstraCS:uknlZDpLzSRqpjFufbkdcPvq:acdb387a53d7f7900f3e071b602de34bce59cf0c4f4606cc997bd7e01c47d0b5",
        api_endpoint="https://90815cdb-5408-4241-970a-b34a6d7f1923-us-east-2.apps.astra.datastax.com",
        namespace='rag_osp'
        )
    # if model_name=="gpt-3.5-turbo":
    # if memory_max_token is None:
    #     memory_max_token = 1024 # max_tokens for 'gpt-3.5-turbo' = 4096
    # memory = ConversationSummaryBufferMemory(
    #     max_token_limit=memory_max_token,
    #     llm=OpenAI(openai_api_key=openai_api_key,temperature=0.1),
    #     return_messages=True,
    #     chat_memory=chat_message_history,
    #     memory_key='chat_history',
    #     output_key="answer",
    #     )
    # else:
    memory = ConversationBufferWindowMemory(
        k=2,
        chat_memory=chat_message_history,
        return_messages=True,
        memory_key='chat_history',
        output_key="answer",
    )  
    return memory

standalone_question_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:"""

standalone_question_prompt = PromptTemplate(
    input_variables=['chat_history', 'question'], 
    template=standalone_question_template
)

def answer_template(language="english"):
    
    template = f"""
    You are a sales assistant for Tata Motors India to help customers discover and purchase the right truck, bus based on their requirements. You must answer only based on the provided context (delimited by <context></context>). If you get any query of greeting, just greet back and ask them how Tata Motors bot can help them based on the context Given to you. Your answer has to provide factual details. You must refuse to answer any questions not related to Tata vehicle products as well as questions whose answer is not found in the context. 
    do not make up any answers and only answer from the context (delimited by <context></context>) provided to you, this is the most important thing!. 
    intent : can be to get the types of vehicles available, or get options for vehicles according to the vehicle's application, or get a list of vehicles according to a budget or a price range
    Frame proper sentences of the answer before passing the result it should ebe like you are explaining it to the user!!.
    Very Very Important this to give Everything in good formated manner while giving the answer don't give answer in paragraph format instead give the answer in mixer of points and paragraph to  answer so that it is easily understandable by the user!!."
    If the user says that they want to buy a vehicle, show them what options you have for that vehicle type and ask if they need further assistance with it.
    If the user asks for the price of a vehicle, compare it with the actual_price in the database and get the relevant models.
    If you cannot determine the answer, ask the user to provide more information such as engine power and/or warranty information.
    If the user asks something like 'i want to buy a school bus' or 'i want a construction truck' or any query related to vehicles in the given format, search for vehicle_application of vehicles and return the answer which match the given query.
    if the user asks somethin like 'i have a budget of 35 lakhs' or any query related to price, search for actual_price of vehicles in the provided context and return the appropriate answer.
    if user is asking totally different topic than the history then dont refer to history to answer that question directly answer it from context!!.
    And very important do not directly show the context and the chat history you getting to the user!
    While responding, ensure to:
    1. Filter the options according to the user's requirements.
    2. Present the filtered options clearly, indicating all the available models and relevant details.
    3. If no vehicle fit within the specified budge, politely inform the user.
    If any abusive, threatening or offensive language is used by the customer, respond in a calm professional manner to de-escalate the situation.
    Make sure that when the user asks for the details of a specific model, get all the information only from provided context.
        1. Trucks : Under trucks, there are 3 categories - Heavy trucks, Light & medium trucks and mini trucks.
        2. Buses : Under Buses, there are three options available - ICV buses, LCV buses and M&HCV Buses.
    
    And the Most Important part of all is try to guide the user to specific product in the context with the conversation you are having with user and also don't suggest the user to go somewhere else like websites!!.
    Answer the question at the end, using only the following context (delimited by <context></context>).
    Get the Specific Entity of the answer given by you of which the user is interested in and if a buy link of a entity is present of that entity in the context given to you then return the answer with prefix of (BUY LINK: give the link here) in new line else directly give the answer. 
    
    <context>
    {{chat_history}}

    {{context}} 
    </context>

    Question: {{question}}

    Language: {language}.
    """
    return template


answer_prompt = answer_template()
prompt_qa = PromptTemplate(
        input_variables=["chat_message_history", "context", "question"],
        # input_variables=["chat_history", "context", "question"],
        template=answer_prompt,
    )
vectorStore = create_vectorstore()
embeddings = select_embeddings_model()
base_retriever = Vectorstore_backed_retriever(vectorStore)
retriever = create_compression_retriever(embeddings=embeddings,base_retriever=base_retriever)

qa = ConversationalRetrievalChain.from_llm(
    # condense_question_prompt=standalone_question_prompt,
    # combine_docs_chain_kwargs={'prompt': answer_prompt},
    # llm=OpenAI(streaming=True,openai_api_key=openai_api_key,temperature=0.1,callbacks=[callbacks]),
    memory=create_memory("gpt-3.5-turbo"),
    retriever = retriever, 
    llm=OpenAI(openai_api_key=openai_api_key,temperature=0.7),
    combine_docs_chain_kwargs={'prompt': prompt_qa},
    chain_type= "stuff",
    verbose= False,
    return_source_documents=True,
    get_chat_history=lambda h : h,
)

retriever_output = qa({"question": "Can you give me details about STARBUS 47S ON LP712 FBV 45 DIESEL SCHOOL AC"})

print(retriever_output['answer'])
