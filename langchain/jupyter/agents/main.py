from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import Chroma

import os
from langchain_openai import OpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool

llm = OpenAI(model_name="gpt-4-turbo", temperature=0, openai_api_base="https://bd-gateway-agent.sdpsg.101.com/openai/v1/", openai_api_key=os.getenv("OPENAI_API_KEY"))
# 导入文档加载器模块，并使用TextLoader来加载文本文件
from langchain_community.document_loaders import TextLoader
# 使用VectorstoreIndexCreator来从加载器创建索引
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

loader = TextLoader('./langchain/jupyter/tests/zj.txt', encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(loader)
embeddings = OpenAIEmbeddings(openai_api_base="https://bd-gateway-agent.sdpsg.101.com/openai/v1/", openai_api_key=os.getenv("OPENAI_API_KEY"))
db = Chroma.from_documents(documents, embeddings)

query = "基础功能组件是什么"
docs = db.similarity_search(query)
print(docs[0].page_content)

embedding_vector = embeddings.embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)



conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=300)
)
conversation.predict(input="组件化是什么？")

'''
# 实例化查询工具
# 更换为自己的 Serp API KEY
os.environ["SERPAPI_API_KEY"] = "541f70b7161c23f1f9ceb34b511383205b1a7c911f070dc1e83b8e34764e324e"
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

# 实例化 SELF_ASK_WITH_SEARCH Agent
self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
'''
