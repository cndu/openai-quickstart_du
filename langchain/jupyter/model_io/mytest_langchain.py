from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate



# 初始化
chat = ChatOpenAI(base_url="https://bd-gateway-agent.sdpsg.101.com/openai/v1", api_key="nd-TDWNsTi0Nl2yg7YHA9L-iz4WUD72OBAnaRs2MB_7",
                  model="gpt-4")

prompt_template = PromptTemplate.from_template(
    "请输入关于{content}，让人{adjective}的主题"
)



memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=chat,
    memory=memory
)



while True:
    print("请输入内容的主题：")
    content = input()
    if content == 'exit' or content == '退出':
        break
    print("请输入内容的引起的情感：")
    adjective = input()
    if adjective == 'exit' or adjective == '退出':
        break
    prompt = prompt_template.format(adjective=adjective, content=content)
    #prompt = "程序员"
    print("请稍等.....")
    response = conversation.predict(input="根据输入的情感和内容主题生成100字以内的笑话，并输出"
                                          "注意不要冷笑话"
                                          "\ninput: " + prompt)
    print("生成结果:\n" + response)