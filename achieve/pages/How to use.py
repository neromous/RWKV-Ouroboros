import streamlit as st

st.set_page_config(page_title="使用说明", page_icon="📚")
st.title("How to use")
st.divider()

with st.sidebar:
    st.caption("""RWKV-Ouroboros  
项目核心和后端代码来自@**纯棉袜子**  
前端来自@**灰不溜湫**  
任何使用问题请及时在项目qq群或github项目中反馈。  
QQ群：873610818""")
    
st.subheader("项目特点")
"1. RWKV-Ouroboros在加载config.py中的RWKV模型后，支持在线全量微调和推理的两种模式，且实现模式之间秒级切换。"
"2. 现阶段UI界面的亮点在于：训练模式中可以对语料实时编辑；可以在服务器运行项目，端口转发手机/笔记本的浏览器远程打开前端。"
"3. 由于本框架在在原版RWKV上增加了special token等技巧以提高模型稳定性，任何RWKV模型都应先在本框架下训练一定的语料（学会specail token），然后才能更好的适应本框架的推理。如果要使用原版rwkv模型，请使用question/answer两个角色进行问答。"
st.divider()

st.subheader("训练技巧")
"1. **关于State**：state是RWKV其特殊的RNN架构所带来的特有性质。在训练模式下，可以通过点击'**reset state**'来重置模型的'训练记忆状态'，这在你切换不同主题的语料时非常重要；在推理模式下，同样可以点击'**reset state**'重置'推理的记忆状态',这在你你决定切换对话主题时非常重要。"
"2. **语料主题一致**：在train model时，确保输入的语料之间是关联的将显著提高模型效果。更换不同主题语料进行训练时，最好reset state。"
"3. **语料质量**：小批次多频率的语料输入有利于控制模型质量、检测训练效果。"
"4. **语料格式**：本框架的语料格式与RWKV原版语料格式有区别：原版为一条json是一轮对话，而本框架是一条json是一个role的发言。"
"""5. **棉袜法**：语料加入了special token来稳定模型的对话。  \
"""
# cola, colb = st.columns([1,10])
# with colb:
st.markdown("**原版rwkv**：以\ n \ n作为停止符，且question/answer没有区别于其他token:\n\n\
    question: 你好\n\n\
    answer: 你好\
    \n\n**Ouroboros框架**：将角色写入的词表避免影响模型性能，加入<|over|>作为替代\ n \ n的停止符: \n\n\
    <|system|>你是一个助人为乐的AI。<|over|>\n\n\
    <|request|>你好<|over|>\n\n\
    <|response|>我很好，有什么能帮到你的吗？<|over|>\n\n\
    <|request|>请你扮演伊芙<|over|>\n\n\
    <|response|>伊芙：好的，我是伊芙，你可以问我任何问题。<|over|>\n\n\
            ")
st.divider()

st.subheader("常见问题")
"1. streamlit版本为1.29.0以上，请更新到较新版本。"
st.code("pip install streamlit --upgrade")
"2. 训练模型时，切换ui界面并不会影响训练进程，也不会丢失信息。但请勿在训练过程中推理（与模型对话），否则会导致api冲突。"
"3. 如果遇到streamlit中无法上传文件，尝试在命令行运行以下命令来重新启动streamlit：  "
st.code("streamlit run WebUI.py --server.enableXsrfProtection=false")
"4. 从前端上传文件的大小限制是200mb，但我想上传更大的文件怎么办？"
"方法1：暂时修改为500mb，在命令行运行以下命令来重新启动streamlit："
st.code("streamlit run WebUI.py --server.maxUploadSize=500")
"方法2：永久修改为500mb，在Streamlit配置文件（config.toml）中设置server.maxUploadSize选项。"
st.code("[server]\n\n\
maxUploadSize = 500\
")
st.divider()

st.subheader("版本更新")
"20231227：引入streamlit-chat，优化了聊天界面 "
"20231225：实现了流式推理，修复了一些bug"
"20231223：修复了角色切换的bug"

