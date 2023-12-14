import streamlit as st

st.set_page_config(page_title="使用说明", page_icon="📚")
st.title("How to use")
st.divider()

with st.sidebar:
    st.caption("""RWKV-Ouroboros  
任何使用问题请及时在项目qq群或github项目中反馈。  
QQ群：873610818""")
st.subheader("项目特点")
"1. RWKV-Ouroboros在加载config.py中的RWKV模型后，支持在线全量微调和推理的两种模式，且实现模式之间秒级切换。"
"2. 现阶段UI界面的亮点在于，训练模式中语料的实时编辑十分丝滑。未来将在语料库管理方面继续优化。"

st.divider()

st.subheader("训练技巧")
"1. **语料主题一致**：在train model时，确保输入的语料之间是关联的将显著提高模型效果。更换不同主题语料进行训练时，最好reset state。"
"2. **语料质量**：小批次多频率的语料输入有利于控制模型质量、检测训练效果。"
"3. **语料管理**：区分好训练过的语料和未训练的语料，方便后续的模型管理。"
st.divider()

st.subheader("常见问题")
"1. streamlit版本为1.29.0以上，请更新到较新版本。"
"2. 训练模型时，切换ui界面并不会影响训练进程，也不会丢失信息。但请勿在训练过程中推理（与模型对话），否则会导致api冲突。"
"3. 如果遇到streamlit中无法上传文件，尝试在命令行运行以下命令来重新启动streamlit：  "
st.code("streamlit run WebUI.py --server.enableXsrfProtection=false")
st.divider()

st.subheader("版本更新")
