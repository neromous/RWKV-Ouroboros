import streamlit as st
import json
import random
import pandas as pd

######## 工具 ##########

def load_json(path, n=10):
    """
    读取jsonl文件
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    messages = [json.loads(x) for x in data]
    return messages

######## 界面 ###########
st.set_page_config(page_title="训练数据处理", page_icon="📚")
st.title("数据处理")
st.caption("数据处理功能测试中：上传文件后，可对数据在线编辑然后保存。建议数据量不要过大。")


st.sidebar.caption("""RWKV-Ouroboros  
  
该功能测试中，仅支持jsonl格式的数据文件。  
该功能旨在为后续的数据管理和数据库检索作基础测试。  
QQ群：873610818""")

mode = st.selectbox("选择文件加载方式", ["拖拽文件", "输入文件路径"])

data = None

# 上传文件
if mode == "拖拽文件":
    uploaded_file = st.file_uploader("上传文件", type=["jsonl"])
    if uploaded_file:
        # 将上传的文件读取为字符串
        stringio = uploaded_file.getvalue().decode("utf-8")
        data = stringio.split("\n")  # 按行分割
        data = [json.loads(x) for x in data if x]  # 转换为json，并排除空行
        st.success("读取成功")

# 本地文件
elif mode == "输入文件路径":
    file_path = st.text_input("输入文件路径",placeholder="例如：/home/xu/liubintao/RWKV-Ouroboros/resources/dialogues/log1.jsonl")
    if file_path:
        data = load_json(file_path)
        st.success("读取成功,预览数据")

# 加载数据后预览文件
if data:
    st.caption("双击可编辑数据，编辑后可保存。（任何更改不会影响原始数据）")
    df = pd.DataFrame(data)
    edited_data = st.data_editor(df,num_rows="dynamic")
    # st.write(type(edited_data))
    json_data = edited_data.to_json()
else:
    with st.container(border=True):
        st.caption("数据编辑区域")
        st.empty()
        ""
        ""
        ""
        ""
        ""
        ""
        ""
        ""
        ""
        ""
st.divider()

# 保存数据
a,b,c= st.columns([2,1,1])
with a:
    save_dir = st.text_input("保存路径", placeholder="./resources/dialogues/log1.jsonl")
    if st.button("保存数据"):
        if json_data and save_dir:
            with open(save_dir, 'w', encoding='utf-8') as f:
                f.write(json_data)
            st.success("保存成功")
        else:
            st.error("保存失败，请检查数据和路径是否正确")