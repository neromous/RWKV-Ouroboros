import requests
import streamlit as st
from config import config
import plotly.graph_objects as go
import os
import json
import copy
import pandas as pd

# 角色列表
role_keys = config["role"].keys()
# 端口
port = config['port']
url = f"http://0.0.0.0:{port}/"

st.set_page_config(page_title="RWKV Chatting", page_icon="🏠")
st.title('RWKV-Ouroboros')
st.caption("""
         RWKV-Ouroboros是一个基于RWKV的在线推理与在线训练的框架。其核心理念是’人在回路‘。  
         项目地址:[RWKV-Ouroboros](https://github.com/neromous/RWKV-Ouroboros)
         """)

colaa, colbb, colcc = st.columns([2,1,2])
with colbb:
    mode = st.toggle('', value=True)
colAA, colBB, colCC = st.columns([1,1,1])
with colBB:
    if mode:
        st.caption("当前为：训练模式")
    else:
        st.caption(f"当前为：推理模式")

if mode:
# ================
# Train Mode
# ================
    with st.sidebar:
        st.title("Training Settings")

        train_mode = st.selectbox(label="选择训练模式", options=["tx-data(推荐)","tokens(测试中)"],key="train_mode")
        if train_mode == "tx-data(推荐)":
            route = "/trainer/by/tx-data"
        elif train_mode == "tokens(测试中)":
            route = "/trainer/by/tokens"

        with st.container(border = True):
            col11, col22 = st.columns(2)
            with col11:
                max_loss = st.number_input(label="max_loss", value = config['trainer']["max_loss"], key="max_loss")
                min_loss = st.number_input(label="min_loss", value= config['trainer']["min_loss"], key="min_loss")
                ctx_len = st.number_input(label="ctx_len", value=config['model']["ctx_len"],help="将输入的训练data切分成的长度", key="ctx_len")
            with col22:
                max_loss_fix = st.number_input(label="max_loss_fix", value=config['trainer']["max_loss_fix"], key="max_loss_fix")
                min_loss_fix = st.number_input(label="min_loss_fix", value=config['trainer']["min_loss_fix"], key="min_loss_fix")
                window = st.number_input(label="window", value=config['trainer']["window"],help="滑动窗口长度，0为不滑动", key="window")


    # --------------- 1.训练数据 -------------------
    with st.container(border = True):
        # st.subheader("1.训练数据")
        # 选择训练数据上传方式
        data_mode = st.selectbox(label="选择数据上传方式",index=1, options=["拖拽文件", "手动编辑数据"],key="data_mode")
        with st.expander("数据格式参考", expanded=False):
            st.caption("说明：数据格式应该为jsonl文件，其中每一条json数据对应一个角色的发言：包括role和text两个字段。")
            json_examp = [{"role": "system", "text": "你是一个乐于助人的AI。"},
                        {"role": "request", "text": "你好"},
                        {"role": "response", "text": "我很好，你呢"},
                        {"role": "request", "text": "你叫什么名字"},
                        {"role": "response", "text": "我叫伊芙"},]
            st.write(json_examp)

        if data_mode == "拖拽文件":
            # 上传jsonl文件
            uploaded_file =  st.file_uploader(label="上传训练数据", type=["jsonl"], key="tx_data")
            if uploaded_file:
                # 读取上传的jsonl文件
                stringio = uploaded_file.getvalue().decode("utf-8")
                json_data = stringio.split("\n")
                json_data = [json.loads(x) for x in json_data if x]

                st.success("读取成功")
                with st.expander("预览数据", expanded=False):
                    st.write(json_data)
            else:
                # 以示例数据json_list为默认值
                json_data = json_examp
                st.caption("文件未上传，默认使用示例数据")

            tx_data = { "max_loss": max_loss,
                        "min_loss": min_loss,
                        "min_loss_fix": min_loss_fix,
                        "max_loss_fix": max_loss_fix,
                        "ctx_len": ctx_len,
                        "window": window,
                        "messages":json_data,
                        }

        elif data_mode == "手动编辑数据":
            # 新建一个空的可编辑的数据表格
            df = pd.DataFrame(columns=["role","text"])
            # 添加一行空数据
            df.loc[0] = ["system",""]
            st.write("编辑数据:")
            # 显示数据表格
            edited_df = st.data_editor(df, 
                                        num_rows="dynamic", 
                                        key="tx_data",
                                        use_container_width=True,
                                        hide_index=False,
                                        column_config={
                                            "_index": st.column_config.NumberColumn(
                                                "index",
                                                help="发言顺序，请确保此列为不同的整数且不为空",
                                                default=None,
                                                required=True,
                                                width="small",
                                                ),
                                            "role": st.column_config.SelectboxColumn(
                                                help="发言角色，从config.py中定义的role中选择",
                                                width="medium",
                                                default=None,
                                                options=role_keys,
                                                required=True,
                                                ),
                                            "text": st.column_config.TextColumn(
                                                help="发言内容，请手动输入",
                                                width="large",
                                                default=None,
                                                required=True,
                                                ),
                                            },
                                         )
            st.caption("""说明：  
1. 务必保证每一行的index为**不同的整数**且**不为空**，否则数据会丢失。  
2. 可以增删表格row，以控制对话的轮数。  
3. 可以自选角色（自定义角色需要编辑config.py）  
4. 表格可全屏显示，方便编辑。
                       """)
            # 删除edited_df中的空行，并将每一行转换为json，所有行合并一个list格式,utf-8格式
            json_data = edited_df.dropna(how='all').to_json(orient="records", force_ascii=False)
            json_list = json.loads(json_data)
            with st.expander(label="数据预览及备份",expanded=False):
                st.write(json_list)

                train_data_dir = st.text_input(label="备份数据的名称：", placeholder ="例如log1(默认保存路径为./resources/train_data/)", key="save_data_dir")
                if st.button("备份数据", help="将当前编辑的数据保存为jsonl文件"):
                    # 检查路径是否存在
                    if not os.path.exists("./resources/train_data"):
                        # 如果不存在，则创建新的文件夹
                        os.makedirs("./resources/train_data")
                    # 将st.session_state中的对话记录以jsonl格式保存
                    with open(f"./resources/train_data/{train_data_dir}.jsonl", 'w', encoding='utf-8') as f:
                        for message in json_list:
                            json_record = json.dumps(message, ensure_ascii=False)
                            f.write(json_record + '\n')
                    st.success("保存成功")
            
            tx_data = { "max_loss": max_loss,
            "min_loss": min_loss,
            "min_loss_fix": min_loss_fix,
            "max_loss_fix": max_loss_fix,
            "ctx_len": ctx_len,
            "window": window,
            "messages":json_list,
            }

    # --------------- 2.训练效果 -------------------
    with st.container(border = True):
        st.subheader("2. 训练效果")
        if "losses" not in st.session_state:
            st.session_state["losses"] = []

        # 初始化Plotly图表
        fig = go.Figure()
        # 检查是否已有数据
        fig.add_trace(go.Scatter(
            x=list(range(1, max(len(st.session_state["losses"]) + 1,3))),
            y=st.session_state["losses"],
            mode='lines+markers',  # 线条+标记
            name='Loss'
        ))
        fig.update_layout(title='Loss损失函数',
                        xaxis_title='训练次数',
                        yaxis_title='Loss')
        chart = st.plotly_chart(fig, use_container_width=True)

        col_1, col_2= st.columns([4,1])
        with col_1:
            my_bar = st.progress(0, text="训练进度")
        with col_2:
            if st.button('清空loss绘图'):
                st.session_state["losses"] = []
                st.rerun()

        # 训练次数
        col_A,col_B = st.columns(2)
        with col_A:
            iter = st.number_input(label="训练次数", value=3, placeholder="请输入训练次数",key="iter")
            if st.button('开始训练'):
                with st.spinner('Training...'):
                    for i in range(iter):
                        r = requests.post(url + route, json=tx_data)
                        if r.status_code == 200:
                            loss = r.json().get("loss")
                            st.session_state["losses"].append(loss)
                            # 更新图表数据
                            fig.data[0].x = list(range(1, len(st.session_state["losses"]) + 1))
                            fig.data[0].y = st.session_state["losses"]
                            # 重新绘制图表
                            chart.plotly_chart(fig, use_container_width=True)

                        else:
                            st.error(f"第{i+1}次迭代训练失败,结果如下：")
                            st.write(f"服务器返回状态码：{r.status_code}")
                            st.write(r.text)
                            break
                        # 更新进度条
                        my_bar.progress((i+1)/iter, text=f"training...{i+1}/{iter}， loss: {(loss):.4f}")
                st.success(f"训练完成")
        with col_B:
            save_model_dir = st.text_input(label="输入保存模型的名称：", placeholder ="例如default", key="save_model_dir")
            if st.button('保存model', help="默认路径为'./resources/weights/**.pth'"):
                r = requests.post(url+"/trainer/model/save-to-disk",json={"save_name" : f"{save_model_dir}"})
                if r.status_code == 200:
                    r = r.json()
                    if r.get("message"):
                        st.success(f"{r['message']}")
                    else:
                        st.error("保存模型失败,结果如下：")
                        st.write(r)
                else:
                    st.error(f"保存模型失败,服务器状态码：{r.status_code}")

# ================
# Infer Mode
# ================
elif not mode:
    with st.sidebar:
        st.title("Inference Settings")
        infer_mode = st.selectbox(label="**选择推理模式：**", options=["tx-data(推荐)","tokens(测试中)"], key="infer_mode")
        with st.container(border = True):
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.number_input(label="temperature", value=0.1, key="temperature", help="温度越高，生成的文本越随机；温度越低，生成的文本越固定；为0则始终输出相同的内容。")
                token_count = st.number_input(label="token_count", value=256, key="token_count")
                token_ban = st.number_input(label="token_ban", value=None, key="token_ban", help="token_ban:使模型避免输出该token。")
                token_stop = st.number_input(label="token_stop", value = None, key="token_stop", help="token_stop:使模型停止输出的token。")
            with col2:
                top_p = st.number_input(label="top_p", value=0.85, key="top_p", help="top_p越高，生成的文本越多样。")
                alpha_presence = st.number_input(label="存在惩罚", value=0.2, key="alpha_presence", help="alpha_presence:正值鼓励主题多样，负值鼓励主题一致。")
                alpha_frequency = st.number_input(label="频率惩罚", value=0.2, key="alpha_frequency", help="alpha_frequency:正值避免重复内容，负值鼓励重复内容。")
                alpha_decay = st.number_input(label="惩罚衰减", value=0.996, key="alpha_decay", help="alpha_decay:惩罚力度衰减系数。")

            debug = st.checkbox(label="debug模式", value=False,help="是否在终端打印state变化", key="debug")
        
        if infer_mode == "tx-data(推荐)":
            route = "/inference/tx-data"
        # elif infer_mode == "messages":
        #     route = "/inference/by/messages"
        elif infer_mode == "tokens(测试中)":
            route = "/inference/by/tokens"

# ================
# State Process
# ================
with st.sidebar:
    with st.expander("高级设置(State 处理)", expanded=False):
        if config["trainer"]["infctx_on"]:
            st.caption("已开启infctx模式")
        else:
            st.caption("未开启infctx模式,不能处理train state")

        # 如果是训练模式，就是trainer的state处理，
        if mode:
            reset_route = "/trainer/state/reset"
            save_route = "/trainer/state/save"
            load_route = "/trainer/state/load"
            to_disk_route = "/trainer/state/save-to-disk"

        # 否则是inference的state处理
        else:
            reset_route = "/inference/state/reset"
            save_route = "/inference/state/save"
            load_route = "/inference/state/load"
            to_disk_route = "/inference/state/save-to-disk"

        if st.button('Reset State',help="清空state为初始状态(根据train/infer模式自动选择train state/infer state)"):
            r = requests.post(url+reset_route,json={"messages" : ""})
            if r.status_code == 200:
                r = r.json()
                if r.get("message"):
                    st.success(f"{r['message']}")
                else:
                    st.error("重置train state失败,结果如下：")
                    st.write(r)

        col_a, col_b = st.columns(2)
        with col_a:
            save_state_name = st.text_input("存储state到内存", placeholder="请输入state名称", key="save_state_name")
            st.session_state.setdefault("state_names", [])

            if st.button("Save State", help="将当前模型的state暂时保存到内存"):
                if save_state_name and save_state_name not in st.session_state["state_names"]:
                    r = requests.post(url + save_route, json={"save_state": save_state_name})

                    if r.status_code == 200 :
                        r = r.json()
                        message = r.get("message")
                        if message == "success":
                            st.success(f"保存state成功")
                            st.session_state["state_names"].append(save_state_name)
                        else:
                            st.error(f"保存state失败,请确保state不为初始化状态")
                    else:
                        st.error(f"服务器返回状态码 {r.status_code}")
                else:
                    st.error("保存train state失败：名称不能为空或已存在")

            save_state_dir = st.text_input("存储state到硬盘", placeholder="请输入state名称", key="save_state_dir")
            if st.button('Save State to Disk',help="默认保存State到’./resources/states_for_infer/"):
                r = requests.post(url+to_disk_route,json={"save_state" : f"{save_state_dir}"})
                
        with col_b:
            load_state_name = st.selectbox("加载内存中的state", options=st.session_state["state_names"], key="load_state_dir")
            if st.button('Load State'):
                r = requests.post(url+load_route,json={"load_state" : f"{load_state_name}"})
                r = r.json()
                if r.get("message"):
                    st.success(f"{r['message']}")
                else:
                    st.error("加载train state失败,结果如下：")
                    st.write(r)
        

# ===============聊天界面==================
# 推理模式
if not mode:
    a, b, c, = st.columns([4,1,1])
    with a:
        advance_dialog = st.checkbox(label="高级对话模式", value=False,help="普通对话模式默认对话角色为2个。高级对话模式可自定义对话角色。", key="advance_dialog")
    with b:
        if st.button("清空对话", help="清空对话记录并重置state为初始状态"):
            st.session_state["messages"] = []
            r = requests.post(url+reset_route,json={"messages" : ""})
            st.rerun()
    with c:
        if st.button("保存对话", help="将对话记录保存为jsonl,默认路径为‘./resources/dialogues/’"):
            # 检查路径是否存在
            if not os.path.exists("./resources/dialogues"):
                # 如果不存在，则创建新的文件夹
                os.makedirs("./resources/dialogues")
            # 将st.session_state中的对话记录以jsonl格式保存
            with open("./resources/dialogues/log1.jsonl", 'w', encoding='utf-8') as f:
                for message in st.session_state.messages:
                    # 创建一个新的字典来存储修改后的信息
                    new_message = copy.deepcopy(message)
                    # 在新的字典中重命名 'content' 为 'text'
                    new_message['text'] = new_message.pop('content')
                    json_record = json.dumps(new_message, ensure_ascii=False)
                    f.write(json_record + '\n')
            st.success("保存成功")
    choose_role_placeholder = st.empty()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    ask_placeholder = st.empty()
    answer_placeholder = st.empty()
    if advance_dialog:
        df_dialog = pd.DataFrame(
            columns=["role","text","over","token_count","token_stop"]
        )
        data_dialog = st.data_editor(df_dialog,
                                num_rows="dynamic",
                                key="advance_dialog_data",
                                use_container_width=True,
                                disabled=("over","token_count","token_stop"),
                                hide_index=False,
                                column_config={
                                    "_index": st.column_config.NumberColumn(
                                        "index",
                                        help="请确保此列为不同的整数",
                                        default=None,
                                        required=True,
                                        width="small",
                                        ),
                                    "role": st.column_config.SelectboxColumn(
                                        help="从config.py中定义的role中选择",
                                        width="medium",
                                        default=None,
                                        options=role_keys,
                                        required=True,
                                        ),
                                    "text": st.column_config.TextColumn(
                                        help="请手动输入训练数据",
                                        width="medium",
                                        default=None,
                                        required=True,
                                        ),
                                    "over": st.column_config.CheckboxColumn(
                                        help="是否结束对话",
                                        width="small",
                                        default=True,
                                        required=False,
                                        ),
                                    "token_count": st.column_config.NumberColumn(
                                        help="token_count",
                                        width="small",
                                        default=0,
                                        required=False,
                                        ),
                                    "token_stop": st.column_config.NumberColumn(
                                        help="token_stop",
                                        width="small",
                                        default=None,
                                        required=False,
                                        ),
                                    },
                                )
        dialog_json_data = data_dialog.dropna(how='all').to_json(orient="records", force_ascii=False)
        dialog_json_list = json.loads(dialog_json_data)
        # st.write(dialog_json_list)
        answer_roles = st.multiselect("选择model回复时所用角色", options=role_keys, default=["answer"],max_selections=1, key="answer_role",help="请选择一个角色（多角色回复测试中）")
        token_stop = [261] if "answer" in answer_roles else [65535]
        for role in answer_roles:
            dialog_json_list.append({"role":role,
                                    "text":"",
                                    "over": False,
                                    "token_stop": token_stop,
                                    "token_count": token_count,
                                    "temperature": temperature,
                                    "top_p":top_p,
                                    "alpha_frequency":alpha_frequency,
                                    "alpha_presence":alpha_presence,
                                    "alpha_decay":alpha_decay,
                                    })
        data_dialog = {"messages" : dialog_json_list,
                        "debug" : debug,}
        # st.write(dialog_json_list)
        if st.button("发送对话"):
            for prompt in data_dialog["messages"]:
                if prompt["text"] != "":
                    st.session_state.messages.append({"role":prompt["role"],"content":prompt["text"]})
                    ask_placeholder.chat_message(prompt["role"]).write(prompt["text"])
            r = requests.post(url + route,json = data_dialog)
            if r.status_code != 200:
                st.error(f"服务器返回状态码 {r.status_code}")
            else:
                response = r.json()
                # answers = response["messages"][1]["response"]
                with st.expander("对话原始结果", expanded=False):
                    st.write(response)
                for msg in response["messages"]:
                    if msg["response"] != "":
                        answer_placeholder.chat_message(msg["role"]).write(msg["response"])
                        st.session_state.messages.append({"role":msg["role"],"content":msg["response"]})
                                  
    else:
        choose_role = choose_role_placeholder.multiselect("选择2个对话角色,注意顺序先问后答", options=role_keys,default=["question","answer"], max_selections=2,key="choose_role")
        if prompt := st.chat_input("Ask something"):
            token_stop = [261] if "answer" in choose_role else [65535]
            data_dialog={"messages" : [{"role":"question",
                                "text":f"{prompt}",
                                "over": True,
                                "token_count":0,
                                "token_stop": None,
                                },
                                {"role":"answer",
                                "text":"",
                                "over": False,
                                "token_stop": token_stop,
                                "token_count": token_count,
                                "temperature": temperature,
                                "top_p":top_p,
                                "alpha_frequency":alpha_frequency,
                                "alpha_presence":alpha_presence,
                                "alpha_decay":alpha_decay,
                                },
                                ],
                    "debug" : debug,}
            roles= ["human","assistant","system"]
            st.session_state.messages.append({"role": roles[0], "content":prompt})
            st.chat_message(roles[0]).write(prompt)

            # 模型的反馈结果
            r = requests.post(url + route,json = data_dialog)
            if r.status_code != 200:
                st.error(f"服务器返回状态码 {r.status_code}")
            else:
                response = r.json()
                answer = response["messages"][1]["response"]

                # 双角色对话
                st.chat_message(roles[1]).write(answer)
                st.session_state.messages.append({"role":roles[1],"content":answer})
 


            