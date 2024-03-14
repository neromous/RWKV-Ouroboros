import requests
import streamlit as st
from config import config
import plotly.graph_objects as go
import os
import json
import copy
import pandas as pd
import time

# ------------------utils------------------
def split_data_by_system(messages, chunk_num):
    '''
    在大批量细粒度模式下，将messages切分为chunk_num个。
    如果有system角色，则优先在system之前切分，保证对话的完整性。
    如果没有system角色，则完全按照chunk_num切分。
    '''
    all_chunks = []
    current_chunk = []
    current_count = 0

    # 如果没有system角色，则按照chunk_num切分
    if "system" not in [x["role"] for x in messages[:1000]]: 
        for message in messages:

            # 如果当前块已足够大，且下一条是 'system'，则截断
            if current_count >= chunk_num and message['role'] == 'system':
                all_chunks.append(current_chunk)
                current_chunk = [message]  # 从新的 'system' 开始新块
                current_count = 1
            # 如果当前块不够大
            else:
                # 添加消息到当前块
                current_chunk.append(message)
                current_count += 1

        # 添加最后一个块（如果有）
        if current_chunk:
            all_chunks.append(current_chunk)
        return all_chunks
    # 如果有system角色，则按照优先按照system切分，并考虑chunk_num
    else:
        for message in messages:
            # 如果当前块已足够大，且下一条是 'system'，则截断
            if current_count >= chunk_num and message['role'] == 'system':
                all_chunks.append(current_chunk)
                current_chunk = [message]  # 从新的 'system' 开始新块
                current_count = 1
            # 如果当前块不够大
            else:
                # 添加消息到当前块
                current_chunk.append(message)
                current_count += 1

        # 添加最后一个块（如果有）
        if current_chunk:
            all_chunks.append(current_chunk)

        return all_chunks
# ------------------input------------------
    
# 角色列表
role_keys = config["role"].keys()
# 端口
port = config['port']
url = f"http://0.0.0.0:{port}"
# system prompt
system_prompt = """你是一个名为RWKV的语言模型，你十分聪明，并乐于给予他人关心、帮助他人解决疑惑。
<|request|>通常提出要求、命令和问题。\
<|response|>通常回答问题和解决疑惑。\
注意：我们所有的对话都应该以<|over|>结尾，这是一个停止符。\
"""
# ------------------input------------------

st.set_page_config(page_title="RWKV Chatting", page_icon="🏠")
st.title('RWKV-Ouroboros')
st.caption("""
         RWKV-Ouroboros是一个基于RWKV的在线推理与在线训练的框架。其核心理念是’人在回路‘。  
         项目地址:[RWKV-Ouroboros](https://github.com/neromous/RWKV-Ouroboros)
         """)

colaa, colbb, colcc = st.columns([2,1,2])
colAA, colBB, colCC = st.columns([1,1,1])
with colbb:
    mode = st.toggle('切换模式', value=True,label_visibility="collapsed",help="切换训练/推理模式")
with colBB:
    if mode:
        st.caption("当前为：训练模式")
    else:
        st.caption(f"当前为：推理模式")


# ============= State Process =============
with st.sidebar:
    if mode:
        st.title("Training Settings")
    else:
        st.title("Inference Settings")
        
    with st.expander("高级设置(State处理)", expanded=False):
        # 训练模式，trainer的state
        if mode:
            reset_route = "/trainer/state/reset"
            save_route = "/trainer/state/save"
            load_route = "/trainer/state/load"
            to_disk_route = "/trainer/state/save-to-disk"

        # 推理模式，inference的state
        else:
            reset_route = "/inference/state/reset"
            save_route = "/inference/state/save"
            load_route = "/inference/state/load"
            to_disk_route = "/inference/state/save-to-disk"
    
        if st.button('重置State',help="清空state，在切换对话主题或训练语料主题时很必要"):
            try:
                r = requests.post(url+reset_route,json={"messages" : ""})
                r = r.json()
                if r.get("message"):
                    st.success(f"{r['message']}")
            except requests.HTTPError:
                st.error(f"重置state失败,服务器状态码：{r.status_code}")

        save_state_name = st.text_input("state暂存名称", placeholder="输入state名称,如n_1", key="save_state_name")
        st.session_state.setdefault("state_names", [])
        if st.button("Save", help="将当前的state暂时保存到内存"):
            if save_state_name and save_state_name not in st.session_state["state_names"]:
                try:
                    r = requests.post(url + save_route, json={"save_state": save_state_name})
                    r = r.json()
                    message = r.get("message")
                    if message == "success":
                        st.session_state["state_names"].append(save_state_name)
                        st.success(f"保存state成功")
                    else:
                        st.error(f"保存state失败,请确保state不为初始化状态")
                except requests.HTTPError:
                    st.error(f"服务器返回状态码 {r.status_code}")
            else:
                st.error("保存train state失败：名称为空或已存在")

        load_state_name = st.selectbox("加载暂存的state", options=st.session_state["state_names"], key="load_state_dir")
        if st.button('Load',help="加载暂存在内存中的state"):
            r = requests.post(url+load_route,json={"load_state" : f"{load_state_name}"})
            r = r.json()
            if r.get("message"):
                st.success(f"{r['message']}")
            else:
                st.error("加载train state失败,结果如下：")
                st.write(r)

        save_state_dir = st.text_input("存储state到硬盘", placeholder="请输入state名称", key="save_state_dir")
        if st.button('Save',help="保存state到硬盘，默认保存路径为’./resources/states_for_infer/"):
            r = requests.post(url+to_disk_route,json={"save_state" : f"{save_state_dir}"})

# ===============训练界面==================
if mode:
    # --------------- 0.训练参数 -------------------
    with st.sidebar:
        train_mode = st.selectbox(label="选择训练格式", options=["tx-data(推荐)","tokens(测试中)"],key="train_mode")
        if train_mode == "tx-data(推荐)":
            route = "/trainer/by/tx-data"
        elif train_mode == "tokens(测试中)":
            route = "/trainer/by/tokens"

        with st.container(border = True):
            st.caption("min/max_loss: 用于极化loss，当loss达到阈值，会自动调整学习率。")
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
        "#### 训练数据"
        # 参考数据格式
        with st.expander("数据格式参考", expanded=False):
            st.caption("说明：数据格式应该为jsonl文件，其中每一条json数据对应一个角色的发言：包括role和text两个字段。")
            json_examp = [{"role": "system", "text": "你是一个乐于助人的AI。"},
                        {"role": "request", "text": "你好"},
                        {"role": "response", "text": "我很好，你呢"},
                        {"role": "request", "text": "你叫什么名字"},
                        {"role": "response", "text": "我叫伊芙"},]
            for x in json_examp:
                st.write(str(x))  
        # 选择训练数据上传方式
        st.caption("""**Ouroboros训练模式**会自动为每一条数据添加<|over|>作为结束符，因此不需要手动添加。  
                   普通rwkv模型训练约5k条数据即可学会该框架的special token，提高输出稳定性。""")
        data_mode = st.radio(label="选择训练粒度",
                             index=0, 
                             key="data_mode",
                             options=["小批量粗粒度", "大批量细粒度","小批量手写数据", ],
                             captions=["上传数据：多轮训练，每轮结束后返回平均loss。",
                                       "上传数据：单轮多步训练，按行数切分jsonl数据为多个chunk，返回每个chunk的平均loss。",
                                       "手动编辑数据：多轮训练，每轮结束后返回所有数据的平均loss",
                                       ],)

        if data_mode == "小批量粗粒度":
            # 上传jsonl文件
            st.caption("文件大小限制为200MB，如果上传文件过大，点击左上角“How to use”")
            uploaded_file =  st.file_uploader(label="上传训练数据", type=["jsonl"], key="tx_data")
            if uploaded_file:
                # 读取上传的jsonl文件
                stringio = uploaded_file.getvalue().decode("utf-8")
                json_data = stringio.split("\n")
                json_data = [json.loads(x) for x in json_data if x]
                st.success("读取成功")
                with st.expander("预览数据", expanded=False):
                    data_row = st.number_input("输入要预览第几行数据",value=1,min_value=1,placeholder="例如：10，则预览前10行数据",key="data_row")
                    st.write(json_data[data_row-1:data_row])
            else:
                # 以示例数据json_list为默认值
                json_data = json_examp
                st.caption("文件未上传，训练将默认使用示例数据")

            tx_data = { "max_loss": max_loss,
                        "min_loss": min_loss,
                        "min_loss_fix": min_loss_fix,
                        "max_loss_fix": max_loss_fix,
                        "ctx_len": ctx_len,
                        "window": window,
                        "messages":json_data,}

        elif data_mode == "大批量细粒度":
            # 根据本地路径读取数据
            # file_path = st.text_input("输入jsonl文件路径",value="/home/xu/liubintao/data/openorca_cleaned.jsonl",placeholder="例如：/home/xu/liubintao/RWKV-Ouroboros/resources/dialogues/log1.jsonl")
            # @st.cache_data # 缓存数据，不必每次刷新网页都重新加载
            # def load_json(file_path):
            #     """加载jsonl文件"""
            #     with open(file_path, 'r', encoding='utf-8') as f:
            #             data = f.readlines()
            #     messages = [json.loads(x) for x in data]
            #     return messages

            st.caption("文件大小限制为200MB，如果上传文件过大，点击左上角“How to use”")
            uploaded_file =  st.file_uploader(label="上传训练数据", type=["jsonl"], key="tx_data")
            if uploaded_file:
                # 读取上传的jsonl文件
                stringio = uploaded_file.getvalue().decode("utf-8")
                json_data = stringio.split("\n")
                messages = [json.loads(x) for x in json_data if x]
                st.success("读取成功")
                st.divider()

                "#### 数据切分"
                st.caption("""1、数据切分优先以system角色前切分，并参考chunk_size的大小，以保证对话的完整性。  
                           2、如果没有system角色，则完全按照chunk_size切分。  
                           3、切分后的数据返回一个list，每个元素为一个chunk，chunk内为多条jsonl数据。""")
                if messages is not None:
                    chunk_size = st.number_input("chunk_size: 输入数据分割后每个块的大小",value=30,min_value=1,placeholder="例如：50，则每个chunk包含50条jsonl",help="指每个chunk约包含多少条json数据",key="chunk_size")
                    with st.spinner("正在切分数据..."):
                        chunks = split_data_by_system(messages, chunk_size)
                    st.success("数据切分成功")
                    st.caption(f"共切分为{len(chunks)}个chunk，每个chunk包含约为{chunk_size}条JSONL数据。")
                    # 数据预览
                    with st.expander("预览数据", expanded=False):
                        chunk_id = st.number_input("输入要预览的chunk_id",value=None,min_value=1,placeholder="例如：1，则预览第一个chunk",key="chunk_id")
                        if chunk_id is not None:
                            st.write(chunks[chunk_id-1])

        elif data_mode == "小批量手写数据":
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
                    # 如果路径不存在，则创建新的文件夹
                    if not os.path.exists("./resources/train_data"):
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

    # --------------- 2.训练进程 -------------------
    if data_mode == "小批量粗粒度" or data_mode == "小批量手写数据":
        with st.container(border = True):
            "#### 训练效果"
            st.session_state.setdefault("losses", [])

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
                            yaxis_title='Loss',
                            yaxis=dict(range=[0, None]))
            chart = st.plotly_chart(fig, use_container_width=True)

            col_1, col_2= st.columns([4,1])
            with col_1:
                my_bar = st.progress(0, text="训练进度")
            with col_2:
                if st.button('清空loss绘图'):
                    st.session_state["losses"] = []
                    st.rerun()

            # 训练次数
            train_times = st.number_input(label="训练轮次", value=3, min_value=1, placeholder="请输入训练轮数",key="iter",help="一轮指数据集中所有数据都训练一遍，类似epoch")
            
            col_A,col_B ,col_C, col_D = st.columns([1,1,1,3])
            # 重置进度/暂停训练
            stop_train_placehold = col_B.empty()
            if stop_train_placehold.button("停止训练"):
                st.error("请先开始训练")

            if col_A.button(":red[开始训练]"):
                with st.spinner('Training...'):
                    for i in range(train_times):
                        # 检查是否停止训练
                        if stop_train_placehold.button("停止训练", key=f"stop_train_{i}"):
                            break

                        start_time = time.time()
                        try:
                            r = requests.post(url + route, json=tx_data)
                            r.raise_for_status()
                            loss = r.json().get("loss")
                            st.session_state["losses"].append(loss)
                            # 更新图表数据
                            fig.data[0].x = list(range(1, len(st.session_state["losses"]) + 1))
                            fig.data[0].y = st.session_state["losses"]
                            # 重新绘制图表
                            chart.plotly_chart(fig, use_container_width=True)
                        except requests.HTTPError:
                            st.error(f"第{i+1}次迭代训练失败,服务器返回状态码：{r.status_code}")
                            break

                        end_time = time.time()
                        # 更新进度条
                        my_bar.progress((i+1)/train_times)
                        st.write(f"training...{i+1}/{train_times}， loss: {loss:.4f}, 单轮耗时：{end_time-start_time:.2f}s")
                    else:
                        st.success("训练完成")

    elif data_mode == "大批量细粒度":
        st.session_state.setdefault("iters", [0])
        st.session_state.setdefault("chunk losses", [])

        with st.container(border = True):
            "#### 训练效果"
            if uploaded_file:
                st.caption(f"当前数据训练总体进展: {st.session_state['iters'][-1]}/{len(chunks)} chunks")

            # 初始化Plotly图表
            fig = go.Figure()
            # 检查是否已有数据
            fig.add_trace(go.Scatter(
                x=list(range(1, max(len(st.session_state["chunk losses"]) + 1,3))),
                y=st.session_state["chunk losses"],
                mode='lines+markers',  # 线条+标记
                name='Loss'
            ))
            fig.update_layout(title='Loss损失函数',
                            xaxis_title='训练数据批次',
                            yaxis_title='Loss',
                            yaxis=dict(range=[0, 5]))
            chart = st.plotly_chart(fig, use_container_width=True)

            col_1, col_2= st.columns([4,1])
            with col_1:
                my_bar = st.progress(0, text="训练进度条")
            with col_2:
                if st.button('清空loss绘图'):
                    st.session_state["chunk losses"] = []
                    st.rerun()

            # 训练次数
            num_chunks = st.number_input(label="训练chunk数: ", value=20, placeholder="请输入训练批数",key="num_chunks",help="一次性自动训练的chunk数量")

            col_A,col_B ,col_C, col_D = st.columns([1,1,1,3])
            # 重置进度/暂停训练
            stop_train_placehold = col_B.empty()
            if stop_train_placehold.button("暂停训练"):
                st.error("请先开始训练")
            if col_C.button("重置进度", help="重置数据批次，从第一个chunk开始训练"):
                st.session_state["iters"] = [0]

            # 开始训练/继续训练                
            current_id = st.session_state["iters"][-1]
            if st.session_state["iters"][-1] == 0:
                train_str = "开始训练"
            else:
                train_str = "继续训练"

            if col_A.button(f":red[{train_str}]", help=f"从第{st.session_state['iters'][-1]+1}个chunk开始训练，每个chunk训练完自动reset state"):
                with st.spinner('Training...'):
                    for i, message in enumerate(chunks[current_id:current_id + num_chunks], start=1):
                        start_time = time.time()
                        # 暂停训练
                        if stop_train_placehold.button("暂停训练", key=f"stop_train_{i}"):
                            break

                        tx_data = {
                            "max_loss": max_loss,
                            "min_loss": min_loss,
                            "min_loss_fix": min_loss_fix,
                            "max_loss_fix": max_loss_fix,
                            "ctx_len": ctx_len,
                            "window": window,
                            "messages": message,
                        }

                        try:
                            r = requests.post(url + route, json=tx_data)
                            r.raise_for_status()
                            loss = r.json().get("loss")
                            st.session_state["chunk losses"].append(loss)
                            st.session_state["iters"].append(current_id + i)
                        except requests.HTTPError:
                            st.error(f"第{i}次迭代训练失败,服务器返回状态码：{r.status_code}")
                            break

                        # 更新图表数据
                        fig.data[0].x = list(range(1, len(st.session_state["chunk losses"]) + 1))
                        fig.data[0].y = st.session_state["chunk losses"]
                        chart.plotly_chart(fig, use_container_width=True)

                        # 清除state
                        requests.post(url + reset_route, json={"messages": ""})

                        end_time = time.time()
                        my_bar.progress(i / num_chunks, text=f"当前批次进展{i}/{num_chunks}，当前批次mean-loss: {loss:.4f}，单批耗时：{end_time-start_time:.2f}s")
                    else:
                        st.success("训练完成")

    # 保存模型
    st.caption("""1.训练完成后，无需保存，可以到推理模式立即测试训练效果。  
                2.若要加载保存后的model，需要修改config.py中的load_model路径，然后重启后端。  
                """)
    save_model_dir = st.text_input(label="输入保存模型的名称：", placeholder ="例如default，默认路径为'./resources/weights/", help="默认路径为'./resources/weights/'", key="save_model_dir")

    if st.button('保存model'):
        with st.spinner("正在保存model..."):
            try:
                r = requests.post(url + "/trainer/model/save-to-disk", json={"save_name": f"{save_model_dir}"})
                r.raise_for_status()
                st.success("成功保存模型")
            except requests.HTTPError:
                st.error(f"保存模型失败,服务器状态码：{r.status_code}")
            except requests.RequestException as e:
                st.error(f"保存模型失败，发生了一个错误：{e}")

# ===============推理界面==================
elif not mode:
    # ------------------ 0.推理参数 -------------------
    with st.sidebar:
        infer_mode = st.selectbox(label="**选择推理数据格式**：", options=["tx-data(推荐)","tokens(测试中)"],index=0, key="infer_mode")

        if infer_mode == "tx-data(推荐)":
            infer_route = "/inference/flow-tx"
        # elif infer_mode == "messages":
        #     route = "/inference/by/messages"
        elif infer_mode == "tokens(测试中)":
            infer_route = "/inference/by/tokens"

        with st.container(border = True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state['temperature'] = st.number_input(label="温度", value=st.session_state.get("temperature",0.2), key="temperature_1", help="temperature：L温度越高，生成的文本越随机；温度越低，生成的文本越固定；为0则始终输出相同的内容。")
                st.session_state['token_count']  = st.number_input(label="输出长度", value=st.session_state.get("token_count",512), key="token_count_1",help="token_count：模型一次性回答的最大token数限制。")
                st.session_state['token_ban']  = st.number_input(label="token_ban", value=st.session_state.get("token_ban",None), key="token_ban_1", help="token_ban:整数，使模型避免输出该token_id。一般为空。")
                st.session_state['token_stop']  = st.number_input(label="token_stop", value = st.session_state.get("token_stop",65535), key="token_stop_1", help="停止符:整数，使模型停止输出的token_id。原版rwkv模型为261，Ouroboros框架模型为65535")
            with col2:
                st.session_state['top_p']  = st.number_input(label="top_p", value=st.session_state.get("top_p",0.85), key="top_p_1", help="top_p越高，生成的文本越多样。")
                st.session_state['alpha_presence']  = st.number_input(label="存在惩罚", value=st.session_state.get("alpha_presence",0.2), key="alpha_presence_1", help="alpha_presence:正值鼓励主题多样，负值鼓励主题一致。")
                st.session_state['alpha_frequency']  = st.number_input(label="频率惩罚", value=st.session_state.get("alpha_frequency",0.2), key="alpha_frequency_1", help="alpha_frequency:正值避免重复内容，负值鼓励重复内容。")
                st.session_state['alpha_decay']  = st.number_input(label="惩罚衰减", value=st.session_state.get("alpha_decay",0.995), key="alpha_decay_1", help="alpha_decay:惩罚力度衰减系数。")
            
            temperature = st.session_state['temperature']
            token_count = st.session_state['token_count']
            token_ban = st.session_state['token_ban']
            token_stop = st.session_state['token_stop']
            top_p = st.session_state['top_p']
            alpha_presence = st.session_state['alpha_presence']
            alpha_frequency = st.session_state['alpha_frequency']
            alpha_decay = st.session_state['alpha_decay']

            # 将token_ban和token_stop转换为list
            token_ban = [int(token_ban)] if token_ban else []
            token_stop = [int(token_stop)] if token_stop else []

            debug = st.checkbox(label="debug模式", value=False,help="是否在终端打印state变化", key="debug")
        
    # ------------------ 1.初始化 -------------------
    advance_dialog = st.checkbox(label="自定义对话模式", value=False,help="普通对话模式默认对话角色为request/response。自定义对话模式可自定义对话角色。", key="advance_dialog")

    # 初始化对话记录
    st.session_state.setdefault("messages", [])
    init_prompt_placeholder =  st.empty()
    # 对话记录占位符
    context_placeholder = st.empty().container()
    button_placeholder = st.empty().container()
    with context_placeholder:
        for msg in st.session_state['messages']:
            # 如果msg['content']是列表
            if isinstance(msg['content'], list):
                with st.chat_message("human").container():
                    for content in msg['content']:
                        st.write(content)
            else:
                st.chat_message(msg["role"]).write(msg["content"])
    # ------------------ 2.自定义/普通对话模式-------------------
    # 自定义对话模式
    if advance_dialog:
        # 占为符号
        empty_df = pd.DataFrame(columns=["role","text","over","token_count","token_stop"])
        data_editor = st.data_editor(empty_df,
                                num_rows="dynamic",
                                key="advance_dialog_data",
                                use_container_width=True,
                                disabled=("over"),
                                hide_index=True,
                                column_config={
                                    "_index": st.column_config.NumberColumn(
                                        "index",
                                        help="请确保此列为不同的整数",
                                        default=0,
                                        required=True,
                                        width="small",
                                        ),
                                    "role": st.column_config.SelectboxColumn(
                                        help="从config.py中定义的role中选择",
                                        width="small",
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
        dialog_json_data = data_editor.dropna(how='all').to_json(orient="records", force_ascii=False)
        dialog_json_list = json.loads(dialog_json_data)
        answer_role = st.selectbox("选择model回复时所用角色", options=role_keys,index=4, key="answer_role",placeholder="请选择一个角色（多角色回复测试中）")
        # question/answer user/assistant 是原版rwkv的角色，其停止符强制设置为261，即\n\n
        # token_stop = [261] if "question" or "user" in answer_roles else token_stop
        dialog_json_list.append({"role":answer_role,
                                "text":"",
                                "over": False,
                                "token_stop": [65535],
                                "token_count": token_count,
                                "temperature": temperature,
                                "top_p":top_p,
                                "alpha_frequency":alpha_frequency,
                                "alpha_presence":alpha_presence,
                                "alpha_decay":alpha_decay,
                                })
        data_dialog = {"messages" : dialog_json_list,
                        "debug" : debug,}
        
        all_messages = []

        if st.button("发送对话"):

            # 用户的发言
            with context_placeholder.chat_message("human").container():
                for prompt in data_dialog["messages"]:
                    if prompt["text"] != "":
                        st.write(f'**{prompt["role"].upper()}**: {prompt["text"]}')
                        all_messages.append(f'**{prompt["role"].upper()}**: {prompt["text"]}')

            st.session_state.messages.append({"role":"human","content":all_messages})

            # 模型的反馈结果
            try:
                r = requests.post(url + infer_route, json=data_dialog, stream=True)
                # for role in answer_roles:
                answer = ''
                buffer = b""
                with context_placeholder.empty():
                    for chunk in r.iter_content(chunk_size=1):
                        buffer += chunk
                        try:
                            part = buffer.decode("utf-8")
                            buffer = b""
                            answer += part
                            st.chat_message("assistant").write(f"**{answer_role.upper()}**: {answer}")
                        except UnicodeDecodeError:
                            # 如果解码失败，就继续读取更多的数据
                            continue
                st.session_state.messages.append({"role":"assistant","content":f"**{answer_role.upper()}**: {answer}"})
            except requests.HTTPError:
                st.error(f"服务器返回状态码 {r.status_code}")
                                  
    # 普通对话模式
    else:
        choose_role = ["request","response"]
        if st.session_state.messages == []:
            init_prompt = init_prompt_placeholder.text_input("**SYSTEM PROMPT**", value=None, placeholder=f"默认为：{system_prompt}", key="init_prompt")
            if init_prompt is None or init_prompt.strip() is None:
                init_prompt = system_prompt
        if prompt := st.chat_input("Ask something"):
            # 如果是第一次对话，就将init_prompt加入对话记录
            if st.session_state.messages == []:
                context_placeholder.chat_message("system").write(f"**SYSTEM**: {init_prompt}")
                st.session_state.messages.append({"role": "system", "content":f"**SYSTEM**: {init_prompt}"}) 

                context_placeholder.chat_message("human").write(prompt)
                st.session_state.messages.append({"role": "human", "content":prompt})

                data_dialog={"messages" : [
                                    {"role":"system",
                                    "text":init_prompt,
                                    "over": True,
                                    "token_count":0,
                                    "token_stop": None,
                                    },
                                    {"role":choose_role[0],
                                    "text":prompt,
                                    "over": True,
                                    "token_count":0,
                                    "token_stop": None,
                                    },
                                    {"role":choose_role[1],
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
                
                init_prompt_placeholder.empty()
            else:
                context_placeholder.chat_message("human").write(prompt)
                st.session_state.messages.append({"role": "human", "content":prompt})

                data_dialog={"messages" : [{"role":choose_role[0],
                                    "text":prompt,
                                    "over": True,
                                    "token_count":0,
                                    "token_stop": None,
                                    },
                                    {"role":choose_role[1],
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

            # 模型的反馈结果
            try:
                r = requests.post(url + infer_route,json = data_dialog,stream=True)
                # 流式输出
                answer = ''
                buffer = b""
                with context_placeholder.empty():
                    for chunk in r.iter_content(chunk_size=1):
                        buffer += chunk
                        try:
                            part = buffer.decode("utf-8")
                            buffer = b""
                            answer += part
                            st.chat_message("assistant").write(answer)
                        except UnicodeDecodeError:
                            # 如果解码失败，就继续读取更多的数据
                            continue
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except requests.HTTPError:
                st.error(f"服务器返回状态码 {r.status_code}")
            
    # 清空对话/保存对话
    with button_placeholder.container():
        a, b, c, d = st.columns([1,1,1,3])
        with a:
            if st.button("清空对话", help="清空对话记录并重置state为初始状态"):
                st.session_state["messages"] = []
                r = requests.post(url+reset_route,json={"messages" : ""})
                st.rerun()
        with b:
            if st.button("导出对话", help="将对话记录保存为jsonl,默认路径为‘./resources/dialogues/’"):
                # 检查路径是否存在
                if not os.path.exists("./resources/dialogues"):
                    # 如果不存在，则创建新的文件夹
                    os.makedirs("./resources/dialogues")
                # 将st.session_state中的对话记录以jsonl格式保存，文件名为递增的数字
                with open(f"./resources/dialogues/log_{len(os.listdir('./resources/dialogues'))+1}.jsonl", 'w', encoding='utf-8') as f:
                    for message in st.session_state.messages:
                        # 创建一个新的字典来存储修改后的信息
                        new_message = copy.deepcopy(message)
                        # 在新的字典中重命名 'content' 为 'text'
                        new_message['text'] = new_message.pop('content')
                        json_record = json.dumps(new_message, ensure_ascii=False)
                        f.write(json_record + '\n')
                st.success("保存成功")
        with c:
            if st.button("导入对话"):
                # uploaded_file =  st.file_uploader(label="上传对话", type=["jsonl"], key="dialog")
                # if uploaded_file:
                #     # 读取上传的jsonl文件
                #     stringio = uploaded_file.getvalue().decode("utf-8")
                #     json_data = stringio.split("\n")
                #     json_data = [json.loads(x) for x in json_data if x]
                st.success("功能测试中，请等待更新")
                # else:
                #     st.error("文件未上传")



            