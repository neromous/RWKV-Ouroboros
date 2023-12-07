import requests
import streamlit as st
from config import config

# 角色列表
role_keys = config["role"].keys()
# 端口
port = config['port']
url = f"http://0.0.0.0:{port}/"

############ 工具定义 ###########

def post_request(route, data):
    r = requests.post(url+route,json=data)
    return r.json()

def check_name_exists(name, state_names):
    if name in state_names:
        return f"保存train state失败：名称'{name}'已存在"
    return None

def send_request_and_handle_response(url, route, name, state_names):
    try:
        response = requests.post(url + route, json={"save_state": name})
        if response.status_code == 200:
            response_data = response.json()
            if "message" in response_data:
                st.success(response_data["message"])
            else:
                st.error("保存train state失败：响应中没有消息")
                st.json(response_data)
        else:
            st.error(f"保存train state失败：服务器返回状态码 {response.status_code}")
            st.write(response.text)
    except requests.exceptions.RequestException as e:
        st.error(f"请求错误：{e}")
    except Exception as e:
        st.error(f"发生错误：{e}")

def main_function(name, url, route, state_names):
    error_message = check_name_exists(name, state_names)
    if error_message:
        st.error(error_message)
    else:
        send_request_and_handle_response(url, route, name, state_names)

#######################

st.set_page_config(page_title="RWKV Chatting", page_icon="🏠")
st.title('RWKV-Ouroboros')
st.write("RWKV-Ouroboros是一个基于RWKV的在线推理与在线训练的框架。其核心理念是’人在回路‘。")

with st.sidebar:
    if st.button('Load Model'):
        r = requests.post(url+"/inference/model/load",json={})
    st.caption("默认加载路径见'config.py'")





mode = st.toggle('推理模式/训练模式', value=False)

if mode:
# ================
# Train Mode
# ================
    st.caption(f"当前为：训练模式")
    with st.container(border = True):
        train_mode = st.radio(label="选择训练模式", options=["tx-data","tokens"],captions=["接收经典数据格式(推荐)","接受分词后token_id的整数列表，加快计算"], horizontal=True,key="train_mode")
        if train_mode == "tx-data":
            route = "/trainer/by/tx-data"
        elif train_mode == "tokens":
            route = "/trainer/by/tokens"

        col11, col22, col33 = st.columns(3)
        with col11:
            max_loss = st.number_input(label="max_loss", value = config['trainer']["max_loss"], key="max_loss")
            min_loss = st.number_input(label="min_loss", value= config['trainer']["min_loss"], key="min_loss")
        with col22:
            min_loss_fix = st.number_input(label="min_loss_fix", value=config['trainer']["min_loss_fix"], key="min_loss_fix")
            max_loss_fix = st.number_input(label="max_loss_fix", value=config['trainer']["max_loss_fix"], key="max_loss_fix")
        with col33:
            ctx_len = st.number_input(label="ctx_len", value=config['model']["ctx_len"], key="ctx_len")
            window = st.number_input(label="window", value=config['trainer']["window"], key="window")
        
        col_A, col_B = st.columns(2)
        with col_A:
            role1 = st.selectbox(label="选择角色1", options=role_keys, key="role1")
            message = st.text_input(label="train data",placeholder="输入训练数据", key="message")
        with col_B:
            role2 = st.selectbox(label="选择角色2", options=role_keys, key="role2")
            message2 = st.text_input(label="train data",placeholder="输入训练数据", key="message2")

        tx_data = { "max_loss": max_loss,
                    "min_loss": min_loss,
                    "min_loss_fix": min_loss_fix,
                    "max_loss_fix": max_loss_fix,
                    "ctx_len": ctx_len,
                    "window": window,
                    "messages":[
                            {"role":role1,
                            "text":f"{message}",
                            "prefix":"",
                            "postfix":"",
                            "prefix_token":config["role"][role1]["prefix"],
                            "postfix_token":config["role"][role1]["postfix"],
                            "response":"",
                            "over": True,
                            "no_loss": False,
                            "mask": 1.0,
                            "role_mask": 1.0,
                            },

                            {"role":role2,
                            ######需要修改message
                            "text":f"{message}",
                            "prefix":"",
                            "postfix":"",
                            # fix token可以自动化
                            "prefix_token":config["role"][role2]["prefix"],
                            "postfix_token":config["role"][role2]["postfix"],
                            "response":"",
                            "over": True,
                            "no_loss": False,
                            "mask": 1.0,
                            "role_mask": 1.0,
                            },
                            ],
                    }
        
        if st.button('Train Model'):
            reset_state = requests.post(url + route,json = tx_data)
            loss = reset_state.json().get("loss")
            if loss:
                st.success(f"loss: {loss}")
            else:
                st.write("训练失败,结果如下：")
                st.write(reset_state.json())

        save_model_dir = st.text_input(label="输入保存模型的名称：", value="default", key="save_model_dir")
        if st.button('Save Model to Disk'):
            r = requests.post(url+"/trainer/model/save-to-disk",json={"save_name" : f"{save_model_dir}"})
        st.caption("默认路径为'./resources/weights/**.pth'")
else:
# ================
# Infer Mode
# ================
    st.caption(f"当前为：推理模式")
    with st.expander("Inference Settings", expanded=True):
        infer_mode = st.radio(label="**选择推理模式：**", options=["tx-data(推荐)","tokens"], horizontal=True ,key="infer_mode")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            temperature = st.number_input(label="temperature", value=0.1, key="temperature")
            top_p = st.number_input(label="top_p", value=0.85, key="top_p")
        with col2:
            alpha_presence = st.number_input(label="alpha_frequency", value=0.2, key="alpha_presence")
            alpha_decay = st.number_input(label="alpha_decay", value=0.996, key="alpha_decay")
        with col3:
            alpha_frequency = st.number_input(label="alpha_frequency", value=0.2, key="alpha_frequency")
            token_count = st.number_input(label="token_count", value=256, key="token_count")
        with col4:
            token_ban = st.number_input(label="token_ban", value=None, key="token_ban")
            token_stop = st.number_input(label="token_stop", value = None, key="token_stop")

        debug = st.checkbox(label="debug模式", value=False, key="debug")
        
        if infer_mode == "tx-data(推荐)":
            route = "/inference/tx-data"
        # elif infer_mode == "messages":
        #     route = "/inference/by/messages"
        elif infer_mode == "tokens":
            route = "/inference/by/tokens"

# ================
# State Process
# ================
with st.expander("State 处理", expanded=False):
    if config["trainer"]["infctx_on"]:
        st.caption("已开启infctx模式")
    else:
        st.caption("未开启infctx模式,不能处理train state")

    # 如果是训练模式，就是trainer的state处理，否则是inference的state处理
    if mode:
        reset_route = "/trainer/state/reset"
        save_route = "/trainer/state/save"
        load_route = "/trainer/state/load"
        to_disk_route = "/trainer/state/save-to-disk"

    else:
        reset_route = "/inference/state/reset"
        save_route = "/inference/state/save"
        load_route = "/inference/state/load"
        to_disk_route = "/inference/state/save-to-disk"

    if st.button('Reset State'):
        requests.post(url+reset_route,json={"messages" : ""})
        st.success("重置train state")

    col_a, col_b = st.columns(2)
    with col_a:
        save_state_name = st.text_input("存储state到内存",placeholder="请输入state名称", key="save_state_name")
        if "state_names" not in st.session_state:
            st.session_state["state_names"] = []
        if st.button("Save State"):
            if save_state_name:
                # 检查是否重名，如果重名，返回错误
                if save_state_name in st.session_state["state_names"]:
                    st.error(f"保存train state失败：名称'{save_state_name}'已存在")
                else:    
                    try:
                        # 发送请求
                        response = requests.post(url + save_route, json={"save_state": f"{save_state_name}"})
                        # 检查响应状态码
                        if response.status_code == 200:
                            # 解析JSON响应
                            response_data = response.json()
                            if "message" in response_data:
                                st.success(response_data["message"])
                                st.session_state["state_names"].append(save_state_name)
                            else:
                                st.error("保存train state失败：响应中没有消息")
                                st.json(response_data)  # 显示完整的响应内容
                        else:
                            st.error(f"保存train state失败：服务器返回状态码 {response.status_code}")
                            st.write(response.text)  # 显示原始响应文本

                    except requests.exceptions.RequestException as e:
                        # 网络请求异常处理
                        st.error(f"请求错误：{e}")
                    except Exception as e:
                        # 其他异常处理
                        st.error(f"发生错误：{e}")
            else:
                st.error("保存train state失败：名称不能为空")

        save_state_dir = st.text_input("存储state到硬盘", placeholder="请输入state名称", key="save_state_dir")
        st.caption("默认保存State到’./resources/states_for_infer/") 
        if st.button('Save State to Disk'):
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
if not mode:

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask something"):

        data={"messages" : [{"role":"question",
                            "text":f"{prompt}",
                            "over": True,
                            "token_count":0,
                            "token_stop": None,
                            },
                            
                            {"role":"answer",
                            "text":"",
                            "over": False,
                            "token_stop": [261],
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
        r = requests.post(url + route,json = data)
        response = r.json()
        answer = response["messages"][1]["response"]


        # 双角色对话
        st.chat_message(roles[1]).write(response["messages"][1]["response"])
        st.session_state.messages.append({"role":roles[1],"content":response["messages"][1]["response"]})
        # st.chat_message(roles[2]).write(response)
        # st.session_state.messages.append({"role":roles[2],"content":response})

        #
        #     # 多角色对话
        #     dialogue = response["messages"][1]["response"]
        #     lines = dialogue.strip().split("\n\n")
        #     lines = [line for line in lines if line.strip() != ""]

        #     for line in lines:
        #         role = "assistant"
        #         line = line.lstrip("<|me|>").rstrip("<|over|>")
        #         parts = line.rsplit("|>",1)
        #         if len(parts) == 2:
        #             pre_text = parts[0].strip()
        #             answer = parts[1].strip()
        #             for key in role_keys:
        #                 if key in pre_text:
        #                     role = key
        #                     break
        #             st.chat_message(role).write(f"||{role}||  " + answer)
        #             st.session_state.messages.append({"role":role,"content":f"||{role}||  "+answer})
        #         else:
        #             for key in role_keys:
        #                 if key in line[0:10]:
        #                     role = key
        #             st.chat_message(role).write(line)
        #             st.session_state.messages.append({"role":role,"content":line})

    if st.button("重置对话"):
        st.session_state["messages"] = []
        st.rerun()





    