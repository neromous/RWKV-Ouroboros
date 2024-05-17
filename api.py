from app import Robot, config, build_dynic_len_seq
from bottle import route, run, request, response
import types
import gc
import copy
import os

app = Robot()

@route('/speak', method='POST')
def speak():
    req = request.json
    text = req['text']
    prefix = req.get("prefix", "")
    postfix = req.get("postfix", "")
    if text.strip() == "":
        inputs = 11
    else:
        inputs = prefix + text + postfix
    token_stop = req.get('token_stop', [65535])
    void_token = req.get('void_token', 11)
    ctx_len = req.get('ctx_len', 512)
    app.temperature = req.get('temperature', 0.2)
    app.top_p =  req.get('top_p', 0.2)
    app.decay =  req.get("decay", 0.96)
    app.alpha_frequency = req.get('alpha_frequency', 0)
    app.alpha_presence = req.get('alpha_presence', 0)
    print("------------------------------------------------------")
    print(f"\n\n温度:{app.temperature}  top_p:{app.top_p} ctx_len:{ctx_len}")
    print(f"\n\033[93m[主人's Instruction] >>>\033[0m\n{app.infer_tokenizer.clean_sp(text).strip()}")
    print(f"\n\033[92m[Robot Saying] >>> \033[0m")
    if app.current_states is not None:
        state = app.current_states
    else:
        state = None
    speak_tokens, speak_str, out_state = app.speak(
        inputs,
        state,
        n_max_tokens = ctx_len,
        token_stop = token_stop,
        void_token = void_token,
    )
    print("\n\n")
    print("------------------------------------------------------")
    app.current_states = copy.deepcopy(out_state)
    return {"text" : speak_str}


@route('/listen', method='POST')
def listen_by_text(): 
    req = request.json
    inputs = req['text']
    current_states, scratch_token = app.listen(inputs, app.current_states)
    app.current_states = current_states
    inputs = app.infer_tokenizer.clean_sp(inputs)
    print("------------------------------------------------------")
    print(f"\n\n[Robot Listening]>>>  {inputs.strip()}\n\n")
    print("------------------------------------------------------")
    return {
        "text" : "\n",
        "task_done" : True,
        "scrach_tokens" : app.scratch_tokens
        }

@route('/learn/by/text', method='POST')
def learn_by_text():
    req = request.json
    texts = req['text']
    reset_by_step = req.get('reset_by_step', False)
    qa_mask = req.get("qa_mask", True)
    debug = req.get("debug", False)
    assert isinstance(texts ,list)
    assert isinstance(texts[0], str)
    tokens = map(app.infer_tokenizer.encode, texts)  
    tokens = list(tokens)
    if debug:
        print("--------------------------debug----------------------------") 
        print(tokens)
        print(app.train_tokenizer.decode(tokens))
        print("---------------------------debug---------------------------")
    losses = app.learn(tokens, app.current_states, qa_mask=qa_mask, reset_by_step=reset_by_step)
    for text in texts:
        print("------------------------------------------------------")
        print(f"\n\033[93m[Robot Reading]>>> \033[0m\n\n{text.strip()}\n")
        print("------------------------------------------------------") 
    return {"loss": losses}

@route('/reset/state', method='POST')
def reset_state():
    req = request.json
    task_type = req.get('task_type', "default")
    decay = req.get("decay", 1)
    s1_part = req.get("s1_part", 0.5)
    s2_part = req.get("s2_part", 0.5)
    print("reset state: ", task_type)
    if app.current_states is None:
        return {"text" : "当前state状态为空"}
    elif app.ego is None:
        app.current_states = None
        return {"text" : "清空state状态为None"}
    else:
        if task_type == "merge":
            app.reset_currernt_state(
                task_type=task_type,
                s1_part=s1_part,
                s2_part=s2_part)
        elif task_type == "decay":
            app.current_states.decay(decay)
        elif task_type == "clean":
            app.current_states = None
        else:
            if (1 > decay) and (decay > 0) :
                state = copy.deepcopy(app.ego.decay(decay))
            else:
                app.current_states = copy.deepcopy(app.ego)
            
        return {"text" : f"重置类型: {task_type}"}


@route('/read', method='POST')
def read_file():
    req = request.json
    directory = "./resources/book"
    source_type = "path"
    data = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and (filename.endswith('.txt') or filename.endswith('.jsonl')):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                data.append(content)
    losses = []
    for item in data:
        print("====>\n{item[-200:]}")
        token = app.infer_tokenizer.encode(item)
        token = build_dynic_len_seq(token)
        losses += app.learn(token, app.current_states, qa_mask=False)
    return {"losses" :losses}

@route('/save/weight', method='POST')
def save_weight():
    req = request.json
    n = req.get("text", "default")
    if app.args.lora.lora_on:
        app.save_lora_to_disk()
        app.save_state_to_disk()
    else:
        app.save_weight_to_disk(n)
        app.save_state_to_disk(n)
    return { 
        "task_done" : "sucess"
    }

@route('/debug/log', method='POST')
def save_state():
    req = request.json
    n = req.get("text", "default")
    print("=====wkv_states=====>",app.state.wkv_states.shape)
    print("====shift_states====>",app.state.shift_states.shape)
    return { 
        "task_done" : "sucess"
    }

@route('/save/state', method='POST')
def save_state():
    req = request.json
    n = req.get("text", "default")
    app.save_state_to_disk(n)
    return { 
        "task_done" : "sucess"
    }

@route('/load/state', method='POST')
def load_state():
    req = request.json
    n = req.get("text", "default")
    app.load_state_from_disk(n)
    return { 
        "task_done" : "sucess"
    }


if True:
    run(host='0.0.0.0', port=config.port)
