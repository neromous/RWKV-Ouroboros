from prompts.messages import Message
import types
import gc
import copy
import torch
from bottle import route, run, request, response
import deepspeed
from config import config

args = types.SimpleNamespace()


for k, v in config['model'].items():
    setattr(args, k, v)

for k, v in config['trainer'].items():
    setattr(args, k, v)

tokenizer = config['inference']['tokenizer']

# ================
# init state model
# ================
train_model = None
train_state = None
train_init_state = None
train_state_map = {}
# ================
training_step = 0

# ================
rnn_model = None
infer_state = None
infer_init_state = None
infer_state_map = {}
debug = config['debug']
pos_state = None
neg_state = None
# ================
# load model
# ================

if args.infctx_on:
    from models.v5.model import RWKV
else:
    from models.v5.origin import RWKV
train_model = RWKV(args)
from models.v5.runner import RWKV_RNN


optimizer, lr_scheduler = train_model.get_optimizers()
model_engine, optimizer, _, _ = deepspeed.initialize(model=train_model,
                                                     optimizer=optimizer,
                                                     lr_scheduler=lr_scheduler,
                                                     config=args.ds_config)

# ================
# train-state
# ================

if args.infctx_on:
    @route('/trainer/state/reset', method='POST')
    def clean_train_state():
        global train_state
        gc.collect()
        torch.cuda.empty_cache()
        # req = dict(request.json)
        train_state = None
        # ================
        gc.collect()
        torch.cuda.empty_cache()
        return {"message": "success"}

    @route('/trainer/state/save', method='POST')
    def save_train_state():
        global train_state, train_state_map
        req = dict(request.json)
        s_name = req.get('save_state', False)
        if s_name and train_state is not None:
            train_state_map[s_name] = copy.deepcopy(train_state).cpu()
        return {"message": "success"}

    @route('/trainer/state/load', method='POST')
    def load_train_state():
        global train_state, train_state_map
        req = dict(request.json)
        s_name = req.get('load_state', False)
        if s_name:
            train_state = train_state_map.get(s_name, None)
        else:
            return {"message": "fail"}
        if train_state is not None:
            train_state = train_state.cuda()
        return {"message": "success"}

    @route('/trainer/state/save-to-disk', method='POST')
    def train_state_to_disk():
        global train_state, train_state_map
        req = dict(request.json)
        s_name = req.get('save_name', "v5-3b-81.pth")
        fpath = f"./resources/states_for_train/{s_name}.pth"
        torch.save(copy.deepcopy(train_state_map), fpath)
        return {"message": "save success"}


@route('/trainer/model/save-to-disk', method='POST')
def train_model_to_disk():
    global model_engine, train_model
    req = dict(request.json)
    s_name = req.get('save_name', "default")
    fpath = f"./resources/weights/{s_name}.pth"
    # ================
    gc.collect()
    torch.cuda.empty_cache()
    # ================
    train_model.load_state_dict(model_engine.module.state_dict())
    torch.save(train_model.state_dict(), fpath)
    # ================
    gc.collect()
    torch.cuda.empty_cache()
    return {"message": "save success"}

# ================
# train model
# ================
@route('/trainer/by/tx-data', method='POST')
def train_by_tx_data():
    global train_state,training_step, debug
    req = dict(request.json)
    min_loss = req.get('max_loss', args.min_loss)
    max_loss = req.get('min_loss', args.max_loss)
    min_loss_fix = req.get('min_loss_fix', args.min_loss_fix)
    max_loss_fix = req.get('max_loss_fix', args.max_loss_fix)
    ctx_len = req.get('ctx_len', args.ctx_len)
    window = req.get('window', args.window)
    messages = req.get('messages', [])
    # messages中的每个message都对应一个角色的发言和参数，是dict
    # Message.new()指对mesages中的每个message进行处理，返回一个Message类对象，属性是message中的参数
    # messages是[([tokens1],[masks1]),([tokens2],[masks2])，([tokens3],[masks3])]
    # tokens1是一个list，里面是token的id，masks1是一个list，里面是1或0
    messages = [Message.new(x) for x in messages]
    messages = [x.tokens(for_infer=False) for x in messages]
    tokens = []
    masks = []
    for token, mask in messages:
        tokens += token
        masks += mask
    if len(tokens) == 0:
        return {"loss": 0.0}

    tokens = tokens + [0]
    masks[-1] = 0
    # 此处的tokens是一个list，masks是一个list，里面是1或0
    if len(tokens) == 0:
        return {"response": "no tokens"}
    total = 0
    mean_loss = 0
    i = 0
    if train_state is not None:
        states = copy.deepcopy(train_state)
    else:
        states = train_state
    debug = req.get('debug', debug)
    if debug:
        print("-train-state-before>", states)
        print("-tokens->", tokens)
    # 每次循环截取tokens的0-ctx_len部分,window为滑动窗口的重叠长度
    # 每个被截取的部分都是一个batch、单独进行前向传播和反向传播
    while len(tokens) > 1:
        # training_step += 1
        i += 1
        output = tokens[:ctx_len + 1]
        output_masks = masks[:len(output) - 1]
        # add the token to keep the state have  last one token
        tokens = tokens[ctx_len - window:]
        masks = masks[ctx_len - window:]
        batch = {'input_ids': output,
                 'attention_mask': output_masks}
        # deepspeed的model_engine是一个函数，输入batch和states，前向传播，返回loss和states
        m, states = model_engine(batch, states=states)
        loss = m.item()
        # 修正loss，loss过大则夸大，过小则更小
        if loss < min_loss:
            m = m * min_loss_fix
        elif loss > max_loss:
            m = m * max_loss_fix
        model_engine.backward(m)
        model_engine.step()
        total += loss
        mean_loss = total / i
        print(f"\nmean-loss->{mean_loss}")
    if states is not None:
        train_state = copy.deepcopy(states)
    else:
        train_state = states
    if debug:
        print("-train-state-after>", states)
    gc.collect()
    torch.cuda.empty_cache()
    return {"loss": mean_loss}


@route('/trainer/by/tokens', method='POST')
def train_by_tokens():
    global train_state, debug, step
    req = dict(request.json)
    min_loss = req.get('max_loss', args.min_loss)
    max_loss = req.get('min_loss', args.max_loss)
    min_loss_fix = req.get('min_loss_fix', args.min_loss_fix)
    max_loss_fix = req.get('max_loss_fix', args.max_loss_fix)
    ctx_len = req.get('ctx_len', args.ctx_len)
    window = req.get('window', args.window)
    fix_logit = req.get('fix_logit', 1)
    attention_mask = req.get('attention_mask', None)
    all_tokens = req.get('tokens', [])
    if len(all_tokens) == 0:
        return {"response": "no tokens"}
    losses = []
    total = 0
    mean_loss = 0
    i = 0
    step = 0
    if train_state is not None:
        states = copy.deepcopy(train_state)
    else:
        states = train_state
    if debug:
        print("-train-state-before>", states)
    for tokens in all_tokens:
        # 增加一个[0] 以对齐最后一个token,避免丢失
        if attention_mask is None:
            masks = [fix_logit for x in tokens]
        else:
            masks = [x * fix_logit for x in attention_mask]
        tokens = tokens + [0]
        masks[-1] = 0
        while len(tokens) > 1:
            i += 1
            step += 1
            # 修正最后一个token不进入state的问题。
            output = tokens[:ctx_len + 1]
            tokens = tokens[ctx_len - window:]
            output_masks = masks[:len(output) -1]
            masks = masks[ctx_len - window:]
            # 组装结果
            batch = {'input_ids': output,
                     'attention_mask': output_masks}
            m, states = model_engine(batch, states=states)
            loss = m.item()
            if loss < min_loss:
                m = m * min_loss_fix
            elif loss > max_loss:
                m = m * max_loss_fix

            model_engine.backward(m)
            model_engine.step()

            total += loss
            mean_loss = total / i
            print(f"\nfix-logit->{fix_logit}, total-loss->{total}")
            print(f"round->{step}, mean-loss->{mean_loss}")
            print(f"current->{loss}")
        losses.append(mean_loss)
    if states is not None:
        train_state = copy.deepcopy(states)
    else:
        train_state = states
    if debug:
        print("-train-state-after>", states)
    gc.collect()
    torch.cuda.empty_cache()
    return {"losses": losses}


# ================
# infer-state
# ================


@route('/inference/state/reset', method='POST')
def clean_infer_state():
    global infer_state , pos_state, neg_state
    req = dict(request.json)
    infer_state = None
    pos_state = None
    neg_state = None
    return {"message": "success"}


@route('/inference/state/save', method='POST')
def save_infer_state():
    global infer_state, infer_state_map
    req = dict(request.json)
    s_name = req.get('save_state', False)
    #***
    if s_name and infer_state is not None:
        infer_state_map[s_name] = copy.deepcopy(infer_state).cpu()
        return {"message": "success"}
    else:
        return {"message": "fail"}


@route('/inference/state/load', method='POST')
def load_infer_state():
    global infer_state, infer_state_map
    req = request.json
    req = dict(request.json)
    s_name = req.get('load_state', False)
    if s_name:
        infer_state = infer_state_map.get(s_name, None)
        infer_state = infer_state.cuda()
        return {"message": "success"}
    else:
        return {"message": "fail"}

@route('/inference/state/save-to-disk', method='POST')
def infer_state_to_disk():
    global infer_state, infer_state_map
    req = request.json
    req = dict(request.json)
    s_name = req.get('save_name', "default-states")
    fpath = f"./resources/states_for_inference/{s_name}.pth"
    torch.save(copy.deepcopy(infer_state_map), fpath)
    return {"message": "success"}


@route('/inference/model/load', method='POST')
def infer_model_load():
    global rnn_model, infer_state
    req = request.json
    req = dict(request.json)
    rnn_model = RWKV_RNN(model_engine.module.state_dict(), args)
    infer_state = None
    return {"message": "success"}

# ================
# infer model
# ================
@route('/inference/tx-data', method='POST')
def infer_by_tx_data():
    global rnn_model, infer_state, debug, pos_state, neg_state
    req = request.json
    debug = req.get('debug', debug)
    if rnn_model is None:
        print(f"===load model===")
        infer_model_load()
    messages = req['messages']
    result = []
    state = None
    if infer_state is not None:
        state = copy.deepcopy(infer_state)
    if pos_state is not None:
        pos_state = copy.deepcopy(pos_state)
    if neg_state is not None:
        neg_state = copy.deepcopy(neg_state)
    if debug:
        print("--before--->", state)
    for message in messages:
        infer_config = {"temperature": message.get("temperature", 0.2),
                        "top_p": message.get('top_p', 0.2),
                        "token_count": message.get('token_count', 256),
                        "token_stop": message.get('token_stop', []),
                        "alpha_presence": message.get('alpha_presence', 0.45),
                        "alpha_decay": message.get('alpha_decay', 0.996),
                        "alpha_frequency": message.get('alpha_frequency', 0.45),
                        "token_ban": message.get('token_ban', [])}
        msg, state, pos_state, neg_state = rnn_model.generate(Message.new(message),
                                                              infer_config,
                                                              state=state,
                                                              pos_state=pos_state,
                                                              neg_state=neg_state)
        result.append(msg.json())
    infer_state = copy.deepcopy(state)
    if pos_state is not None:
        pos_state = copy.deepcopy(pos_state)
    if neg_state is not None:
        neg_state = copy.deepcopy(neg_state)
    if debug:
        print("--after--->", state)
    return {'messages': result}

@route('/inference/flow-tx', method='POST')
def infer_by_tx_data():
    global rnn_model, debug, infer_state
    req = request.json
    if rnn_model is None:
        print(f"===load model===")
        infer_model_load()
    messages = req['messages']
    result = []
    state = None
    if infer_state is not None:
        state = copy.deepcopy(infer_state)
    debug = req.get('debug',debug)

    def generate_messages(state=state):
        global infer_state
        if debug:
            print("--before--->", state)
        for message in messages:
            infer_config = {"temperature": message.get("temperature", 0.2),
                            "top_p": message.get('top_p', 0.2),
                            "token_count": message.get('token_count', 256),
                            "token_stop": message.get('token_stop', []),
                            "alpha_presence": message.get('alpha_presence', 0.45),
                            "alpha_decay": message.get('alpha_decay', 0.996),
                            "alpha_frequency": message.get('alpha_frequency', 0.45),
                            "token_ban": message.get('token_ban', [])
                            }
            generator = rnn_model.flow_generate(tokenizer,
                                            Message.new(message),
                                            infer_config,
                                            state=state)
            try:
                while True:
                    out_str = next(generator)
                    yield out_str # 动态输出结果
            except StopIteration as error:
                state = error.value  # 获取并更新 state
            except Exception as error:
                print(f"在处理消息时generate_messages()发生错误: {error}")
                break
        infer_state = copy.deepcopy(state)
        if debug:
            print("--after--->", state)

    response.content_type = 'text/plain'  # 设置响应的Content-Type
    return generate_messages()  # 返回生成器函数


@route('/inference/by/messages', method='POST')
def infer_by_messages():
    global rnn_model, infer_state, debug, pos_state, neg_state
    req = request.json
    debug = req.get('debug',debug)
    if rnn_model is None:
        print(f"===load model===")
        infer_model_load()
    messages = req['messages']
    result = []
    state = None
    if infer_state is not None:
        state = copy.deepcopy(infer_state)
    if pos_state is not None:
        pos_state = copy.deepcopy(pos_state)
    if neg_state is not None:
        neg_state = copy.deepcopy(neg_state)
    if debug:
        print("--before--->", state)
    for message in messages:
        infer_config = {"temperature": message.get("temperature", 0.2),
                        "top_p": message.get('top_p', 0.2),
                        "token_count": message.get('token_count', 256),
                        "token_stop": message.get('token_stop', []),
                        "alpha_presence": message.get('alpha_presence', 0.45),
                        "alpha_decay": message.get('alpha_decay', 0.996),
                        "alpha_frequency": message.get('alpha_frequency', 0.45),
                        "token_ban": message.get('token_ban', [])}
        msg, state, pos_state, neg_state = rnn_model.generate(Message.new(message),
                                                              infer_config,
                                                              state=state,
                                                              pos_state=pos_state,
                                                              neg_state=neg_state)
        result.append(msg.json())
    infer_state = copy.deepcopy(state)
    if pos_state is not None:
        pos_state = copy.deepcopy(pos_state)
    if neg_state is not None:
        neg_state = copy.deepcopy(neg_state)
    if debug:
        print("--after--->", state)
    return {'messages': result}


@route('/inference/by/tokens', method='POST')
def infer_by_tokens():
    global rnn_model, infer_state, debug, pos_state, neg_state
    req = dict(request.json)
    if rnn_model is None:
        print(f"===load model===")
        infer_model_load()
    tokens = req['tokens']
    result = []
    state = None
    if infer_state is not None:
        state = copy.deepcopy(infer_state)

    infer_config = {"temperature": req.get("temperature", 0.1),
                    "top_p": req.get('top_p', 0.85),
                    "token_count": req.get('token_count', 256),
                    "token_stop": req.get('token_stop', []),
                    "alpha_presence": req.get('alpha_presence', 0.2),
                    "alpha_decay": req.get('alpha_decay', 0.996),
                    "alpha_frequency": req.get('alpha_frequency', 0.2),
                    "token_ban": req.get('token_ban', [])
                    }
    msg, state, pos_state, neg_state = rnn_model.generate(Message.new({}),
                                    infer_config,
                                    state=state)
    result.append(msg.json())
    infer_state = copy.deepcopy(state)
    return {'messages': result}


# ================
# utils
# ================

if True:
    run(host='0.0.0.0', port=config['port'])
