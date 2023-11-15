from utils.tokenizer import TRIE_TOKENIZER
from prompts.messages import Message
import types
import gc
import copy
import torch
from bottle import route, run, request
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
# ================
# load model
# ================

if args.rwkv_version == "v4":
    if args.infctx_on:
        from models.v4.infctx import RWKV
    else:
        from models.v4.origin import RWKV
    train_model = RWKV(args)
    from models.v4.runner import RWKV_RNN
elif args.rwkv_version == "v5":
    if args.infctx_on:
        from models.v5.infctx import RWKV
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
    global train_state,training_step
    req = dict(request.json)
    min_loss = req.get('max_loss', args.min_loss)
    max_loss = req.get('min_loss', args.max_loss)
    min_loss_fix = req.get('min_loss_fix', args.min_loss_fix)
    max_loss_fix = req.get('max_loss_fix', args.max_loss_fix)
    ctx_len = req.get('ctx_len', args.ctx_len)
    window = req.get('window', args.window)
    messages = req.get('messages', [])
    messages = [Message.new(x) for x in messages]
    messages = [x.tokens(for_infer=False) for x in messages]
    tokens = []
    masks = []
    for token, mask in messages:
        tokens += token
        masks += mask
    if len(tokens) == 0:
        return {"response": "no tokens"}
    losses = []
    total = 0
    mean_loss = 0
    i = 0
    n = 0
    if train_state is not None:
        states = copy.deepcopy(train_state)
    else:
        states = train_state
    if debug:
        print("-train-state-before>", states)

    while len(tokens) > 0:
        # training_step += 1
        i += 1
        output = tokens[:ctx_len]
        output_masks = masks[:ctx_len]
        tokens = tokens[ctx_len - window:]
        masks = masks[ctx_len - window:]
        batch = {'input_ids': output,
                 'masks': output_masks}
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
    global train_state
    req = dict(request.json)
    min_loss = req.get('max_loss', args.min_loss)
    max_loss = req.get('min_loss', args.max_loss)
    min_loss_fix = req.get('min_loss_fix', args.min_loss_fix)
    max_loss_fix = req.get('max_loss_fix', args.max_loss_fix)
    ctx_len = req.get('ctx_len', args.ctx_len)
    window = req.get('window', args.window)
    all_tokens = req.get('tokens', [])
    if len(all_tokens) == 0:
        return {"response": "no tokens"}
    losses = []
    total = 0
    mean_loss = 0
    i = 0
    n = 0
    if train_state is not None:
        states = copy.deepcopy(train_state)
    else:
        states = train_state
    if debug:
        print("-train-state-before>", states)
    for tokens in all_tokens:
        while len(tokens) > 0:
            i += 1
            output = tokens[:ctx_len]
            tokens = tokens[ctx_len - window:]
            # output_masks = [1 for x in output]
            batch = {'input_ids': output,
                     'masks': None}
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
            print(f"\nmean-loss->{mean_loss}")
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
    global infer_state
    req = dict(request.json)
    infer_state = None
    return {"message": "success"}


@route('/inference/state/save', method='POST')
def save_infer_state():
    global infer_state, infer_state_map
    req = dict(request.json)
    s_name = req.get('save_state', False)
    if s_name and infer_state is not None:
        infer_state_map[s_name] = copy.deepcopy(infer_state).cpu()
    return {"message": "success"}


@route('/inference/state/load', method='POST')
def load_infer_state():
    global infer_state, infer_state_map
    req = request.json
    req = dict(request.json)
    s_name = req.get('save_state', False)
    if s_name:
        infer_state = infer_state_map.get(s_name, None)
    else:
        return {"message": "fail"}
    if infer_state is not None:
        infer_state = infer_state.cuda()
    return {"message": "success"}


@route('/inference/state/save-to-disk', method='POST')
def infer_state_to_disk():
    global infer_state, infer_state_map
    req = request.json
    req = dict(request.json)
    s_name = req.get('save_name', "default-states")
    fpath = f"./resources/states_for_infer/{s_name}.pth"
    torch.save(copy.deepcopy(infer_state_map), fpath)
    return {"message": "save success"}


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
    global rnn_model, infer_state
    req = request.json
    if rnn_model is None:
        print(f"===load model===")
        infer_model_load()
    messages = req['messages']
    result = []
    state = None
    if infer_state is not None:
        state = copy.deepcopy(infer_state)

    if debug:
        print("--before--->", state)
    for message in messages:
        infer_config = {"temperature": message.get("temperature", 0.1),
                        "top_p": message.get('top_p', 0.85),
                        "token_count": message.get('token_count', 256),
                        "token_stop": message.get('token_stop', []),
                        "alpha_presence": message.get('alpha_presence', 0.2),
                        "alpha_decay": message.get('alpha_decay', 0.996),
                        "alpha_frequency": message.get('alpha_frequency', 0.2),
                        "token_ban": message.get('token_ban', [])
                        }
        msg, state = rnn_model.generate(tokenizer,
                                        Message.new(message),
                                        infer_config,
                                        state=state)
        result.append(msg.json())
    infer_state = copy.deepcopy(state)
    if debug:
        print("--after--->", state)
    return {'messages': result}


@route('/inference/by/messages', method='POST')
def infer_by_messages():
    global rnn_model, infer_state
    req = dict(request.json)
    if rnn_model is None:
        print("===load model===")
        infer_model_load()
    messages = req['messages']
    result = []
    state = None
    if infer_state is not None:
        state = copy.deepcopy(infer_state)
    for message in messages:
        infer_config = {"temperature": message.get("temperature", 0.1),
                        "top_p": message.get('top_p', 0.85),
                        "token_count": message.get('token_count', 256),
                        "token_stop": message.get('token_stop', []),
                        "alpha_presence": message.get('alpha_presence', 0.2),
                        "alpha_decay": message.get('alpha_decay', 0.996),
                        "alpha_frequency": message.get('alpha_frequency', 0.2),
                        "token_ban": message.get('token_ban', [])
                        }
        msg, state = rnn_model.generate(tokenizer,
                                        Message.new(message),
                                        infer_config,
                                        state=state)
        result.append(msg.json())
    infer_state = copy.deepcopy(state)
    return {'messages': result}


@route('/inference/by/tokens', method='POST')
def infer_by_tokens():
    global rnn_model, infer_state
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
    msg, state = rnn_model.generate(tokenizer,
                                    Message.new({}),
                                    infer_config,
                                    state=state)
    result.append(msg.json())
    infer_state = copy.deepcopy(state)
    return {'messages': result}


# ================
# utils
# ================

if False:
    data = tokenizer.encode(' My name is gpt. and My favorite color is red')
    batch = {"input_ids": data}
    loss = model_engine.training_step(batch)
    print(f"-loss--->{loss}")
    model_engine.backward(loss)
    model_engine.step()

    data = tokenizer.encode(' My name is gpt. and My favorite color is red')
    batch = {"input_ids": data}
    loss = model_engine.training_step(batch)
    print(f"-loss--->{loss}")
    model_engine.backward(loss)
    model_engine.step()

    data = tokenizer.encode(' My name is gpt. and My favorite color is red')
    batch = {"input_ids": data}
    loss = model_engine.training_step(batch)
    print(f"-loss--->{loss}")
    model_engine.backward(loss)
    model_engine.step()

    rnn_model = RWKV_RNN(model_engine.module.state_dict(), dtype='bf16')
    rnn_model.generate(tokenizer,
                       Message.new(
                           {"role": "text", "text": "\nUser:what's your name?\n\nAssistant:"}),
                       {"temperature": 1.0,
                        "top_p": 0.85,
                        "token_count": 256,
                        "token_stop": [0],
                        "alpha_presence": 0.2,
                        "alpha_decay": 0.996,
                        "alpha_frequency": 0.2,
                        "token_ban": []
                        },
                       None)

if True:
    run(host='0.0.0.0', port=3000)
