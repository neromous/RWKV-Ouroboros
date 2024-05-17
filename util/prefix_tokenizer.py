########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)

    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)

    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]

        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret


sp_map = {
    "<|im-s|>": [65534],
    "<|im-e|>": [65535],
    "<|sys-s|>": [65530],
    "<|sys-e|>": [65531],
    "<|req-s|>": [65532],
    "<|req-e|>": [65533],
    "<|resp-s|>": [65529, 65534],
    "<|resp-e|>": [65529, 65535], 
    "<|page-s|>": [65514],
    "<|page-e|>": [65515],
    "<|me|>":[65529]
    }

class prefix_tokenizer():
    def __init__(self, 
                file_name="/home/neromous/Documents/RWKV-Ourboros/resources/vocab/rwkv_vocab_v20230424.txt",
                sp_map=sp_map):
        self.sp_map = sp_map
        self.idx2token = {}
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode_raw(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        try:
            return self.decodeBytes(tokens).decode('utf-8')
        except:
            return '\ufffd' # bad utf-8

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()

    def encode(self, text):
        res = []
        cache = ""
        for x in text:
            cache += x
            if cache.endswith(">"):
                for mk in self.sp_map.keys():
                    if cache.endswith(mk) and len(cache) != 0:
                        res += self.encode_raw(cache[:-len(mk)])
                        res += self.sp_map[mk]
                        cache = ""

                        
        res += self.encode_raw(cache)
        return res

    def clean_sp(self, text):
        for k in self.sp_map.keys():
            text = text.replace(k, "\n")
        return text


if __name__ == "__main__":
    tokenizer = prefix_tokenizer()
    with open("/home/neromous/Documents/cache/train-data/sft/sft-v3.org", 'r',encoding="utf-8") as f:
        m = f.read()
    text = m[:4000]
    print(text)
    print(tokenizer.encode(text))
    # text = "<|im-s|>你好啊但哎但<|im-e|>"


    # assert text == tokenizer.decode(tokenizer.encode(text)) \
    #                         .replace("\x16","<|im-s|>") \
    #                         .replace("\x17","<|im-e|>")
    # text = "dadsfa<|resp-s|>你好啊但哎但asdfdsf<|resp-e|>dfas<df|>"
    # print(tokenizer.encode(text))
    # print(text)
    # res = tokenizer.decode(tokenizer.encode(text)) \
    #                         .replace("\x16","<|resp-s|>") \
    #                         .replace("\x17","<|resp-e|>")
    # print(res)
    # assert text == res
    # text = "df><ddaasdf<|req-e|>你好啊但哎但<|req-s|>>"
    # assert text == tokenizer.decode(tokenizer.encode(text)) \
    #                         .replace("\x16","<|req-s|>") \
    #                         .replace("\x17","<|req-e|>")
    # text = "dfads<|im-e|>>你好啊但哎但<|im-s|>-s|><|>fadsfdas"
    # assert text == tokenizer.decode(tokenizer.encode(text)) \
    #                         .replace("\x16","<|im-s|>") \
    #                         .replace("\x17","<|im-e|>")
