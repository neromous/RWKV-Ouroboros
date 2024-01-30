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

class TRIE_TOKENIZER():
    def __init__(self, file_name, sp_map=False):
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
        if self.sp_map:
            res = []
            cache = ""
            for x in text:
                cache += x
                if cache.endswith(">"):
                    for mk in self.sp_map.keys():
                        if cache.endswith(mk):
                            res += self.encode_raw(cache[:-len(mk)])
                            res += self.sp_map[mk]
                            cache = ""
                            break
                else:
                    pass
            res += self.encode_raw(cache)
            return res
        else:
            return self.encode_raw(text)


if __name__ == "__main__":
    text = """
<|page-s|>
<|sys-s|>You are a science fiction video creator, and you are currently running a science fiction Youtube channel to share science fiction novels.<|sys-e|>

<|req-s|>Now you need to design the following story into a script with five scenes. The script includes serial number, subtitle, plot summary, composition, narration. Present it in table form. Remember not to reverse the rows and columns. I need a English version.
The story:
<|in-s|>
第一章 游动的暗礁
1866年，发生了一起非同寻常的事件，那真是一种神秘而又无法解释的现象。无疑，人们对当时的情景至今仍不能忘怀。且不说当时那些沿海地区的居民对此感到兴奋异常，到处散布各种传闻，即使是内陆地区的居民也被各种消息搞得心神不宁，情绪极为激动。尤其是那些从事航海工作的海员，他们对这件事尤其充满了兴趣。欧洲、美洲两个大陆的商人、普通海员、船主、船长、各国的海军军官以及两大洲的各国政府等等，大家都密切地关注着事态的进一步发展。
事情的原委是这样的：不久前，有几艘轮船在海上遇到一头“庞然大物”，那是一个长长的东西，呈纺锤形，有时全身散发着磷光，而且它的体积超过了一头鲸鱼，动作也比鲸鱼迅速得多。
有关这个离奇怪物的出现，各种航海日志都留下相关记载。这些日志大都记载了这个物体或者说可疑生物的形状、它在运动时一直保持的高速，以及它令人惊异的运动能量。它那奇特的生命力似乎是与生俱来的本能。如果它是一种鲸类动物，可是它的身体尺寸却超过了迄今为止生物学界研究过的各类鲸鱼。居维叶〔1〕、拉塞佩德〔2〕、迪梅里、德·卡特法日这些博物学家是不会承认这种怪物的存在的，除非他们看到过它，也就是说除非这些科学家亲眼目睹了这头怪物的存在。
综合考虑人们的多次观察结果——我们排除了那些最保守的估计，他们认为这头怪物只有200英尺〔3〕长，同时我们也不能接受那些过于夸张的观点，认为这个怪物足有1英里宽、3英里〔4〕长——最后，我们可以比较公正地得出结论说，如果这个神秘的物体果真存在，那么这个存在物的体积，大大地超过了当前所有鱼类学家所认可的尺寸。
既然这头怪物的存在已经成为一件不容否认的事实，那么我们就不难理解，由于人类的好奇心，这头神奇怪物所引发的兴奋很快便波及到了整个世界。至于说上述所谈到的一切，如果还有人认为那都不过是荒诞不经的传言，那么他的结论是没有人能够接受的。
1866年7月20日，加尔各答—布纳希汽船航运公司的“希金森总督”号汽船，在澳大利亚东海岸5英里处曾经遇到这头移动的庞然大物。最初，贝克船长以为在自己面前出现的是一座至今还没有被人们发现的暗礁，他甚至准备测定一下它的准确位置，可是就在这个时候，那个不名物体忽然喷出两股水柱。随着咝咝的声音，那两股水柱一直射向半空，竟然高达150多英尺。如果不是这座暗礁上有一个间歇泉〔5〕，那么“希金森总督”号所遇到的正是一种至今尚不为人所知的水栖哺乳动物，因此在它的喷水孔中才可以喷出含有空气和水汽的水柱。
类似事件又出现在同年的7月23日，当时西印度—太平洋汽船航运公司的“哥伦布”号正航行在太平洋上。可是，这头特别的鲸类物体竟然能够以惊人的速度从一个地方转移到另一个地方，因为仅仅相隔三天的时间，它居然会出现在两个相隔遥远的海域，而从“希金森总督”号到“哥伦布”号当时发现它的海域，这两个不同的地点相距2100多英里
<|in-e|>
The table form must follow the format like this.
```
| number | subtitle | plot_summary | composition | narration |
```<|req-e|>

<|resp-s|>| number | subtitle | plot_summary | composition | narration |
|--------|----------|--------------|-------------|----------|
| 1 | The Moving Reef | In 1866, a mysterious phenomenon rocks the maritime world. Several ships encounter an enormous, unidentified creature in the ocean, sparking excitement and theories among sailors, scientists, and governments worldwide. | The script opens with a global perspective, setting the stage for the mystery. | In the year 1866, an extraordinary event unfolded that would leave an indelible mark on the world's collective memory. |
| 2 | The Phosphorescent Giant | The logs of various ships describe a colossal, phosphorescent creature with unparalleled speed and agility, defying the known limits of marine biology. | Close-up on the mystery, introducing the physical attributes of the creature. | The creature was vast, with a form like none other, shimmering with a bioluminescence that stunned those who witnessed it. |
| 3 |科学界的 skepticism | The scientific community, including renowned naturalists like Cuvier and Linné, doubts the existence of such a creature without concrete evidence. | Introducing the skepticism of the scientific community and setting the stage for the evidence to come. | The learned minds of the era, firm in their knowledge, refused to believe in the monster's existence without firsthand proof. |
| 4 | The Sighting of the "Governor Hitchinson" | On July 20th, the "Governor Hitchinson" of the Calcutta and Bombay Steam Navigation Company encounters the creature, mistaking it for an undiscovered reef, only to witness it spray water columns into the air. | A detailed account of the first encounter, building the tension and mystery. | Captain Beck, aboard the "Governor Hitchinson," was fooled by what appeared to be an immovable reef, until the creature erupted with a force that could not be mistaken. |
| 5 | The Elusive Beast | Just three days later, the "Columbus" of the West India and Pacific Steam Navigation Company sights the creature in the Pacific, thousands of miles away from the previous sighting, raising questions about its incredible speed and mobility. | The script concludes with a mystery deepening, leaving viewers in suspense. | The world was left to grapple with the impossible: a creature capable of crossing vast distances in mere days, defying the known laws of nature. |<|resp-e|>
<|page-e|>


<|page-s|>
<|req-s|>Extract the script data in the table in the text below and organize it into json format.
```
| number | subtitle | plot_summary | composition | narration |
|--------|----------|--------------|-------------|----------|
| 1 | The Moving Reef | In 1866, a mysterious phenomenon rocks the maritime world. Several ships encounter an enormous, unidentified creature in the ocean, sparking excitement and theories among sailors, scientists, and governments worldwide. | The script opens with a global perspective, setting the stage for the mystery. | In the year 1866, an extraordinary event unfolded that would leave an indelible mark on the world's collective memory. |
| 2 | The Phosphorescent Giant | The logs of various ships describe a colossal, phosphorescent creature with unparalleled speed and agility, defying the known limits of marine biology. | Close-up on the mystery, introducing the physical attributes of the creature. | The creature was vast, with a form like none other, shimmering with a bioluminescence that stunned those who witnessed it. |
| 3 |科学界的 skepticism | The scientific community, including renowned naturalists like Cuvier and Linné, doubts the existence of such a creature without concrete evidence. | Introducing the skepticism of the scientific community and setting the stage for the evidence to come. | The learned minds of the era, firm in their knowledge, refused to believe in the monster's existence without firsthand proof. |
| 4 | The Sighting of the "Governor Hitchinson" | On July 20th, the "Governor Hitchinson" of the Calcutta and Bombay Steam Navigation Company encounters the creature, mistaking it for an undiscovered reef, only to witness it spray water columns into the air. | A detailed account of the first encounter, building the tension and mystery. | Captain Beck, aboard the "Governor Hitchinson," was fooled by what appeared to be an immovable reef, until the creature erupted with a force that could not be mistaken. |
| 5 | The Elusive Beast | Just three days later, the "Columbus" of the West India and Pacific Steam Navigation Company sights the creature in the Pacific, thousands of miles away from the previous sighting, raising questions about its incredible speed and mobility. | The script concludes with a mystery deepening, leaving viewers in suspense. | The world was left to grapple with the impossible: a creature capable of crossing vast distances in mere days, defying the known laws of nature. |
```
In addition
1. If Chinese words appears in the text, you must replace it with corresponding English words.
2. The data must have the following key: number, subtitle, plot_summary, composition, narration.<|req-e|>

<|resp-s|>Here is the data from the table organized into JSON format. The Chinese words have been translated into English:
```json
[
  {
    "number": 1,
    "subtitle": "The Moving Reef",
    "plot_summary": "In 1866, a mysterious phenomenon rocks the maritime world. Several ships encounter an enormous, unidentified creature in the ocean, sparking excitement and theories among sailors, scientists, and governments worldwide.",
    "composition": "The script opens with a global perspective, setting the stage for the mystery.",
    "narration": "In the year 1866, an extraordinary event unfolded that would leave an indelible mark on the world's collective memory."
  },
  {
    "number": 2,
    "subtitle": "The Phosphorescent Giant",
    "plot_summary": "The logs of various ships describe a colossal, phosphorescent creature with unparalleled speed and agility, defying the known limits of marine biology.",
    "composition": "Close-up on the mystery, introducing the physical attributes of the creature.",
    "narration": "The creature was vast, with a form like none other, shimmering with a bioluminescence that stunned those who witnessed it."
  },
  {
    "number": 3,
    "subtitle": "Scientific Skepticism",
    "plot_summary": "The scientific community, including renowned naturalists like Cuvier and Linnaeus, doubts the existence of such a creature without concrete evidence.",
    "composition": "Introducing the skepticism of the scientific community and setting the stage for the evidence to come.",
    "narration": "The learned minds of the era, firm in their knowledge, refused to believe in the monster's existence without firsthand proof."
  },
  {
    "number": 4,
    "subtitle": "The Sighting of the 'Governor Hitchinson'",
    "plot_summary": "On July 20th, the 'Governor Hitchinson' of the Calcutta and Bombay Steam Navigation Company encounters the creature, mistaking it for an undiscovered reef, only to witness it spray water columns into the air.",
    "composition": "A detailed account of the first encounter, building the tension and mystery.",
    "narration": "Captain Beck, aboard the 'Governor Hitchinson,' was fooled by what appeared to be an immovable reef, until the creature erupted with a force that could not be mistaken."
  },
  {
    "number": 5,
    "subtitle": "The Elusive Beast",
    "plot_summary": "Just three days later, the 'Columbus' of the West India and Pacific Steam Navigation Company sights the creature in the Pacific, thousands of miles away from the previous sighting, raising questions about its incredible speed and mobility.",
    "composition": "The script concludes with a mystery deepening, leaving viewers in suspense.",
    "narration": "The world was left to grapple with the impossible: a creature capable of crossing vast distances in mere days, defying the known laws of nature."
  }
]
```
Note: I replaced the Chinese characters with the English translation of the subtitle "Scientific Skepticism" and corrected the spelling of Linnaeus, who was referred to as Linné in the original text.<|resp-e|>
<|page-e|>
"""

    print("======text=======")
    print(text)
    print("======tokens=======")
    prompts = text
    tokenizer = TRIE_TOKENIZER("../resources/vocab_file/rwkv_vocab_v20230424.txt")
    text = "".join(prompts)
    tokens = tokenizer.encode(text)
    print(tokens)
    print("======after=======")
    after = tokenizer.decode(tokens)
    print(after)
