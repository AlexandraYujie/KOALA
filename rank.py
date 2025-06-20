from koala.llm_rank import rank
from koala.config.config import RankConfig

config = RankConfig()
config.configure(
    language="cn",  # set "cn" or "en"
    style_name="强制性",  # enter the style name
    style_definition="语气越强硬、措施或要求越详细具体、涉及到硬性的数字标准，则文本的“强制性”越强。",  # enter the definition
    data_num=500,  # limit data_num per domain, set to None when no limit
    trg_domains={"eco", "gov"},
    train_ratio=0.8,
    api_key="sk-",  # enter your key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # enter the base url
    llm_model="qwen-plus",  # the name of ranking LLM
    max_workers=8,
    print_resp=False  # set True, each resp of LLM will print out
)

rank(config)
