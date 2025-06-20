from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
from openai import OpenAI
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import dashscope
from http import HTTPStatus
from koala.utils.time_limit import RateLimiter
from retrying import retry

def call_chatbot(llm_model, api_key, base_url, huggingface=False):
    '''


    Args:
        llm_model: llm could be the local dir (vllm) or just model name (call api and using huggingface)
        huggingface: hugging face loading mode

    Returns: chatbot (instance)
    note: chatbot.chat(messages=[{}, ], model=llm_model)

    '''
    if not huggingface:  # call by api
        if llm_model.lower() in {"gpt-4o-mini", "gpt-4o"}:  # series models of gpt
            chatbot = ChatBot(api_key=api_key,
                              base_url=base_url,
                              model=llm_model)

        elif "/" in llm_model.lower():  # local model using vllm, it is the local dir with "/"
            chatbot = ChatBot(api_key="EMPTY", base_url="http://localhost:8000/v1",
                              model=llm_model)

        elif any([i in llm_model.lower() for i in ["qwen"]]):  # series models of gpt qwen, like qwen-turbo-2024-11-01
            chatbot = ChatBot(api_key=api_key,
                              base_url=base_url,  # "https://dashscope.aliyuncs.com/compatible-mode/v1"
                              model=llm_model)
        elif "farui" in llm_model:
            chatbot = DashScope(api_key=api_key,
                                base_url=base_url,
                                model=llm_model)  # "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif "deepseek" in llm_model.lower():
            chatbot = ChatBot(api_key=api_key,
                              base_url=base_url,  # "https://api.deepseek.com"
                              model=llm_model)
        else:
            raise NotImplemented

    if huggingface:  # using huggingface to call local model
        assert "/" not in llm_model
        if llm_model.lower().startswith("qwen"):  # like Qwen2.5-7B-Instruct
            chatbot = Qwen(model_name=llm_model)
        elif llm_model.lower().startswith("meta-llama"):  # Meta-Llama-3.1-8B-Instruct
            chatbot = LLAMA(model_name=llm_model)
        else:
            raise NotImplemented
    return chatbot


class DashScope:
    def __init__(self, api_key: str, base_url: str, model: str):
        dashscope.api_key = api_key
        dashscope.base_url = base_url
        self.model = model

    def chat(self, messages):
        messages = [{'role': 'system',
                     'content': 'You are a helpful assistant.'}] + messages
        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
        )
        if response.status_code == HTTPStatus.OK:
            return response["output"]["choices"][0]["message"]["content"]
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            return None


class ChatBot:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=api_key,
            base_url=base_url  # "https://api.chatanywhere.tech/v1"
            # base_url="https://api.chatanywhere.com.cn/v1",
        )
        self.model = model

    def chat(self, messages, return_token_num=False, **kwargs):  #
        try:
            completion = self.client.chat.completions.create(
                model=self.model,  # "gpt-4o-mini",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=messages,
                **kwargs
            )  # {'role': 'system/user', 'content': 'You are a helpful assistant.'}
            # print("Q:", messages[0]["content"])
            completion = json.loads(completion.model_dump_json())
            response = completion['choices'][0]['message']['content'].strip()

            if return_token_num:
                token_num = {"prompt_tokens": completion['usage']["prompt_tokens"],
                             "total_tokens": completion['usage']["total_tokens"]}
                return response, token_num
            else:
                return response

        except Exception as e:
            print(f"[Call LLM Error]: {e}")
            if return_token_num:
                return None, None
            else:
                return None


class Qwen():
    def __init__(self, model_name):
        # model_name = f"/root/pilab_jiang/hf-model/{model_name}/"
        # model_name = ""
        # model_name = "/root/pilab_jiang/hf-model/DeepSeek-R1-Distill-Qwen-14B/"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print(self.model.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def chat(self, messages):
        # prompt = "Give me a short introduction to large language model."
        # messages = [
        #     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        #     {"role": "user", "content": prompt}
        # ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
        return response


class LLAMA():
    def __init__(self, model_name):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(self.pipe.device)

    def chat(self, messages):
        # messages = [
        #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        #     {"role": "user", "content": message},
        # ]
        outputs = self.pipe(
            messages,
            max_new_tokens=256,
        )
        response = outputs[0]["generated_text"][-1]["content"]

        return response

class ChatClient:
    def __init__(self, chatbot_name, api_key, base_url, print_resp, rate_limit_count=1155, rate_limit_period=60):
        self.limiter = RateLimiter(rate_limit_count, rate_limit_period)
        self.chatbot = call_chatbot(chatbot_name, base_url=base_url, api_key=api_key)
        self.print_resp = print_resp

    def chat(self, prompt):
        if self.limiter is not None:
            self.limiter.acquire()
        messages = [{"role": "user", "content": prompt}]
        resp = self.chatbot.chat(messages)
        return resp

    @retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=5000)
    def get_valid_structured_output(self, prompt, parse_func=None):
        """尝试生成并解析判决原因字典，最多重试5次"""
        resp = self.chat(prompt)
        if self.print_resp:
            print("RESP:", resp)
        if not parse_func:
            return resp
        parsed_result = parse_func(resp)
        if parsed_result is None:
            raise ValueError("[Parsed error!]")
        return parsed_result