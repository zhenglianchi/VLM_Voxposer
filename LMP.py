from time import sleep
from openai import RateLimitError, APIConnectionError,OpenAI
from utils import load_prompt
import time
from VLM_demo import encode_image
from world_state import read_state
import json


class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._planner_prompt = load_prompt(f"{env}/{self._cfg['planner_prompt_fname']}.txt")
        self._stop_tokens = [self._cfg['stop']]
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self._context = None
        #set your api_key Qwen
        self.api_key= "sk-df55df287b2c420285feb77137467576"
        self.base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.state_json_path = "./tmp/state_front.json"
        #set your api_key lingyi
        #self.api_key= "f972734155394670bf3d86d36884b7ed"
        #self.base_url="https://api.lingyiwanwu.com/v1"
        #zhipu
        #self.api_key="875ccefccb36535614c10fa4d0a62f97.rUp0uGBiIQIy4vD5"
        #self.base_url="https://open.bigmodel.cn/api/paas/v4/"

    def build_prompt(self, query, model):
        prompt = self._base_prompt

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'

        planner_prompt = self._planner_prompt

        if self._context :
            user_query = f"# Objects : {self._context}\n" + user_query

        client = OpenAI(api_key=self.api_key,base_url=self.base_url)
        
        #这里加图像的prompt
        image_path = "./tmp/state_front.jpeg"
        base64_image = encode_image(image_path)

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user","content": [
                #{"type": "text","text": "Detect all objects in the image and return their locations in the form of coordinates. The format of output should be like {“bbox”: [x1, y1, x2, y2], “label”: the name of this object in Chinese}"},
                {"type": "text","text": f"This is a robotic arm operation scene image.\n{planner_prompt}\nThe above are some examples of planning, please give the corresponding planning according to the image I gave you next:\n{user_query}. The output format likely is\n" + "planner : ['', '', '', '']\nOther than that, don't give me any superfluous information and hints"},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}],
        )

        planner = completion.choices[0].message.content

        planning = json.loads(planner.split(":")[-1].strip())

        user_query = user_query + f"\n# {planner}"

        prompt += f'\n{user_query}'

        return prompt, user_query, planning
    
    def get_state(self, state_json_path,lock):
      state = read_state(state_json_path,lock)
      return state
    
    def _cached_api_call(self, **kwargs):
        # check whether completion endpoint or chat endpoint is used
        if kwargs['model'] != 'gpt-3.5-turbo-instruct' and \
            any([chat_model in kwargs['model'] for chat_model in ['gpt-3.5', 'gpt-4', 'yi-large','glm-4-flash',"qwen2.5-72b-instruct"]]):
            # add special prompt for chat endpoint
            user1 = kwargs.pop('prompt')
            user_query = kwargs.pop('user_query')
            lock = kwargs.pop('lock')
            action = kwargs.pop('action')
            user_query = user_query + f"\n# action : {action}"

            user1 = ''.join(user1.split('# Query : ')[:-1]).strip()
            user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
            assistant1 = f'Got it. I will complete what you give me next.'
            user2 = "Please do not include empty lines in the generated code.There needs to be a line break to split each line.\n" + user_query

            messages=[
                {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": user2},
            ]
            #print(user2)

            kwargs['messages'] = messages
            client = OpenAI(api_key=self.api_key,base_url=self.base_url)

            start_time = time.time()

            response = client.chat.completions.create(**kwargs)

            print(f'*** OpenAI API call took {time.time() - start_time:.2f}s ***')

            full_content = ""
            print("流式输出内容为：")

            skip_first_flag = False
            preview_content = None
            last_content = None
            gvars = merge_dicts([self._fixed_vars, self._variable_vars])
            lvars = kwargs

            #print(gvars,lvars)

            for chunk in response:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    #print(repr(content))  # 输出当前流式数据
                    
                    if "\n" in content and skip_first_flag:  # 若当前内容包含换行符，说明已接收到完整代码，可以执行代码
                        code_lines = full_content.split('\n')
                        preview_content = code_lines[-2]
                        if "```" in preview_content:
                            continue
                        #print(code_lines)
                        print(f"执行代码：{preview_content}")
                        try:
                            state = self.get_state(self.state_json_path,lock)
                            new_global_vars = {
                                'state': state,
                            }
                            gvars.update(new_global_vars)

                            exec(preview_content, gvars, lvars)  # 执行每一行代码
                        except Exception as e:
                            print(f"执行代码时发生错误: {e}")
                    
                    skip_first_flag = True  # 跳过第一行的'''python

            last_content = full_content.split('\n')[-1].strip()
            if "```" not in last_content and last_content:
                print(f"执行代码：{last_content}")
                try:
                    state = self.get_state(self.state_json_path,lock)
                    new_global_vars = {'state': state,}
                    gvars.update(new_global_vars)
                    exec(last_content, gvars, lvars)  # 执行每一行代码
                except Exception as e:
                    print(f"执行代码时发生错误: {e}")

            print(full_content)
            


        else:
            print("请更换您的模型为['gpt-3.5', 'gpt-4', 'yi-large','glm-4-flash','qwen2.5-72b-instruct']之一")



    def __call__(self, query, lock):
        prompt, user_query, planning = self.build_prompt(query,self._cfg['planner_model'])
 
        for action in planning:
            try:
                self._cached_api_call(
                    prompt=prompt,
                    user_query = user_query,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=self._cfg['max_tokens'],
                    stream = True,
                    lock = lock,
                    action = action,
                )

            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 3s.')
                sleep(3)


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e
    