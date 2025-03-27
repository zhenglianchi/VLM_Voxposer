from time import sleep
from openai import RateLimitError, APIConnectionError,OpenAI
from utils import load_prompt
import time
from VLM_demo import encode_image
from VLM_demo import read_state
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
        self.gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        self.lvars = self.gvars
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
    
    def _api_call(self, **kwargs):
        # check whether completion endpoint or chat endpoint is used
        if kwargs['model'] != 'gpt-3.5-turbo-instruct' and \
            any([chat_model in kwargs['model'] for chat_model in ['gpt-3.5', 'gpt-4', 'yi-large','glm-4-flash',"qwen2.5-72b-instruct"]]):
            # add special prompt for chat endpoint
            user1 = kwargs.pop('prompt')
            user_query = kwargs.pop('user_query')
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
            
            generate_code = response.choices[0].message.content.replace("```","").replace("python","")
            print(generate_code)

            return generate_code
            
            
        else:
            print("请更换您的模型为['gpt-3.5', 'gpt-4', 'yi-large','glm-4-flash','qwen2.5-72b-instruct']之一")


    def _vlmapi_call(self, action,planning):
        client = OpenAI(
            api_key="sk-df55df287b2c420285feb77137467576",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        base64_image = encode_image("./tmp/state_front.jpeg")

        completion = client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct",  
            messages=[{"role": "user","content": [
                    {"type": "text","text": f"This is a robotic arm operation scene, {planning} is a sequence plan, please tell me if this picture completes the {action} in the {planning}. Only output yes or no, and do not have any other information or hints"},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                    }
                    ]}]
            )

        #print(completion.choices[0].message.content[7:-3])
        flag = completion.choices[0].message.content

        return flag

    def __call__(self, query, lock):
        prompt, user_query, planning = self.build_prompt(query,self._cfg['planner_model'])
    
        action = planning.pop(0)
        planning_completed = False
        while not planning_completed:
            print(f"Action: {action}")
            action_completed = False
            try:
                generate_code = self._api_call(
                    prompt=prompt,
                    user_query = user_query,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=self._cfg['max_tokens'],
                    action = action,
                )
                while True:
                    state = self.get_state(self.state_json_path,lock)
                    new_global_vars = {
                        'state': state,
                    }
                    self.gvars.update(new_global_vars)
                    exec(generate_code, self.gvars, self.lvars)
                    stop = self.lvars["stop"]
                    # 到达目标位置
                    if stop:
                        break

                while not action_completed:
                    flag = self._vlmapi_call(action,planning)
                    print("MVLMs give a final answer:",flag)
                    flag = "yes"
                    if flag == "yes":
                        if len(planning) == 0:
                            print("The plan has been completed.")
                            planning_completed = True
                            action_completed = True
                        else:
                            action = planning.pop(0)
                            action_completed = True
                    elif flag == "no":
                        break
                    else:
                        print("MVLMs are not give a final answer.")
                        pass


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
    