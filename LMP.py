from time import sleep
from openai import RateLimitError, APIConnectionError,OpenAI
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from utils import load_prompt
import time
from LLM_cache import DiskCache
import traceback

class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = [self._cfg['stop']]
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self._context = None
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])
        #set your api_key Qwen
        self.api_key= "sk-df55df287b2c420285feb77137467576"
        self.base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        #set your api_key lingyi
        #self.api_key= "f972734155394670bf3d86d36884b7ed"
        #self.base_url="https://api.lingyiwanwu.com/v1"
        #zhipu
        #self.api_key="875ccefccb36535614c10fa4d0a62f97.rUp0uGBiIQIy4vD5"
        #self.base_url="https://open.bigmodel.cn/api/paas/v4/"

    def build_prompt(self, query):
        prompt = self._base_prompt

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'

        if self._context :
            user_query = f"# Objects : {self._context}\n" + user_query

        prompt += f'\n{user_query}'

        return prompt, user_query
    
    def _cached_api_call(self, **kwargs):
        # check whether completion endpoint or chat endpoint is used
        if kwargs['model'] != 'gpt-3.5-turbo-instruct' and \
            any([chat_model in kwargs['model'] for chat_model in ['gpt-3.5', 'gpt-4', 'yi-large','glm-4-flash',"qwen2.5-72b-instruct"]]):
            # add special prompt for chat endpoint
            user1 = kwargs.pop('prompt')
            new_query = '# Query : ' + user1.split('# Query : ')[-1]
            user1 = ''.join(user1.split('# Query : ')[:-1]).strip()
            user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
            assistant1 = f'Got it. I will complete what you give me next.'
            user2 = new_query

            messages=[
                {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": user2},
            ]
            
            kwargs['messages'] = messages
            if kwargs in self._cache:
                print('(usingrandom_colostater cache)', end=' ')
                return self._cache[kwargs]
            else:
                client = OpenAI(api_key=self.api_key,base_url=self.base_url)
                response = client.chat.completions.create(**kwargs)
                ret = response.choices[0].message.content
                # post processing
                ret = ret.replace('```', '').replace('python', '').strip()
                self._cache[kwargs] = ret
                return ret
        else:
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs]
            else:
                client = OpenAI(api_key=self.api_key,base_url=self.base_url)
                ret = client.chat.completions.create(**kwargs).choices[0].text.strip()
                self._cache[kwargs] = ret
                return ret

    def __call__(self, query, **kwargs):
        prompt, user_query = self.build_prompt(query)
 
        start_time = time.time()
        while True:
            try:
                code_str = self._cached_api_call(
                    prompt=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=self._cfg['max_tokens']
                )
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 3s.')
                sleep(3)
        print(f'*** OpenAI API call took {time.time() - start_time:.2f}s ***')

        to_exec = code_str
        to_log = f'{user_query}\n{to_exec}\n{self._stop_tokens[0]}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        print('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}\n')

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)
            except Exception as e:
                print(f'Error: {e}')
                import pdb ; pdb.set_trace()
        else:
            exec_with_line_print(to_exec, gvars, lvars)
            #exec_safe(to_exec, gvars, lvars)



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
    

def exec_with_line_print(code_str, gvars=None, lvars=None):
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
    # 将代码按行分割
    lines = code_str.splitlines()
    for line_num, line in enumerate(lines, start=1):
        try:
            # 打印当前行号和代码内容
            print(f"Executing line {line_num}: {line}")
            # 执行当前行
            exec(line, custom_gvars, lvars)
        except Exception as e:
            # 打印错误信息
            print(f"Error at line {line_num}: {line}")
            traceback.print_exc()
            raise e