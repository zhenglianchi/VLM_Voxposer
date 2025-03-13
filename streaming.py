from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-df55df287b2c420285feb77137467576",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

completion = client.chat.completions.create(
    model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "给我几个两个数相加的代码例子,只给我一段python代码,不要其他任何输出与提示内容,不要使用函数，代码之间不要有空行"}
    ],
    stream=True
)

full_content = ""
print("流式输出内容为：")

skip_first_flag = False
preview_content = None
last_content = None

for chunk in completion:
    if chunk.choices:
        content = chunk.choices[0].delta.content
        full_content += content
        #print(repr(content))  # 输出当前流式数据
        
        if "\n" in content and skip_first_flag:  # 若当前内容包含换行符，说明已接收到完整代码，可以执行代码
            code_lines = full_content.split('\n')
            preview_content = code_lines[-2]
            #print(code_lines)
            print(f"执行代码：{preview_content}")
            exec(preview_content)
        
        skip_first_flag = True  # 跳过第一行
        # 每次接收到新的内容后，按行分割并逐行执行
        '''code_lines = content.split('\n')
        for line in code_lines:
            if line.strip():  # 忽略空行
                try:
                    exec(line)  # 执行每一行代码
                except Exception as e:
                    print(f"执行代码时发生错误: {e}")'''
        

last_content = full_content.split('\n')[-1]
print(f"执行代码：{last_content}")
exec(last_content)


# 最终输出完整内容
print(f"完整内容为：\n{full_content}")
