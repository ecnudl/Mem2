# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import aiohttp  # 导入 aiohttp 以便发起异步 HTTP 请求
import os  # 导入 os 读取环境变量
import asyncio  # 导入 asyncio 管理事件循环
from transformers import AutoTokenizer  # 导入分词器加载工具

URL = os.getenv("URL", "http://localhost:8000/v1")  # 获取服务地址，默认指向本地部署
API_KEY = os.getenv("API_KEY", "sk-HG6X66wxKNXtnDPjYhOYZHw7BOdHgqA5sM43DEMQPPztG332")  # 读取 API 密钥，默认使用示例值

RECURRENT_MAX_CONTEXT_LEN = 120000  # 限制总上下文的最大 token 数
RECURRENT_CHUNK_SIZE = 5000  # 每次切分发送给模型的 token 数
RECURRENT_MAX_NEW = 256  # 控制模型回复的最大 token 数

# 定义提示模板，引导模型只输出更新后的记忆
TEMPLATE = """You are presented with a problem, a section of an article, and a previous memory.
Update the memory strictly as follows:

1. Read the section carefully.
2. Extract only the information useful to answer the problem.
3. Merge it with the previous memory if necessary.
4. Output only the new memory, on a single line, starting exactly with:
Updated memory: 

⚠️ Do not add explanations, reasoning, or any extra words.
⚠️ Do not output anything else except the updated memory.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:"""

NO_MEMORY = "No previous memory"  # 当没有历史记忆时使用的占位字符串


def clip_long_string(string, max_length=2000):
    # 截断过长字符串，保留首尾用于日志展示
    if not len(string) > max_length:
        return string  # 长度在阈值内，直接返回
    target_len = max_length - len("\n\n...(truncated)\n\n")  # 计算截断后可用的长度
    return string[: target_len // 2] + "\n\n...(truncated)\n\n" + string[-target_len // 2 :]  # 拼接保留片段和提示


async def async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95):  # 定义异步函数用于循环调用模型
    # 定义异步函数，用于分块调用模型并更新记忆
    idx = item["_id"]  # 读取样本的唯一标识，方便调试
    context = item["context"].strip()  # 获取上下文并去除首尾空白
    prompt = item["input"].strip()  # 获取任务描述并去除首尾空白

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400)) as session:  # 创建长超时的 HTTP 会话
        max_len = RECURRENT_MAX_CONTEXT_LEN  # 读取允许的最大上下文长度
        input_ids = tokenizer.encode(context)  # 将上下文编码为 token id 列表
        if len(input_ids) > max_len:  # 若超过最大长度，则执行截断
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]  # 保留前后各一半，丢弃中间部分

        memory = NO_MEMORY  # 初始化记忆占位符
        for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):  # 按配置的步长遍历所有 token
            chunk = input_ids[i : i + RECURRENT_CHUNK_SIZE]  # 取出当前分块的 token 序列
            msg = TEMPLATE.format(prompt=prompt, chunk=tokenizer.decode(chunk), memory=memory)  # 将问题、记忆和片段填入模板

            print("=== USER PROMPT ===")  # 打印提示头用于调试
            print(clip_long_string(msg))  # 打印当前用户提示，必要时截断

            try:  # 捕获请求过程中可能出现的异常
                async with session.post(
                    url=URL + "/chat/completions",  # 指定 chat completions 接口
                    headers={"Authorization": f"Bearer {API_KEY}"},  # 填入认证头
                    json=dict(
                        model=model,  # 指定模型名称
                        messages=[{"role": "user", "content": msg}],  # 构造单轮对话消息
                        temperature=temperature,  # 设置采样温度
                        top_p=top_p,  # 设置核采样阈值
                        max_tokens=RECURRENT_MAX_NEW,  # 限制生成长度
                    ),
                ) as resp:  # 发送请求并等待响应
                    status = resp.status  # 读取 HTTP 状态码
                    if status != 200:  # 非 200 视为失败
                        print(f"ERROR {status=}, {model=}")  # 打印错误信息
                        return ""  # 返回空字符串中断流程
                    data = await resp.json()  # 解析 JSON 响应体
                    print("DEBUG chunk raw:", data)  # 打印原始响应便于调试
                    memory = data["choices"][0]["message"]["content"]  # 提取模型返回的更新记忆

                    print("=== ASSISTANT RESPONSE ===")  # 打印回复头
                    print(clip_long_string(memory))  # 打印模型输出，必要时截断

            except Exception as e:  # 捕获任何异常
                import traceback  # 按需导入 traceback 输出堆栈

                traceback.print_exc()  # 打印详细异常信息
                return ""  # 返回空字符串通知调用方失败

        return memory  # 所有分块处理完毕后返回最终记忆


if __name__ == "__main__":  # 当脚本以主程序方式运行时执行以下逻辑
    import argparse  # 导入 argparse 以解析命令行参数

    parser = argparse.ArgumentParser(description="quick start")  # 创建命令行解析器并设置描述
    parser.add_argument("--model", type=str, required=True, help="model name used in your deployment/model service endpoint")  # 添加模型名称参数
    parser.add_argument("--chunk", type=int, default=RECURRENT_CHUNK_SIZE, help="chunk size of context")  # 添加上下文分块大小参数
    parser.add_argument("--max_new", type=int, default=RECURRENT_MAX_NEW, help="max new tokens, also the max length of memory")  # 添加生成长度参数
    args = parser.parse_args()  # 解析命令行参数

    RECURRENT_CHUNK_SIZE = args.chunk  # 使用命令行参数覆盖默认分块大小
    RECURRENT_MAX_NEW = args.max_new  # 使用命令行参数覆盖默认生成长度
    model = args.model  # 提取模型名称

    tok = AutoTokenizer.from_pretrained(model)  # 根据模型名称加载对应的分词器

    a = asyncio.run(  # 使用 asyncio.run 执行异步任务并获取结果
        async_query_llm(  # 调用异步逻辑处理示例数据
            {
                "context": "The passage says explicitly: The magic number is 1.",  # 示例上下文
                "input": "Please extract the magic number and update memory accordingly. Only output in the format: 'Updated memory: ...'.",  # 示例问题
                "_id": 0,  # 示例编号
            },
            model,  # 传入模型名称
            tok,  # 传入分词器实例
        )
    )
    print("FINAL OUTPUT:", a)  # 输出最终记忆结果
