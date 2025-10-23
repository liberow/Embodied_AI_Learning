'''
@Desc    :   GRPO with DDP for Qwen2.5-1.5B-Instruct

    State: 当前上下文, 即输入 和 已生成的所有 token（包括prompt和completion）。
    Action: 下一个要生成的 token。

'''
################################ Part 1: Basic Setup and Imports
# Basic
import re
import os
import copy
import wandb
import random 
import datetime
import tempfile
import numpy as np

# 设置可见 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# deep learning
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 设置 HuggingFace 镜像和缓存路径
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = ""

# Hugging Face
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def set_random_seed(seed: int = 0):
    """
    设置随机种子以确保在Python、NumPy和PyTorch中的可重复性。

    Args:
        seed (int): 用于随机数生成的种子值。

    Returns:
        None

    Explanation:
        1. 为Python内置的random模块设置种子，用于基本的随机操作。
        2. 为NumPy设置种子，确保在数组操作中生成一致的随机数。
        3. 为PyTorch的CPU操作设置种子。
        4. 如果CUDA可用，为所有GPU设备设置种子。
        5. 配置cuDNN以确保确定性行为：
           - 将确定性标志设置为True，确保结果可重复。
           - 禁用基准测试，以防止基于硬件的算法选择。

    Note:
        设置确定性行为可能会影响性能，但可以确保在多次运行中获得一致的结果，
        这对于调试和研究至关重要。
    """

    # 为Python内置的random模块设置种子
    random.seed(seed)
    # 为NumPy设置种子
    np.random.seed(seed)
    # 为PyTorch设置种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 确保cuDNN的确定性行为（可能会影响性能）    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(2025)

os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_PROJECT"] = "GRPO-DDP-Qwen2.5-1.5B-Instruct"   

################################ Part 2: Data Formatting and Answer Extraction

SYSTEM_PROMPT = """ 
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_answer_from_model_output(text):
    """
    从文本中最后一个 <answer> 标签中提取值。

    Args:
        text (str): 包含 XML 样式 <answer> 标签的模型生成文本。

    Returns:
        str 或 None: <answer> 标签内的内容，如果未找到有效答案则返回 None。

    Explanation:
        1. 在 <answer> 标签处拆分文本，以隔离标签后的内容。
        2. 检查文本中是否至少存在一个 <answer> 标签。
        3. 对于最后一个 <answer> 部分：
            - 验证其是否包含闭合的 </answer> 标签。
            - 仅提取标签之间的内容。
        4. 如果答案为空（仅为 "..."）或缺少标签，则返回 None。
    """
    parts = text.split("<answer>")
    if len(parts) < 2:
        return None
    last_part = parts[-1]

    if "</answer>" not in last_part:
        return None
   
    answer = last_part.split("</answer")[0].strip()
    return None if answer == "..." else answer

def extract_answer_from_dataset(text):
    """
    从 GSM8K 数据集示例中提取答案。

    Args:
        text (str): 包含问题和答案的数据集示例文本。

    Returns:
        str 或 None: 提取 '####' 分隔符后的答案部分，如果未找到则返回 None。

    Explanation:
        1. 检查文本是否包含分隔问题和答案的 '####' 分隔符。
        2. 如果找到，则在分隔符处拆分文本并返回第二部分（即答案）。
        3. 答案会去除前导和尾随空格。
        4. 如果不存在分隔符，则返回 None。
    """      
    if "####" not in text:
        return None
    return text.split("####")[-1].strip()

################################ Part 3: Dataset Preparation

def build_prompt(messages):
    """
    从消息列表构建单个提示字符串。

    Args:
        messages (list): 一个消息字典列表，每个字典包含 'role' 和 'content' 键。

    Returns:
        str: 所有消息内容连接成的字符串。

    Explanation:
        1. 接收一个典型的聊天格式的消息字典列表。
        2. 从每条消息中提取 'content' 字段并去除空白字符。
        3. 将所有内容字符串用换行符连接，创建一个单一的提示。
        4. 这保留了训练格式，同时将结构化消息转换为字符串。
    """
    return "\n".join(msg["content"].strip() for msg in messages)

def prepare_dataset(split="train"):
    """
    加载并准备 GSM8K 数据集，用于使用字符串提示进行训练。

    Args:
        split (str): 要加载的数据集划分（"train" 或 "test"）。默认为 "train"。

    Returns:
        list: 一个包含格式化示例的列表，每个示例包含一个提示字符串和答案。

    Explanation:
        1. 从 Hugging Face 数据集中心加载 GSM8K 数据集。
        2. 对于数据集中的每个示例：
            - 创建一个包含系统提示和问题的消息列表。
            - 使用 build_prompt() 将此列表转换为单个字符串提示。
            - 从数据集示例中提取答案。
            - 创建一个包含提示和答案的格式化示例字典。
        3. 返回准备好的格式化示例列表，可用于模型训练或评估。
    """
    data = load_dataset("openai/gsm8k", "main")[split] 
    formatted_data = []
    for example in data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]}
        ]
        prompt_str = build_prompt(messages)
        formatted_example = {
            "prompt": prompt_str,
            "answer": extract_answer_from_dataset(example["answer"]),
        }
        formatted_data.append(formatted_example)

    return formatted_data

################################ Part 4: Evaluation Functions  
  
def extract_last_number(text):
    """
    从文本中提取最后一个出现的数字。

    Args:
        text (str): 要从中提取数字的文本。

    Returns:
        float 或 None: 文本中的最后一个数字，如果未找到数字则返回 None。

    Explanation:
        1. 从文本中移除美元符号和百分号。
        2. 使用正则表达式查找出现在文本末尾的数字（可能在空白字符之后）。
        3. 该模式匹配出现在字符串末尾的数字，可以包含或不包含小数点。
        4. 返回找到的数字作为浮点数，如果未找到匹配项则返回 None。
    """
    text = text.replace('$', '').replace('%', '')
    pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$' # 用于匹配文本中的最后一个数字。
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def extract_single_number(text):
    """
    如果文本中恰好存在一个数字，则从中提取该数字。

    参数:
        text (str): 要从中提取数字的文本。

    返回:
        float 或 None: 文本中的单个数字，如果未找到或找到多个数字则返回 None。

    解释:
        1. 使用正则表达式查找文本中的所有数字（包括负数和小数）。
        2. 如果恰好找到一个数字，则将其作为浮点数返回。
        3. 如果未找到或找到多个数字，则返回 None。
    """
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return float(numbers[0]) if len(numbers) == 1 else None

def evaluate_model(model, tokenizer, eval_data):
    """
    在示例集上评估模型并打印详细结果。

    参数:
        model: 要评估的语言模型。
        tokenizer: 用于编码输入和解码输出的分词器。
        eval_data (list): 评估示例列表，每个示例包含 "prompt" 和 "answer"。

    返回:
        float: 准确率百分比（正确预测数 / 总示例数 * 100）。

    解释:
        1. 将模型设置为评估模式。
        2. 对于评估集中的每个示例：
            - 对提示进行编码并使用模型生成响应。
            - 从生成的响应中提取预测答案。
            - 使用多种方法将预测答案与预期答案进行比较：
                a. 精确字符串匹配
                b. 提取单个数字并进行比较
                c. 提取最后一个数字并进行比较
            - 打印每个示例的详细信息。
        3. 计算并返回总体准确率。
        4. 将模型恢复为训练模式。
    """
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = len(eval_data)
    print("\n" + "=" * 100)
    print("EVALUATION ON", total, "EXAMPLES")
    print("=" * 100)

    for example in eval_data:
        full_prompt = example["prompt"]
        expected = example["answer"]

        # Tokenize and generate response
        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)   
        try:
            #Extract answer and check correctness
            predicted = extract_answer_from_model_output(response)

            # Try different matching methods
            if predicted == expected:
                is_correct =True
            else:
                # Try single number matching
                pred_num = extract_single_number(str(predicted))
                exp_num = extract_single_number(str(expected)) ### ？？ 这个是否有必要，对于 expected  不用提取吧，已经是数字了
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    # Try last number matching
                    pred_num = extract_last_number(str(predicted))
                    exp_num = extract_last_number(str(expected)) ### ？？ 这个是否有必要，对于 expected  不用提取吧，已经是数字了
                    is_correct = (pred_num is not None and 
                                exp_num is not None and
                                pred_num == exp_num) 
            if is_correct:
                correct += 1  

            # Print evaluation details
            print("\nPrompt:")
            print(full_prompt)
            print("\nExpected Answer:")
            print(expected)
            print("\nExtracted Answer:")
            print(predicted)
            print("\nFull Generated Responsse:")
            print(response)
            print("\nCorrect:", "✓" if is_correct else "✗")
            print("-" * 100)

        except Exception as e:
            print("\nFailed to parse model output for prompt")
            print(full_prompt)
            print("\nError:", e)      
            print("-" * 100)

    # Calculate and print final accuracy
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print("=" * 100)    

    # Return model to training mode
    model.train()
    return accuracy

def evaluate_ddp(model, tokenizer, eval_data, rank):
    """分布式评估"""
    if rank == 0:
        # 只在主进程上执行评估
        model.eval()

        eval_subset = eval_data[:min(50, len(eval_data))]
        accuracy = evaluate_model(model, tokenizer, eval_subset)
        model.train()
        return accuracy
    return None

################################ Part 5: Reward Functions
    
def correctness_reward(completions, answer):
    """
    根据模型答案的正确性分配奖励。

    Args:
        completions (list): 模型生成的补全列表，每个补全包含内容。
        answer (list): 预期答案列表。

    Returns:
        list: 每个补全的数值奖励列表。

    Explanation:
        1. 从每个补全中提取内容。
        2. 使用 extract_answer_from_model_output 从每个响应中提取答案部分。
        3. 根据匹配标准分配奖励：
            - 精确匹配得 2.0 分
            - 数值相等但格式不同得 1.5 分
            - 错误答案得 0.0 分
        4. 跟踪补全长度以进行分析。
    """
    # responses = [completion[0]['content'] for completion in completions]
    # responses = [completion if isinstance(completion, str) else completion[0]['content'] for completion in completions]
    responses = []
    for completion in completions:
        if isinstance(completion, str):
            responses.append(completion)
        elif isinstance(completion, list) and isinstance(completion[0], dict):  ### ？？为什么要做这个判断
            responses.append(completion[0]['content'])
        else: 
            print("Unexcepted completion format: ", completion)
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        if r == a:
            rewards.append(2.0)
        else:
            r_num = extract_single_number(str(r))
            a_num = extract_single_number(str(a))    
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                r_num = extract_last_number(str(r))
                a_num = extract_last_number(str(a))
                if r_num is not None and a_num is not None and r_num == a_num:
                    rewards.append(1.5)
                else:
                    rewards.append(0.0)
    return rewards  
            
def format_reward(completions):
    """
    为符合所需的 XML 格式分配奖励。

    Args:
        completions (list): 模型生成的补全列表，每个补全包含内容。

    Returns:
        list: 每个补全的格式合规性分数列表。

    Explanation:
        1. 从每个补全中提取内容。
        2. 通过检查所需的 XML 标签来评估格式合规性：
            - 每个标签（<reasoning>, </reasoning>, <answer>, </answer>）存在得 0.2 分
            - 完美格式合规性最高得 0.8 分
        3. 存储并返回格式合规性分数。
    """
    # responses = [completion[0]['content'] for completion in completions]
    responses = []
    for completion in completions:
        if isinstance(completion, str):
            responses.append(completion)
        elif isinstance(completion, list) and isinstance(completion[0], dict):
            responses.append(completion[0]['content'])
        else: 
            print("Unexcepted completion format: ", completion)
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "</answer>" in response: score += 0.2
        rewards.append(score)

    return rewards

def combined_reward(completions, answer):
    """
    结合正确性和格式奖励。

    Args:
        completions (list[list[dict]]): 补全字典列表
        answer (list[str]): 预期答案列表

    Returns:
        list[float]: 每个提示-补全对的组合奖励

    Explanation:
        1. 分别计算正确性和格式合规性的奖励。
        2. 使用以下权重组合奖励：
            - 正确性分数范围：0.0 到 2.0
            - 格式分数范围：0.0 到 0.8
            - 总分范围：0.0 到 2.8
        3. 返回每个示例的组合奖励。
    """
    # Get individual rewards
    correctness_scores = correctness_reward(completions, answer)
    format_scores = format_reward(completions)

    # Combine rewards - correctness is weighted more heavily
    combined_rewards = []
    for c_score, f_score in zip(correctness_scores, format_scores):
        combined_rewards.append(c_score + f_score)
    
    return combined_rewards

################################ Part 6: GRPO From Scratch

def selective_log_softmax(logits, input_ids):
    """
    计算 completion token 的对数概率，用于计算概率比率和 KL 惩罚

    参数:
        logits (torch.Tensor): 模型输出的原始 logits，shape: (batch_size, seq_len, vocab_size)
        input_ids (torch.Tensor): 需要计算对数概率的 token ID，shape: (batch_size, seq_len)

    返回:
        torch.Tensor: 所选 token 的对数概率，shape: (batch_size, seq_len)

    解释:
        1. 应用 log softmax 将 logits 转换为词汇表上的对数概率。
        2. 使用 gather 提取与 input_ids 对应的对数概率，shape: (batch_size, seq_len, 1)，
           然后squeeze(-1)得到shape: (batch_size, seq_len)
        3. 移除额外维度以匹配 input_ids 的原始形状。
    """
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    """
    计算当前策略对所执行动作（completion token）的对数概率, 不包括 prompt token 

    参数:
        model: 语言模型。
        input_ids (torch.Tensor): 输入序列的 token ID。
        attention_mask (torch.Tensor): 输入序列的注意力掩码。
        logits_to_keep (int): 从序列末尾保留的 token 数量。

    返回:
        torch.Tensor: 所选 token 的对数概率。

    解释:
        1. 获取输入序列的模型 logits。
        2. 选择除最后一个 token 外的所有 logits（因为预测下一个 token）。
        3. 从 logits 和 input_ids 中选择最后 'logits_to_keep' 个 token。
        4. 使用 selective_log_softmax 计算这些 token 的对数概率。
    """
    device = next(model.parameters()).device 
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)

def create_completion_mask(completion_ids, eos_token_id):
    """
    为补全 token 创建掩码，排除 EOS token 之后的 token。

    参数:
        completion_ids (torch.Tensor): 生成的补全 token ID。
        eos_token_id (int): 序列结束 token 的 ID。

    返回:
        torch.Tensor: 二进制掩码，有效 token 为 1，EOS token 之后为 0。

    解释:
        1. 识别每个序列中 EOS token 出现的位置。
        2. 找到每个序列中第一个 EOS token 的索引。
        3. 创建一个掩码，第一个 EOS token 及其之前的位置为 1，其他位置为 0。
        4. 如果序列中没有 EOS token，则所有位置设置为 1。
    """
    is_eos = completion_ids == eos_token_id # is_eos(bool tensor) 标记哪些 token 是 eos token
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists= is_eos.any(dim=1) # 在序列维度上检查是否存在 True（即是否存在 EOS token）
    # is_eos.int(): 将布尔张量转换为整数张量（True -> 1，False -> 0）。
    # argmax(dim=1): 在序列维度上找到第一个 1（即第一个 EOS token）的索引。
    # [mask_exists]: 只更新包含 EOS token 的序列。
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()

def generate_completions(model, tokenizer, prompts, 
                         num_generations=4, max_completion_length=32):
    """
    为每个 prompt 生成多个 回复(completion)。

    参数:
        model: 语言模型。
        tokenizer: 用于编码和解码文本的分词器。
        prompts (list): 文本 prompt 列表。
        num_generations (int): 每个 prompt 生成的回复(completion)数量。
        max_completion_length (int): 生成的最大 token 数量。

    返回:
        tuple: 包含 prompt ID、 prompt mask、completion id 和 completion mask 。

    解释:
        1. 对 prompt 进行编码并将其移动到适当的设备。
        2. 将每个 prompt 重复 num_generations 次以生成多个回复。
        3. 使用模型生成回复，并指定参数。
        4. 提取回复 ID(不包括prompt token).
        5. 使用 create_completion_mask 创建补全掩码。
    """
    device = next(model.parameters()).device # 获取当前设备
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    print(f"Input batch size: {prompt_ids.size(0)}, Device before model: {prompt_ids.device}")
    prompt_length = prompt_ids.size(1)
    # 将每个 prompt 重复 num_generations 次,  [1, T] -> [1 * num_generations, T]
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0) 
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=False,
    )
    print(f"Output batch size: {outputs.size(0)}, Device after model: {outputs.device}")
    completion_ids = outputs[:, prompt_length:].to(device) # 获取生成的completion token id(去掉 prompt 部分)
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id).to(device)
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(model, ref_model, tokenizer, batch_samples, 
                          num_generations, max_completion_length):
    """
    生成 GRPO 回放数据，包括 completion 和对数概率。

    参数:
        model: 正在训练的模型。
        ref_model: 用于 KL 散度计算的参考模型。
        tokenizer: 用于编码和解码文本的分词器。
        batch_samples (list): 训练样本批次。
        num_generations (int): 每个样本生成的 completion 数量。
        max_completion_length (int): 最大 completion 长度。

    返回:
        dict: 包含 GRPO 更新所需的所有数据的字典。

    解释:
        1. 从批次样本中提取 prompt 和预期答案。
        2. 使用当前模型生成 Completion 。
        3. 组合 prompt 和 completion token。
        4. 计算模型和参考模型的对数概率。
        5. 格式化补全以进行奖励计算。
        6. 重复提示和答案以匹配生成的补全数量。
        7. 返回 GRPO 损失计算所需的所有数据。
    """
    device = next(model.parameters()).device
    ref_model = ref_model.to(device)

    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]
    with torch.no_grad():
        prompt_ids, prompts_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompts_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
    formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_token=True)}] for ids in completion_ids]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask, 
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations,
    }

def grpo_loss(model, ref_model, rollout_data, reward_function, 
              beta=0.01, epsilon=0.2):
    """
    计算 GRPO 损失以更新模型。

    参数:
        model: 正在训练的模型。
        ref_model: 用于 KL 散度计算的参考模型。
        rollout_data (dict): 由 generate_rollout_data 生成的数据。
        reward_function: 计算补全奖励的函数。
        beta (float): KL 惩罚系数。
        epsilon (float): PPO 裁剪参数。

    返回:
        torch.Tensor: 需要最小化的 GRPO 损失。

    解释:
        1. 使用模型计算当前 token 的对数概率。
        2. 计算当前策略和旧策略之间的概率比率。
        3. 使用 reward_function 计算奖励。
        4. 通过标准化奖励计算优势。
        5. 计算带有裁剪的 PPO 代理目标。
        6. 计算参考模型和策略模型之间的 KL 散度。
        7. 组合代理损失和 KL 惩罚。
        8. 对所有 token 和批次取平均损失。
    """
    device = next(model.parameters()).device
    ref_model = ref_model.to(device)
    input_ids = rollout_data["input_ids"].to(device)
    attention_mask = rollout_data["attention_mask"].to(device)
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs) # 计算当前策略和旧策略之间的概率比率
    rewards = torch.tensor(
        reward_function(completions=rollout_data["formatted_completions"], 
                        answer=rollout_data["repeated_answers"]),
        dtype=torch.float32,
        device=device,                
    )
    print(f"Rewards: {rewards}")  # Debug rewards
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    rewards = rewards.view(batch_size, num_generations)
    avg_reward = rewards.mean().item()
    print("Average Reward:", avg_reward)
    mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
    std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
    per_token_loss = surrogate_loss - beta * kl
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss, avg_reward

################################ Part 7: Optimization

def optimize_model_memory(model):
    """
    优化模型以减少训练期间的内存使用。

    参数:
        model: 要优化的语言模型。

    返回:
        优化后的模型。

    解释:
        1. 将模型设置为训练模式。
        2. 禁用 KV 缓存以节省内存。
        3. 启用梯度检查点以用计算换取内存。
        4. 确保输入嵌入需要梯度：
           - 如果模型支持内置方法，则使用该方法。
           - 否则，在输入嵌入层添加前向钩子。
        5. 返回优化后的模型，准备进行内存高效训练。
    """    
    model.train()
    model.config.use_cache = False # 禁用 KV Cache
    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)        

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model

################################ Part 8: DDP

def setup(rank, world_size):
    """
    初始化分布式训练环境
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12367'
    # 提高稳定性与可观测性
    os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')
    os.environ.setdefault('NCCL_DEBUG', 'INFO')
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')

    dist.init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size,
        timeout=datetime.timedelta(minutes=10),
    )
    torch.cuda.set_device(rank)

def cleanup():
    """
    清理分布式进程
    """
    dist.destroy_process_group()

def initialize_models_ddp(rank, model_name, checkpoint_path):
    """初始化模型和tokenizer的分布式设置"""
    # 所有进程都需要加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 各 rank 独立加载模型（从本地路径或 HF 缓存）；避免 barrier + checkpoint 带来的超时
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    model.config.pad_token_id = tokenizer.pad_token_id 
    model.config.eos_token_id = tokenizer.eos_token_id
    model = optimize_model_memory(model)

    model = model.to(rank)
    
    # 包装为DDP模型
    model = DDP(model, device_ids=[rank], output_device=rank)
    print(f"Rank {rank}: Model wrapped with DDP")
    
    return model, tokenizer

def train_grpo_step(model, ref_model, optimizer, batch_samples, tokenizer, 
                   reward_function, training_config, rank, step):
    """执行单个GRPO训练步骤"""
    device = rank  # 使用当前rank作为device ID
    
    # 生成rollout数据
    with torch.no_grad():
        rollout_data = generate_rollout_data(
            model.module,
            ref_model,
            tokenizer,
            batch_samples,
            training_config["num_generations"],
            training_config["max_completion_length"],
        )

    # 执行多次GRPO更新
    losses = []
    for grpo_iter in range(training_config["mu"]):
        # 计算GRPO损失
        loss, avg_reward = grpo_loss(
            model.module,
            ref_model,
            rollout_data,
            reward_function,
            beta=training_config["beta"],
            epsilon=training_config["epsilon"],
        )
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        
        losses.append(loss.item())
        
        if rank == 0:
            print("-" * 50)
            print(f"GRPO iter {grpo_iter + 1}/{training_config['mu']}, Step {step}, "
                  f"Loss: {loss.item():.4f}, Avg Reward: {avg_reward:.2f}")
    
    return sum(losses) / len(losses) if losses else 0.0

def train_with_grpo_ddp(rank, world_size, 
                        model_name, train_data, eval_data, 
                        reward_function, training_config):
    """分布式训练函数"""
    # 使用统一的检查点路径
    checkpoint_path = tempfile.gettempdir() + "/model.checkpoint"

    # 初始化分布式环境
    setup(rank, world_size)
    

    
    try:
        # 1. 初始化模型和tokenizer
        model, tokenizer = initialize_models_ddp(rank, model_name, checkpoint_path)
        
        # 2. 初始化WandB (只在rank 0)
        if rank == 0:
            wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True)
            wandb.config.update(training_config)
            print("Weight & Biases initialized.")
        
        # 3. 训练循环
        for iteration in range(training_config["num_iterations"]):
            if rank == 0:
                print(f"\n=== Iteration {iteration + 1}/{training_config['num_iterations']} ===")
            
            # 创建参考模型
            ref_model = copy.deepcopy(model.module).to(rank)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            
            # 初始化优化器
            optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["learning_rate"])
            model.train()
            
            # 训练步骤
            for step in range(training_config["num_steps"]):
                # 随机采样批次
                batch_samples = random.sample(train_data, training_config["batch_size"])
                
                # 训练步骤
                avg_loss = train_grpo_step(
                    model, ref_model, optimizer, batch_samples, tokenizer,
                    reward_function, training_config, rank, step,
                )
                
                # 定期评估和日志记录 (只在rank 0)
                if rank == 0 and (step + 1) % 1000 == 0:  # 每50步评估一次
                    eval_accuracy = evaluate_ddp(model.module, tokenizer, eval_data, rank)
                    wandb.log({
                        "iteration": iteration + 1,
                        "step": step + 1,
                        "loss": avg_loss,
                        "eval_accuracy": eval_accuracy if eval_accuracy else 0.0,
                    })
                    print(f"Iter {iteration + 1}, Step {step + 1}: "
                          f"Loss={avg_loss:.4f}, Eval Acc={eval_accuracy:.2f}%")
                elif rank == 0:
                    wandb.log({
                        "iteration": iteration + 1,
                        "step": step + 1,
                        "loss": avg_loss,
                    })
        
        # 4. 保存最终模型 (只在rank 0)
        if rank == 0:
            print("\nSaving final model...")
            model.module.save_pretrained("grpo_finetuned_model")
            tokenizer.save_pretrained("grpo_finetuned_model")
            wandb.finish()
            
    finally:
        # 清理
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        cleanup()
        print(f"Rank {rank}: Training completed.")

################################ Part 9: Training Setup and Execution

def main():
    # 评估GPU
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型配置
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPUs")
    
    # 准备数据
    all_data = prepare_dataset("train")
    random.shuffle(all_data)
    size_of_eval_data = 2 # 使用10个样本进行评估
    eval_data = all_data[:size_of_eval_data] 
    train_data = all_data[size_of_eval_data:]  
    
    # 训练配置
    training_config = {
        "num_iterations": 1,
        "num_steps": 100,  
        "batch_size": 1,
        "num_generations": 8,
        "max_completion_length": 200,
        "beta": 0.04,
        "learning_rate": 5e-6,
        "mu": 1,
        "epsilon": 0.1,
    }
   
    # 初始评估（避免在 spawn 前占用 GPU，若需要，可放入 rank 0 子进程内部执行）
    print("\nInitial evaluation:")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    evaluate_model(model, tokenizer, eval_data)  
    
    if world_size:
        # 分布式训练
        mp.spawn(
            train_with_grpo_ddp,
            args=(world_size, model_name, train_data, eval_data, 
                 combined_reward, training_config),
            nprocs=world_size,
            join=True,
        )
    else:
        print(f"Warining: The numbers of GPU is {world_size}. DDP must GPU > 0")
    
    
    # 最终评估
    if world_size:
        print("\nFinal evaluation:")
        model = AutoModelForCausalLM.from_pretrained(
            "grpo_finetuned_model",
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2", 
            attn_implementation="sdpa",  # 使用sdpa   
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained("grpo_finetuned_model")
        evaluate_model(model, tokenizer, eval_data)

if __name__ == "__main__":
    main()