import json
import os
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
import requests
import dashscope
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message

# ==========================================
MODEL_NAME = 'wan2.7-image-pro'
IMAGE_SIZE = '2K'
PROMPT_FILE = 'temporal_prompts.json'
OUTPUT_ROOT = 'temporal_results'
# ==========================================

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


def download_image(url: str, filename: Path) -> None:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f'保存成功: {filename}')
    except Exception as e:
        print(f'下载失败 {filename}: {e}')


def load_temporal_prompts(prompt_file: Path):
    data = json.loads(prompt_file.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError(f'Expected JSON array in {prompt_file}')
    return data


def build_prompt_text(item: dict) -> str:
    prompt = str(item.get('prompt', '')).strip()
    if prompt:
        return ' '.join(prompt.split())

    global_prompt = ' '.join(str(item.get('global_prompt', '')).split())
    sub_prompts = [' '.join(str(x).split()) for x in item.get('sub_prompts', []) if str(x).strip()]
    parts = [p for p in [global_prompt, *sub_prompts] if p]
    return ' '.join(parts)


def parse_num_images(item: dict) -> int:
    num_images = item.get('num_images', 1)
    try:
        num_images = int(num_images)
    except Exception:
        num_images = len(item.get('sub_prompts', [])) or 1
    return max(1, num_images)


def run_single_inference(api_key: str, item: dict, output_root: Path) -> None:
    prompt_id = str(item.get('id', 'unknown'))
    prompt_text = build_prompt_text(item)
    image_count = parse_num_images(item)
    save_dir = output_root / prompt_id

    if not prompt_text:
        print(f'跳过 {prompt_id}: prompt 为空')
        return

    print(f'\n==== {prompt_id} ====')
    print(f'张数={image_count}, 尺寸={IMAGE_SIZE}')

    message = Message(role='user', content=[{'text': prompt_text}])

    response = ImageGeneration.async_call(
        model=MODEL_NAME,
        api_key=api_key,
        messages=[message],
        enable_sequential=True,
        n=image_count,
        size=IMAGE_SIZE,
    )

    if response.status_code != HTTPStatus.OK:
        print(f'任务提交失败: {prompt_id}')
        print(f'错误码: {response.code}')
        print(f'错误信息: {response.message}')
        return

    task_id = response.output.task_id
    print(f'任务已提交, ID: {task_id}')

    status = ImageGeneration.wait(task=response, api_key=api_key)
    if status.output.task_status != 'SUCCEEDED':
        print(f'任务执行失败, 状态: {status.output.task_status}')
        if hasattr(status, 'code') and status.code:
            print(f'错误码: {status.code}')
        if hasattr(status, 'message') and status.message:
            print(f'错误信息: {status.message}')
        return

    print('任务执行成功, 开始处理...')
    try:
        scheduled_time_str = status.output.get('scheduled_time') if hasattr(status.output, 'get') else getattr(status.output, 'scheduled_time', None)
        end_time_str = status.output.get('end_time') if hasattr(status.output, 'get') else getattr(status.output, 'end_time', None)
        if scheduled_time_str and end_time_str:
            def parse_time(t_str):
                fmt = '%Y-%m-%d %H:%M:%S.%f' if '.' in t_str else '%Y-%m-%d %H:%M:%S'
                return datetime.strptime(t_str, fmt)
            actual_duration = (parse_time(end_time_str) - parse_time(scheduled_time_str)).total_seconds()
            print(f'模型生成耗时: {actual_duration:.2f} 秒')
    except Exception as e:
        print(f'时间解析异常: {e}')

    save_dir.mkdir(parents=True, exist_ok=True)
    content_list = status.output.choices[0].message.content
    image_idx = 1
    for item_content in content_list:
        img_url = item_content.get('image')
        if not img_url:
            continue
        filename = save_dir / f'{image_idx}.png'
        download_image(img_url, filename)
        image_idx += 1


def main() -> None:
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("请设置环境变量: DASHSCOPE_API_KEY")
        return

    base_dir = Path(__file__).resolve().parent
    prompt_file = base_dir / PROMPT_FILE
    output_root = base_dir / OUTPUT_ROOT

    if not prompt_file.exists():
        print(f'未找到输入文件: {prompt_file}')
        return

    prompts = load_temporal_prompts(prompt_file)
    print(f'读取到 {len(prompts)} 条 prompt')

    for item in prompts:
        try:
            run_single_inference(api_key, item, output_root)
        except Exception as e:
            prompt_id = item.get('id', 'unknown')
            print(f'{prompt_id} 运行异常: {e}')

    print('\n流程结束')


if __name__ == '__main__':
    main()
