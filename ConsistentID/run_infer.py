import argparse
import json
import os
from datetime import datetime
from http import HTTPStatus
from pathlib import Path

import dashscope
import requests
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message


dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / 'evaluation' / 'multi_image_prompts.json'
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / 'multi_image_results'
DEFAULT_MODEL_NAME = 'wan2.7-image-pro'
DEFAULT_IMAGE_SIZE = '2K'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run grouped image generation from rewritten multi-image prompt JSON.')
    parser.add_argument('--prompt-file', type=str, default=str(DEFAULT_PROMPT_FILE), help='Path to the multi-image prompt JSON file.')
    parser.add_argument('--output-root', type=str, default=str(DEFAULT_OUTPUT_ROOT), help='Directory to save grouped generation results.')
    parser.add_argument('--model-name', type=str, default=DEFAULT_MODEL_NAME, help='DashScope image generation model name.')
    parser.add_argument('--image-size', type=str, default=DEFAULT_IMAGE_SIZE, help='Image size passed to DashScope, e.g. 1024*1024 or 2K.')
    parser.add_argument('--limit', type=int, default=None, help='Only run the first N prompt entries.')
    return parser.parse_args()


def download_image(url: str, filename: Path) -> None:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'wb') as handle:
            handle.write(response.content)
        print(f'保存成功: {filename}')
    except Exception as error:
        print(f'下载失败 {filename}: {error}')


def load_prompt_items(prompt_file: Path):
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
    parts = [part for part in [global_prompt, *sub_prompts] if part]
    return ' '.join(parts)


def parse_num_images(item: dict) -> int:
    num_images = item.get('num_images', 1)
    try:
        num_images = int(num_images)
    except Exception:
        num_images = len(item.get('sub_prompts', [])) or 1
    return max(1, num_images)


def run_single_inference(api_key: str, item: dict, output_root: Path, *, model_name: str, image_size: str) -> None:
    prompt_id = str(item.get('id', 'unknown'))
    prompt_text = build_prompt_text(item)
    image_count = parse_num_images(item)
    save_dir = output_root / prompt_id

    if save_dir.exists():
        print(f'跳过 {prompt_id}: 输出目录已存在 {save_dir}')
        return

    if not prompt_text:
        print(f'跳过 {prompt_id}: prompt 为空')
        return

    print(f'\n==== {prompt_id} ====')
    print(f'张数={image_count}, 尺寸={image_size}, 输出目录={save_dir}')
    print(f'Prompt预览: {prompt_text[:200]}{"..." if len(prompt_text) > 200 else ""}')

    message = Message(role='user', content=[{'text': prompt_text}])

    response = ImageGeneration.async_call(
        model=model_name,
        api_key=api_key,
        messages=[message],
        enable_sequential=True,
        n=image_count,
        size=image_size,
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
            def parse_time(time_str):
                fmt = '%Y-%m-%d %H:%M:%S.%f' if '.' in time_str else '%Y-%m-%d %H:%M:%S'
                return datetime.strptime(time_str, fmt)
            actual_duration = (parse_time(end_time_str) - parse_time(scheduled_time_str)).total_seconds()
            print(f'模型生成耗时: {actual_duration:.2f} 秒')
    except Exception as error:
        print(f'时间解析异常: {error}')

    save_dir.mkdir(parents=True, exist_ok=True)
    content_list = status.output.choices[0].message.content
    image_index = 1
    for item_content in content_list:
        image_url = item_content.get('image')
        if not image_url:
            continue
        filename = save_dir / f'{image_index}.png'
        download_image(image_url, filename)
        image_index += 1


def main() -> None:
    args = parse_args()

    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print('请设置环境变量: DASHSCOPE_API_KEY')
        return

    prompt_file = Path(args.prompt_file)
    output_root = Path(args.output_root)

    if not prompt_file.exists():
        print(f'未找到输入文件: {prompt_file}')
        return

    prompt_items = load_prompt_items(prompt_file)
    if args.limit is not None:
        prompt_items = prompt_items[: args.limit]

    print(f'读取到 {len(prompt_items)} 条组图 prompt')
    print(f'输出目录: {output_root}')

    for item in prompt_items:
        try:
            run_single_inference(
                api_key,
                item,
                output_root,
                model_name=args.model_name,
                image_size=args.image_size,
            )
        except Exception as error:
            prompt_id = item.get('id', 'unknown')
            print(f'{prompt_id} 运行异常: {error}')

    print('\n流程结束')


if __name__ == '__main__':
    main()
