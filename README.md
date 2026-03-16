# WD14 Tagger Minimal

一个独立的、面向人和 AI 的精简版 WD14 打标项目。

目标很单纯：

1. 指定一个图片文件夹
2. 用 WD14 做标签推理
3. 按类别补充你想强制加入的标签
4. 输出同名 `.txt` 标签文件

这个项目是独立的，核心逻辑都在本目录自己的 `run_wd14_tagger.py` 里。

## 目录结构

```text
wd14_tagger_minimal/
  AGENTS.md
  README.md
  requirements.txt
  meta_tag_black_list.txt
  run_wd14_tagger.py
  test_data/
    测试/
      ...
```

## 安装

建议单独创建环境，然后安装：

```bash
python -m pip install -r requirements.txt
```

最小依赖只有这些：

- `huggingface-hub`
- `numpy`
- `onnx`
- `onnxruntime`
- `pillow`
- `tqdm`

## 最小用法

```bash
python run_wd14_tagger.py --input_dir /你的图片目录
```

默认行为：

- 默认镜像：`https://hf-mirror.com`
- 默认模型：`SmilingWolf/wd-v1-4-convnext-tagger-v2`
- 默认递归扫描子目录
- 默认读取 `meta_tag_black_list.txt`
- 默认输出 `.txt`

## 测试命令

对自带测试集运行：

```bash
python run_wd14_tagger.py \
  --input_dir test_data/测试 \
  --caption_extension .wdtest.txt \
  --add_special_tags "1girl" \
  --add_character_tags "raiden shogun" \
  --add_copyright_tags "genshin impact" \
  --add_artist_tags "yoneyama mai" \
  --add_general_tags "solo, blue eyes" \
  --add_meta_tags "highres" \
  --add_rating_tags "safe" \
  --add_quality_tags "best quality, masterpiece"
```

## 主要参数

- `--input_dir`
  - 必填，输入图片目录
- `--mirror`
  - Hugging Face 镜像，默认 `https://hf-mirror.com`
- `--repo_id`
  - WD14 模型仓库
- `--model_dir`
  - 模型缓存目录
- `--batch_size`
  - 推理批大小
- `--caption_extension`
  - 输出标签文件扩展名
- `--meta_tag_blacklist_file`
  - meta 标签黑名单文件
- `--force_download`
  - 强制重新下载模型

## 支持的标签追加类别

支持 8 类追加标签：

- `--add_special_tags`
- `--add_character_tags`
- `--add_copyright_tags`
- `--add_artist_tags`
- `--add_general_tags`
- `--add_meta_tags`
- `--add_rating_tags`
- `--add_quality_tags`

输出顺序固定为：

1. `special`
2. `character`
3. `copyright`
4. `artist`
5. `general`
6. `meta`
7. `rating`
8. `quality`

## 黑名单

`meta_tag_black_list.txt` 会被自动加载。

用途：

- 过滤模型预测出的 meta 标签
- 过滤你手动通过 `--add_meta_tags` 等加入的标签

## 输出

每张图会在旁边生成同名标签文件。

例如：

```text
1_a/example.jpg
1_a/example.txt
```

如果你设成 `--caption_extension .wdtest.txt`，则会输出：

```text
1_a/example.jpg
1_a/example.wdtest.txt
```
