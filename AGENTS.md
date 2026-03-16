# Project Notes

## Purpose

This is a standalone WD14 tagging project.

It should be understandable without reading any external repository.

## Main File

- `run_wd14_tagger.py`

That file contains:

- model download
- image preprocessing
- ONNX inference
- tag categorization
- meta blacklist filtering
- category-based manual tag injection

## Interface

Always use:

```bash
python run_wd14_tagger.py --input_dir test_data/测试
```

Do not convert `--input_dir` back into a positional argument.

## Tag Order

Keep the output order:

1. special
2. character
3. copyright
4. artist
5. general
6. meta
7. rating
8. quality

## Key Local Files

- `README.md`
- `requirements.txt`
- `meta_tag_black_list.txt`
- `test_data/测试`
