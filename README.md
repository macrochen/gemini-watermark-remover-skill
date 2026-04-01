# gemini-watermark-remover-skill

去除 Gemini 图片右下角水印，并输出清洗后的图片。

## Installation

This skill is intended to run in the local Codex/Gemini skill workspace.

If you are working in this repository, use the skill directly from:

```bash
.gemini/skills/gemini-watermark-remover-skill
```

## Documentation

# Gemini Watermark Remover

## Codex Compatibility
- 可以辅助识别 `~/Downloads` 中最新图片，但默认先确认用户是否接受自动选择。
- 默认优先使用当前 Skill 或项目约定的虚拟环境，不使用系统 Python。

## Command

```bash
./.venv/bin/python .gemini/skills/gemini-watermark-remover-skill/remover.py "<input_image_path>"
```

## 默认输出位置

如果没有传第二个输出路径参数，结果默认保存在原图旁边，并追加 `_clean` 后缀：

```text
/path/to/image.png -> /path/to/image_clean.png
```
