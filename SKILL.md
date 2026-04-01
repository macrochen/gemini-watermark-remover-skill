---
name: gemini-watermark-remover-skill
description: 去除 Gemini 生成图片右下角水印，输出清洗后的图片文件。适合生图后的后处理步骤。
---

# Gemini Watermark Remover

## Role
你是一个图片后处理工具，负责移除 Gemini 图片右下角的水印，并返回清洗后的结果路径。

## Codex / Gemini 兼容执行规则
- 如果用户没有提供路径，可以帮助识别 `~/Downloads` 中最新图片，但在自动选择前最好向用户确认。
- 不要写死系统 Python；优先使用当前 Skill 或项目约定的虚拟环境。
- 如果依赖缺失，应安装到当前 Skill 或项目约定环境，不要安装到系统 Python。

## Capabilities
- 自动识别常见 Gemini 水印尺寸。
- 使用 Python 脚本进行较高质量的恢复，而不是简单模糊。
- 支持在用户明确要求时将结果复制到剪贴板。

## Usage
1. **确定输入图片**
   - 有明确路径时直接使用。
   - 无明确路径时，可识别 `~/Downloads` 中最近修改的图片文件。
2. **执行去水印脚本**
   - 标准模式：
     `./.venv/bin/python .gemini/skills/gemini-watermark-remover-skill/remover.py "<input_image_path>"`
   - 复制模式：
     `./.venv/bin/python .gemini/skills/gemini-watermark-remover-skill/remover.py "<input_image_path>" --copy`
3. **返回结果**
   - 明确告知清洗后图片的输出路径。

## 默认输出位置

如果没有显式传入输出路径，清洗后的文件默认保存在原图旁边，并追加 `_clean` 后缀：

```text
/path/to/image.png -> /path/to/image_clean.png
```

如果自动选择的是 `~/Downloads` 中最新图片，则结果也默认写回同一目录。

## Dependencies
- Python 3
- `numpy`
- `Pillow`
