#!/usr/bin/env python3
"""
创建 🩴 Emoji Logo
"""

from PIL import Image, ImageDraw, ImageFont
import os

SIZE = 1024
BG_COLOR = (102, 126, 234)

img = Image.new('RGB', (SIZE, SIZE), BG_COLOR)
draw = ImageDraw.Draw(img)

# Noto Color Emoji 使用固定大小，尝试 109（这是字体支持的大小）
try:
    font = ImageFont.truetype("NotoColorEmoji.ttf", 109)
except:
    # 如果失败，尝试系统默认方式
    font = ImageFont.load_default()

# 获取文本大小
bbox = draw.textbbox((0, 0), "🩴", font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]

# 居中
x = (SIZE - text_width) // 2
y = (SIZE - text_height) // 2

# 绘制 Emoji
draw.text((x, y), "🩴", font=font, embedded_color=True)

# 保存
os.makedirs("logo", exist_ok=True)

sizes = [1024, 512, 256, 128, 64]
for s in sizes:
    resized = img.resize((s, s), Image.Resampling.LANCZOS)
    resized.save(f"logo/emoji_logo_{s}.png", "PNG")

img.save("logo/emoji_logo.png", "PNG")

print("✅ 🩴 Emoji Logo 生成完成！")
