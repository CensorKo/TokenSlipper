#!/usr/bin/env python3
"""
🩴 TokenSlipper Logo 生成器
简约风格：拖鞋轮廓 + Token 元素
"""

from PIL import Image, ImageDraw, ImageFont
import os

# 创建画布 (1024x1024 适合各种用途)
SIZE = 1024
CENTER = SIZE // 2

# 配色方案 - 科技感的蓝紫渐变
colors = {
    'bg_start': (102, 126, 234),      # #667eea - 淡紫
    'bg_end': (118, 75, 162),         # #764ba2 - 深紫
    'slipper': (255, 255, 255),       # 白色拖鞋
    'token': (255, 255, 255),         # 白色 Token 符号
    'shadow': (0, 0, 0, 30),          # 轻微阴影
}

def create_gradient_background(size, color1, color2):
    """创建渐变背景"""
    img = Image.new('RGB', (size, size), color1)
    draw = ImageDraw.Draw(img)
    
    for y in range(size):
        # 计算渐变比例
        ratio = y / size
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        draw.line([(0, y), (size, y)], fill=(r, g, b))
    
    return img

def draw_rounded_slipper(draw, cx, cy, width, height, radius, fill_color, shadow_color=None):
    """绘制圆角拖鞋形状"""
    # 阴影
    if shadow_color:
        shadow_offset = 20
        draw.rounded_rectangle(
            [cx - width//2 + shadow_offset, cy - height//2 + shadow_offset,
             cx + width//2 + shadow_offset, cy + height//2 + shadow_offset],
            radius=radius,
            fill=shadow_color
        )
    
    # 拖鞋主体 - 椭圆形的简约拖鞋
    draw.rounded_rectangle(
        [cx - width//2, cy - height//2,
         cx + width//2, cy + height//2],
        radius=radius,
        fill=fill_color
    )

def draw_token_symbol(draw, cx, cy, size, color):
    """绘制 Token 符号 </>"""
    # 使用简洁的 </> 表示代码/Token
    line_width = max(8, size // 20)
    gap = size // 6
    
    # < 符号 (左)
    left_x = cx - gap
    draw.line([(left_x - size//4, cy - size//3), (left_x, cy)], fill=color, width=line_width)
    draw.line([(left_x, cy), (left_x - size//4, cy + size//3)], fill=color, width=line_width)
    
    # > 符号 (右)
    right_x = cx + gap
    draw.line([(right_x, cy - size//3), (right_x + size//4, cy)], fill=color, width=line_width)
    draw.line([(right_x + size//4, cy), (right_x, cy + size//3)], fill=color, width=line_width)
    
    # 中间斜杠 /
    draw.line([(cx - size//12, cy + size//3), (cx + size//12, cy - size//3)], fill=color, width=line_width)

def create_logo_with_text():
    """创建带文字的完整 Logo"""
    # 创建画布
    img = create_gradient_background(SIZE, colors['bg_start'], colors['bg_end'])
    draw = ImageDraw.Draw(img)
    
    # 绘制拖鞋形状 (中心位置)
    slipper_width = 400
    slipper_height = 600
    slipper_radius = 120
    
    # 拖彖稍微倾斜，增加动感
    draw_rounded_slipper(
        draw, CENTER, CENTER + 50,
        slipper_width, slipper_height, slipper_radius,
        colors['slipper'],
        colors['shadow']
    )
    
    # 在拖鞋上绘制 Token 符号
    draw_token_symbol(draw, CENTER, CENTER + 50, 200, colors['bg_start'])
    
    return img

def create_icon_only():
    """创建仅图标的 Logo (用于 favicon/头像)"""
    img = create_gradient_background(SIZE, colors['bg_start'], colors['bg_end'])
    draw = ImageDraw.Draw(img)
    
    # 拖鞋形状
    slipper_width = 500
    slipper_height = 700
    slipper_radius = 150
    
    draw_rounded_slipper(
        draw, CENTER, CENTER,
        slipper_width, slipper_height, slipper_radius,
        colors['slipper'],
        colors['shadow']
    )
    
    # Token 符号
    draw_token_symbol(draw, CENTER, CENTER, 250, colors['bg_start'])
    
    return img

def create_banner():
    """创建横向 Banner (用于 README/GitHub 封面)"""
    width = 2400
    height = 800
    center_x = width // 2
    center_y = height // 2
    
    img = Image.new('RGB', (width, height), colors['bg_start'])
    draw = ImageDraw.Draw(img)
    
    # 渐变背景
    for x in range(width):
        ratio = x / width
        r = int(colors['bg_start'][0] * (1 - ratio) + colors['bg_end'][0] * ratio)
        g = int(colors['bg_start'][1] * (1 - ratio) + colors['bg_end'][1] * ratio)
        b = int(colors['bg_start'][2] * (1 - ratio) + colors['bg_end'][2] * ratio)
        draw.line([(x, 0), (x, height)], fill=(r, g, b))
    
    # 左侧拖鞋图标
    icon_size = 300
    draw_rounded_slipper(
        draw, 400, center_y,
        icon_size, int(icon_size * 1.5), 80,
        colors['slipper'],
        colors['shadow']
    )
    draw_token_symbol(draw, 400, center_y, 150, colors['bg_start'])
    
    # 右侧文字 - 尝试加载系统中文字体
    def get_font(size, is_chinese=False):
        font_paths = [
            # macOS 中文字体
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            # Linux 中文字体
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            # Windows 中文字体
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            # 通用字体
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
        return ImageFont.load_default()
    
    title_font = get_font(120)
    slogan_font = get_font(48, is_chinese=True)
    
    # 主标题
    title = "TokenSlipper"
    draw.text((750, 280), title, fill=colors['slipper'], font=title_font)
    
    # 标语
    slogan = "让大模型 API 使用回归理性"
    draw.text((750, 450), slogan, fill=(255, 255, 255, 200), font=slogan_font)
    
    return img

def save_logo(img, filename, sizes=None):
    """保存并生成多种尺寸"""
    if sizes is None:
        sizes = [1024, 512, 256, 128, 64, 32]
    
    # 保存原始尺寸
    img.save(f"logo/{filename}.png", "PNG")
    
    # 生成不同尺寸
    for size in sizes:
        if size != 1024:
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(f"logo/{filename}_{size}.png", "PNG")
    
    print(f"✅ {filename} 已生成，尺寸: {sizes}")

def main():
    """主函数"""
    # 创建输出目录
    os.makedirs("logo", exist_ok=True)
    
    print("🎨 正在生成 TokenSlipper Logo...")
    print()
    
    # 1. 图标 Logo
    print("1️⃣  生成图标 Logo...")
    icon_logo = create_icon_only()
    save_logo(icon_logo, "icon")
    
    # 2. 横幅 Banner
    print("2️⃣  生成横幅 Banner...")
    banner = create_banner()
    banner.save("logo/banner.png", "PNG")
    banner.save("logo/banner.jpg", "JPEG", quality=95)
    print("✅ banner 已生成")
    
    print()
    print("=" * 50)
    print("🩴 TokenSlipper Logo 生成完成！")
    print("=" * 50)
    print()
    print("📁 输出文件：")
    print("   logo/icon.png          - 主图标 (1024x1024)")
    print("   logo/icon_*.png        - 各种尺寸图标")
    print("   logo/banner.png        - GitHub 封面横幅")
    print()
    print("🎨 配色方案：")
    print("   主色: #667eea (淡紫) -> #764ba2 (深紫)")
    print("   辅色: #ffffff (白色)")
    print()

if __name__ == "__main__":
    main()
