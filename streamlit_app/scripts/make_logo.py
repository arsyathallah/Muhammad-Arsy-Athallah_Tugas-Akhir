from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SIZE = 512
OUT  = Path(__file__).parent.parent / "assets" / "logo.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

mask = Image.new("L", (SIZE, SIZE), 0)
md   = ImageDraw.Draw(mask)
font = ImageFont.truetype("C:/Windows/Fonts/ariblk.ttf", 470)
bb   = md.textbbox((0, 0), "S", font=font)
w, h = bb[2] - bb[0], bb[3] - bb[1]
x    = (SIZE - w) / 2 - bb[0]
y    = (SIZE - h) / 2 - bb[1]
md.text((x, y), "S", fill=255, font=font)

top = (240, 86, 78)
bot = (197, 30, 33)
grad = Image.new("RGB", (SIZE, SIZE))
px = grad.load()
for yy in range(SIZE):
    t = yy / (SIZE - 1)
    col = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
    for xx in range(SIZE):
        px[xx, yy] = col

logo = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
logo.paste(grad, (0, 0), mask)

SS = 4
dots = Image.new("RGBA", (SIZE * SS, SIZE * SS), (0, 0, 0, 0))
dd = ImageDraw.Draw(dots)

def circle(cx, cy, r, fill):
    dd.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], fill=fill)

for cx, cy in [(150, 188), (256, 260), (368, 332)]:
    circle(cx, cy, 30, (255, 255, 255, 255))
for cx, cy in [(398, 150), (114, 372)]:
    circle(cx, cy, 25, (255, 176, 174, 150))

dots = dots.resize((SIZE, SIZE), Image.LANCZOS)
logo = Image.alpha_composite(logo, dots)

logo.save(OUT)
print(f"Saved {OUT}  ({logo.size[0]}x{logo.size[1]})")
