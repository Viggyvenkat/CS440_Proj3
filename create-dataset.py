from PIL import Image, ImageDraw
import random

def generate_image(assignment:int, iteration: int):
    image = Image.new("RGB", (20, 20), "white")
    draw = ImageDraw.Draw(image)

    colors = ["red", "blue", "yellow", "green"]
    third_color = None

    start_with_rows = random.choice([True, False])
    used_colors = set()

    row, col, is_dangerous = None, None, False

    for x in range(2):
        for y in range(2):

            if start_with_rows:
                row = random.randint(0, 19)
                color = random.choice(colors)
                draw.line([(0, row), (19, row)], fill=color, width=1)
            else:
                col = random.randint(0, 19)
                color = random.choice([c for c in colors if c != image.getpixel((col, 0))])
                draw.line([(col, 0), (col, 19)], fill=color, width=1)

            colors.remove(color)
            used_colors.add(color)

            if len(used_colors) == 3:
                third_color = color

            if color == "red" and "yellow" in colors:
                is_dangerous = True

            start_with_rows = not start_with_rows

    if assignment == 1:
        if is_dangerous:
            image.save(f"dataset_{assignment}/is_dangerous/generated_image_{iteration}.png")
        else:
            image.save(f"dataset_{assignment}/safe/generated_image_{iteration}.png")
    elif assignment == 2:
        image.save(f"dataset_{assignment}/{third_color}/generated_image_{iteration}.png")

if __name__ == "__main__":
    for i in range(10000):
        generate_image(1, i)