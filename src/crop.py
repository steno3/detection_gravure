from PIL import Image
import random

output_dir = './murs'

def crop_random(img_path, output_path, crop_size=(1024, 1024)):
    with Image.open(img_path) as img:
        width, height = img.size
        crop_w, crop_h = crop_size

        if width < crop_w or height < crop_h:
            raise ValueError("Crop size is larger than the image size.")

        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        right = left + crop_w
        bottom = top + crop_h

        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path)

def crop_n_random(img_path, output_dir, num_crops=10, crop_size=(1024, 1024)):
    with Image.open(img_path) as img:
        width, height = img.size
        crop_w, crop_h = crop_size

        if width < crop_w or height < crop_h:
            raise ValueError("Crop size is larger than the image size.")

        for i in range(num_crops):
            left = random.randint(0, width - crop_w)
            top = random.randint(0, height - crop_h)
            right = left + crop_w
            bottom = top + crop_h

            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(f"{output_dir}/crop_{i+1}.png")

def crop_middle(img_path, output_path, crop_size=(1024, 1024)):
    with Image.open(img_path) as img:
        width, height = img.size
        crop_w, crop_h = crop_size

        if width < crop_w or height < crop_h:
            raise ValueError("Crop size is larger than the image size.")

        left = (width - crop_w) // 2 + 350
        top = (height - crop_h) // 2 -40
        right = left + crop_w
        bottom = top + crop_h

        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path)

if __name__ == '__main__':
    # crop_random('../../002.png', 'mur1.png', crop_size=(1024, 1024))
    #crop_n_random('../../002.png', output_dir, num_crops=100, crop_size=(1024, 1024))
    crop_middle('../data/omoplate_1_unimsps.png', './omoplate.png', crop_size=(1024, 1024))
    #print(f"Cropped images saved to {output_dir}.")