import os
import regex

full_path = os.getcwd()
# config_path = os.path.join(full_path, 'data/kalo_v1-sebastian-2020-10-16/config.yaml')
config_path_kalo = os.path.join(full_path, 'kalo_v2-sebastian-2020-11-11/config.yaml')
config_path_work_surface = os.path.join(full_path, 'work_surface-sebastian-2020-11-19/config.yaml')

print("config_path", config_path_kalo)


def get_images(input_dir):
    images = [img for img in os.listdir(input_dir) if img.endswith(".png")]
    images.sort(key=lambda f: int(regex.sub('\D', '', f)))
    images = [os.path.join(input_dir, image) for image in images]
    return images
