import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
from PIL import Image
from tqdm import tqdm

import ptwt


def generate_frequency_packet_image(packet_array: np.ndarray, degree: int):
    """Create a ready-to-polt image with frequency-order packages.
       Given a packet array in natural order, creat an image which is
       ready to plot in frequency order.
    Args:
        packet_array (np.ndarray): [packet_no, packet_height, packet_width]
            in natural order.
        degree (int): The degree of the packet decomposition.
    Returns:
        [np.ndarray]: The image of shape [original_height, original_width]
    """
    wp_freq_path = ptwt.WaveletPacket2D.get_freq_order(degree)
    wp_natural_path = ptwt.WaveletPacket2D.get_natural_order(degree)

    image = []
    # go through the rows.
    for row_paths in wp_freq_path:
        row = []
        for row_path in row_paths:
            index = wp_natural_path.index(row_path)
            packet = packet_array[index]
            row.append(packet)
        image.append(np.concatenate(row, -1))
    return np.concatenate(image, 0)


def load_image(path_to_file: str) -> torch.Tensor:
    image = Image.open(path_to_file)
    tensor = torch.from_numpy(np.nan_to_num(np.array(image), posinf=255, neginf=0))
    return tensor


def process_images(tensor: torch.Tensor, paths: list) -> torch.Tensor:
    tensor = torch.mean(tensor / 255.0, -1)
    packets = ptwt.WaveletPacket2D(tensor, pywt.Wavelet("Haar"))

    packet_list = []
    for node in paths:
        packet = torch.squeeze(packets["".join(node)], dim=1)
        packet_list.append(packet)
    wp_pt = torch.stack(packet_list, dim=1)
    # return wp_pt
    return torch.log(torch.abs(wp_pt) + 1e-12)


def load_images(path: str) -> list:
    image_list = []
    for root, _, files in os.walk(path, topdown=False):
        for name in tqdm(files):
            path = os.path.join(root, name)
            packets = load_image(path)
            image_list.append(packets)
    return image_list


if __name__ == "__main__":
    freq_path = ptwt.WaveletPacket2D.get_freq_order(level=3)
    natural_path = ptwt.WaveletPacket2D.get_natural_order(level=3)
    print("Loading ffhq images:")
    ffhq_images = load_images("./ffhq_style_gan/source_data/A_ffhq")
    print("processing ffhq")
    ffhq_images = torch.stack(ffhq_images).split(2500)
    ffhq_packets = []
    for image_batch in tqdm(ffhq_images):
        ffhq_packets.append(process_images(image_batch, natural_path))

    mean_packets_ffhq = torch.mean(torch.cat(ffhq_packets), 0)
    del ffhq_images
    del ffhq_packets

    print("Loading style-gan images")
    gan_images = load_images("./ffhq_style_gan/source_data/B_stylegan")
    print("processing style-gan")
    gan_images = torch.stack(gan_images).split(2500)
    gan_packets = []
    for image_batch in tqdm(gan_images):
        gan_packets.append(process_images(image_batch, natural_path))

    mean_packets_gan = torch.mean(torch.cat(gan_packets), 0)
    del gan_images
    del gan_packets

    plot_ffhq = generate_frequency_packet_image(mean_packets_ffhq, 3)
    plot_gan = generate_frequency_packet_image(mean_packets_gan, 3)

    fig = plt.figure(figsize=(9, 3))
    fig.add_subplot(1, 2, 1)
    plt.imshow(plot_ffhq, vmax=1.5, vmin=-7)
    plt.title("real")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    fig.add_subplot(1, 2, 2)
    plt.imshow(plot_gan, vmax=1.5, vmin=-7)
    plt.title("fake")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    plt.show()

    plt.plot(torch.mean(mean_packets_ffhq, (1, 2)).flatten().numpy(), label="real")
    plt.plot(torch.mean(mean_packets_gan, (1, 2)).flatten().numpy(), label="fake")
    plt.xlabel("mean packets")
    plt.ylabel("magnitude")
    plt.legend()
    plt.show()
