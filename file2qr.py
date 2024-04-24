import argparse
from pathlib import Path
import numpy as np
import base64
from PIL import Image

import io
import qrcode

from utilus import shuffle_color

chunk_size = 2953

# QR code generator
qr = qrcode.QRCode(
    version=40,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

def gen_qr_image(text :str) -> np.ndarray:

    # Clear data
    qr.clear()

    # Add text
    qr.add_data(text)

    # Create image from QR code
    qr_image = qr.make_image(fill_color="black", back_color="white")
    
    # Convert PIL image to numpy array
    qr_image = np.asarray(qr_image)

    # Convert to 0/1
    qr_img = np.where(qr_image > 0.5, 1, 0)

    return qr_img

def main(file_path :str, output_path :str, n_row :int=2, n_col :int=3, seed :int=-1) -> None:
    # Read file as binary data
    print(f"Open \"{file_path}\"")
    with open(file_path, 'rb') as f:
        bytes_data = f.read()
    
    # Encode file data as Base64
    encoded_data = base64.b64encode(bytes_data)

    # Split data into chunks
    chunks = [encoded_data[i : i+chunk_size] for i in range(0, len(encoded_data), chunk_size)]
    print(f"There are {len(chunks)} raw QR images")
 
    colr_dim = 3
    bits_dim = 8
    group_size = n_row * n_col * colr_dim * bits_dim

    group_qr_imgs = []

    qr_img_list = []
    for i_chunk, chunk in enumerate(chunks):
        qr_img_list.append(gen_qr_image(chunk.decode('utf-8')))

        # Append the QR code image
        if (i_chunk % group_size) == group_size - 1:
            # Append the gray QR code image
            group_qr_imgs.append(qr_img_list)

            # Reset
            qr_img_list = []

        if i_chunk % 100 == 0:
            print("")
        print("*", end="", flush=True)
    
    if (i_chunk % group_size) != group_size - 1:
        group_qr_imgs.append(qr_img_list)

    print("")

    bit_shift_arr = 2 ** np.arange(0, 8, 1)

    output_imgs = []
    for qr_img_list in group_qr_imgs:
        max_h = max([img.shape[0] for img in qr_img_list])
        max_w = max([img.shape[1] for img in qr_img_list])

        pack_img = np.zeros((max_h * n_row, max_w * n_col, colr_dim, bits_dim), dtype=np.uint8)

        img_idx = 0
        for i_colr in range(colr_dim):
            for i_bits in range(bits_dim):
                for i_row in range(n_row):
                    for i_col in range(n_col):
                        if img_idx < len(qr_img_list):
                            img = qr_img_list[img_idx]
                            img = np.pad(img, ( (max_h - img.shape[0], 0), (max_w - img.shape[1], 0) ))
                        else:
                            img = np.random.randint(2, size=(max_h, max_w))

                        img_idx += 1

                        pack_img[i_row*max_h : (i_row+1)*max_h, i_col*max_w : (i_col+1)*max_w, i_colr, i_bits] = img

        output_imgs.append(np.sum(pack_img * bit_shift_arr[None, None, :], axis=3))

    # Shuffle the color channel
    for i, img in enumerate(output_imgs):
        output_imgs[i] = shuffle_color(img, seed)

    # Save these images
    n_images = len(output_imgs)
    print(f"There are {n_images} combined images")

    img_path = Path(output_path) / Path(file_path).name
    img_path.parent.mkdir(parents=True, exist_ok=True)

    for img_idx, pack_img in enumerate(output_imgs):
        im = Image.fromarray(np.uint8(pack_img))
        im.save(f"{img_path}_{img_idx}.png")
        print(f"Save to {img_path}_{img_idx}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a file to QR code")
    parser.add_argument("-i", "--input", help="path of file", default="./utilus.py")
    parser.add_argument("-r", "--nrow", help="number of row", default=2)
    parser.add_argument("-c", "--ncol", help="number of column", default=3)
    parser.add_argument("-s", "--seed", help="random seed", default=123)
    parser.add_argument("-o", "--output", help="output folder", default="output")

    args = parser.parse_args()

    main(args.input, args.output, n_row=args.nrow, n_col=args.ncol, seed=args.seed)
