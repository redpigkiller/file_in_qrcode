import argparse
from pathlib import Path
import numpy as np
import cv2
import base64
from pyzbar import pyzbar

from utilus import unshuffle_color, load_file_pattern


def main(qr_img_path_pattern :str, recovered_file_name :str, seed :int=-1) -> None:

    img_path_list = load_file_pattern(qr_img_path_pattern)

    if len(img_path_list) == 0:
        print("[warning] no image!")
        return

    # Read and decode data from QR codes
    pack_image_list = []

    for img_path in img_path_list:
        # Read the images
        if Path(img_path).is_file():
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"Load \"{img_path}\"")
        else:
            break

        pack_image_list.append(img)
    

    # Unshuffle the color channel
    for i, img in enumerate(pack_image_list):
        pack_image_list[i] = unshuffle_color(img, seed)

    colr_dim = 3
    bits_dim = 8

    # Read and decode data from QR codes
    chunks = ''

    # Unpack the image
    for pack_img in pack_image_list:
        for i_colr in range(colr_dim):
            for i_bits in range(bits_dim):
                qr_img = pack_img[:, :, i_colr]
                qr_img = np.uint8(((qr_img // (2 ** i_bits)) % 2) * 255)

                # Decode the image
                decoded_objects = pyzbar.decode(qr_img, symbols=[pyzbar.ZBarSymbol.QRCODE])

                # Append the decoded bytes
                if len(decoded_objects) == 0:
                    flag_end_of_file = True
                    break

                decoded_objects = sorted(decoded_objects, key=lambda obj: (obj.rect.top, obj.rect.left))

                for obj in decoded_objects:
                    chunks += obj.data.decode('utf-8')

        if flag_end_of_file:
            break

    decoded_data = base64.b64decode(chunks)

    # Write decoded data to file
    file_path = Path(recovered_file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as f:
        f.write(decoded_data)
        print(f"Save to {file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert QR code to a file")
    parser.add_argument("-i", "--input", help="Pattern for QR images (e.g., 'test_*.png')", default="output/utilus.py_*.png")
    parser.add_argument("-f", "--file_name", help="Recovered file name", default="utilus_recovered.py")
    parser.add_argument("-s", "--seed", help="random seed", default=123)

    args = parser.parse_args()

    main(args.input, args.file_name, args.seed)
