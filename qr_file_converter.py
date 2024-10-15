import qrcode
from qreader import QReader
import cv2
import numpy as np
import base64
import os
import zlib
import argparse
from PIL import Image
import random
from tqdm import tqdm

def compress_data(data):
    return zlib.compress(data)

def decompress_data(compressed_data):
    return zlib.decompress(compressed_data)

def file_to_qr_codes(file_path, chunk_size = 1000, qr_version=None, error_correction=qrcode.constants.ERROR_CORRECT_L):
    print(f"Read {file_path}")
    with open(file_path, 'rb') as f:
        data = f.read()

    compressed_data = compress_data(data)
    encoded_data = base64.b64encode(compressed_data).decode('utf-8')
    
    chunks = [encoded_data[i:i+chunk_size] for i in range(0, len(encoded_data), chunk_size)]
    total_chunks = len(chunks)
    qr_images = []
    
    print("Converting to QR code...")
    qr = qrcode.QRCode(version=qr_version, 
                        error_correction=error_correction,
                        box_size=10, 
                        border=4)
    
    for idx, chunk in enumerate(tqdm(chunks)):
        # Clear data
        qr.clear()

        data_to_encode = f"{idx+1}/{total_chunks}:{chunk}"
        try:
            qr.add_data(data_to_encode)
            qr.make(fit=True)
        except qrcode.exceptions.DataOverflowError:
            print(f"Error: Data overflow at chunk {idx+1}. Try a larger QR version or a lower error correction level.")
            return None
        
        img = qr.make_image(fill_color="black", back_color="white")
        qr_images.append(img)
    
    return qr_images

def create_multi_channel_qr(qr_images):
    assert len(qr_images) == 3, ValueError("3 QR code images are required for multi-channel storage.")
    
    # Convert PIL images to numpy arrays
    y_channel = np.array(qr_images[0].convert('L'))
    cb_channel = np.array(qr_images[1].convert('L'))
    cr_channel = np.array(qr_images[2].convert('L'))
    
    # Create YCbCr image
    ycbcr_image = np.stack([y_channel, cb_channel, cr_channel], axis=-1)
    
    # Convert YCbCr to RGB
    rgb_image = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2RGB)
    
    return Image.fromarray(rgb_image)

def combine_qr_codes(qr_images, grid_size=(5, 3)):
    rows, cols = grid_size
    single_size = qr_images[0].size[0]
    combined_width = single_size * cols
    combined_height = single_size * rows
    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')
    
    for i, img in enumerate(qr_images):
        row = i // cols
        col = i % cols
        combined_image.paste(img, (col * single_size, row * single_size))
        
        if i + 1 == len(qr_images):
            break
    
    return combined_image

def save_combined_qr_codes(qr_images, output_folder, grid_size=(5, 3)):
    os.makedirs(output_folder, exist_ok=True)
    combined_images = []
    
    print(f"Total {len(qr_images)} black QR code")
    qr_color_images = []
    for i in range(0, len(qr_images), 3):
        batch = qr_images[i:i+3]
        if len(batch) == 3:
            multi_channel_qr = create_multi_channel_qr(batch)
            qr_color_images.append(multi_channel_qr)
        else:
            # If we have leftover QR codes, save them individually
            for img in batch:
                qr_color_images.append(img)
    
    print(f"Total {len(qr_color_images)} color QR code")
    for i in range(0, len(qr_color_images), grid_size[0] * grid_size[1]):
        batch = qr_color_images[i:i + grid_size[0] * grid_size[1]]
        combined = combine_qr_codes(batch, grid_size)
        combined_images.append(combined)
    
    for idx, img in enumerate(combined_images):
        img.save(os.path.join(output_folder, f"combined_qr_codes_{idx+1}.png"))

def read_qr_codes(input_folder):
    qreader = QReader()
    chunks = {}
    total_chunks = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            print(f"Read {filename}")

            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img.shape[2] == 3:  # Multi-channel QR
                # Convert RGB to YCbCr
                ycbcr_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

                # Extract QR codes from each channel
                for channel in cv2.split(ycbcr_image):
                    # Threshold the channel to create a binary image
                    _, binary_qr = cv2.threshold(channel, 128, 255, cv2.THRESH_BINARY)

                    decoded_text = qreader.detect_and_decode(image=binary_qr)
                    print(f"\tFind {len(decoded_text)} QR code")
                    for data in decoded_text:
                        if data:
                            chunk_info, chunk_data = data.split(':', 1)
                            current, total = map(int, chunk_info.split('/'))
                            chunks[current] = chunk_data
                            total_chunks = max(total_chunks, total)

            else:  # Single-channel QR (for leftover QR codes)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Detect and decode multiple QR codes in a single image
                decoded_text = qreader.detect_and_decode(image=gray)
                print(f"\tFind {len(decoded_text)} QR code")
                for data in decoded_text:
                    if data:
                        chunk_info, chunk_data = data.split(':', 1)
                        current, total = map(int, chunk_info.split('/'))
                        chunks[current] = chunk_data
                        total_chunks = max(total_chunks, total)

    if len(chunks) != total_chunks:
        print(f"Warning: Only {len(chunks)} out of {total_chunks} chunks were successfully decoded.")
    
    sorted_chunks = [chunks[i] for i in range(1, total_chunks + 1) if i in chunks]
    return ''.join(sorted_chunks)

def qr_codes_to_file(input_folder, output_file):
    encoded_data = read_qr_codes(input_folder)
    compressed_data = base64.b64decode(encoded_data)
    decompressed_data = decompress_data(compressed_data)
    
    print(f"Save to {output_file}")
    with open(output_file, 'wb') as f:
        f.write(decompressed_data)

def main():
    parser = argparse.ArgumentParser(description="Convert files to QR codes and back.")
    parser.add_argument("action", choices=["encode", "decode"], help="Action to perform")
    parser.add_argument("input", help="Input file or folder")
    parser.add_argument("output", help="Output folder or file")
    parser.add_argument("--qr-version", type=int, default=None, help="QR code version (1-40)")
    parser.add_argument("--error-correction", choices=["L", "M", "Q", "H"], default="L", help="Error correction level")
    parser.add_argument("--grid", nargs=2, type=int, default=[5, 3], help="Grid size for combining QR codes (e.g., 3 2 for 3x2)")
    args = parser.parse_args()

    error_correction_levels = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H
    }

    if args.action == "encode":
        qr_images = file_to_qr_codes(args.input, 
                                     qr_version=args.qr_version, 
                                     error_correction=error_correction_levels[args.error_correction])
        if qr_images:
            save_combined_qr_codes(qr_images, args.output, tuple(args.grid))
            print(f"Combined QR codes saved in: {args.output}")
        else:
            print("Failed to generate QR codes. Please adjust QR version or error correction level.")
    elif args.action == "decode":
        qr_codes_to_file(args.input, args.output)
        print(f"File reconstructed: {args.output}")

if __name__ == "__main__":
    main()