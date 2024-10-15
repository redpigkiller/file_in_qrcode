# file_in_qrcode

This is just a simple script to convert any file to many QR code and recover the file from these QR code.

It uses [python-qrcode](https://github.com/lincolnloop/python-qrcode) package to generate QR code and uses [QReader](https://github.com/Eric-Canas/QReader) to decode the QR code.

## Usage
### Encode
```bash
python qr_file_converter.py encode test.pdf output_folder --qr-version 40 --error-correction H 
```
#### Options
+ `--qr-version`: QR code version: 1 to 40
+ `--error-correction`: Error correction level: "L", "M", "Q", "H"
+ `--grid`: Grid size for combining QR codes (e.g., 3 2 for 3x2)

### Decode
```bash
python qr_file_converter.py decode output_folder recovered_test.pdf
```
