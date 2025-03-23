from PIL import Image
import numpy as np
import os, argparse, subprocess

INTERMEDIATE = "tests/intermediate_img.txt"

def img_to_txt(input_path: str, output_path: str): 
    pixels = np.asarray(Image.open(input_path))
    with open(output_path, "w") as output_file: 
        for x in range(pixels.shape[0]):
            for y in range(pixels.shape[1]):
                hex_str = '%02x%02x%02x%02x' % tuple(pixels[x,y].tolist())
                output_file.write(hex_str)
                if y != pixels.shape[1]-1:
                    output_file.write(" ")
            if x != pixels.shape[0]-1:
                output_file.write("\n")

#img_to_txt("images/sheet_music.bmp", "tests/image.txt")

def txt_to_img(input_path: str, output_path: str):
    with open(input_path, "r") as input_file:
        hex_data = [j.split(" ") for j in [i for i in input_file.read().split("\n")]]
    img_arr = []
    for i in range(len(hex_data)): 
        row = []
        for j in range(len(hex_data[i])): 
            h = hex_data[i][j]
            row.append([int(h[i:i+2], 16) for i in (0, 2, 4)])
        img_arr.append(row)
    Image.fromarray(np.uint8(img_arr)).save(output_path)

def compress(input_path: str, output_path: str):
    img_to_txt(input_path, INTERMEDIATE)
    subprocess.run(["python3", "lm_to_compressor.py", "-c", INTERMEDIATE, output_path])

def decompress(input_path: str, output_path: str):
    subprocess.run(["python3", "lm_to_compressor.py", "-d", input_path, INTERMEDIATE])
    txt_to_img(INTERMEDIATE, output_path)

def main():
    parser = argparse.ArgumentParser(description="Compress or decompress images.")
    parser.add_argument("-c", "--compress", action="store_true", help="Compress an image.")
    parser.add_argument("-d", "--decompress", action="store_true", help="Decompress a file.")
    parser.add_argument("input", help="The input file to compress or decompress.")
    parser.add_argument("output", help="The output file to save compressed or decompressed data.")
    args = parser.parse_args()

    if args.compress:
        compress(args.input, args.output)
    elif args.decompress:
        decompress(args.input, args.output)

if __name__ == "__main__":
	main()