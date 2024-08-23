import argparse
import os
import json

def read_txt_files(input_dir):
    data = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                dic = {
                    "instruction": "Generate a speech in Jensen Huang's style.",
                    "output": file.read()
                }
                data.append(dic)
    return data

def main():
    parser = argparse.ArgumentParser(description="Read all .txt files from a directory and save their contents into a JSON file.")
    parser.add_argument('input_dir', type=str, help="The input directory containing .txt files.")
    parser.add_argument('output_file', type=str, help="The output JSON file path.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"The directory {args.input_dir} does not exist.")
        return
    
    data = read_txt_files(args.input_dir)
    
    with open(args.output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"Data has been written to {args.output_file}")

if __name__ == "__main__":
    main()