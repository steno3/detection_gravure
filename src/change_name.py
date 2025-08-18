import os
import sys

def rename_files_in_folder(folder_path, old_word, new_word):
    for filename in os.listdir(folder_path):
        if old_word in filename:
            new_filename = filename.replace(old_word, new_word)
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            os.rename(old_file, new_file)
            print(f'Renamed: {filename} -> {new_filename}')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python change_name.py <folder_path> <old_word> <new_word>")
        sys.exit(1)
    folder = sys.argv[1]
    old = sys.argv[2]
    new = sys.argv[3]
    rename_files_in_folder(folder, old, new)