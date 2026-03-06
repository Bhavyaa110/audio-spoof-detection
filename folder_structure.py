import os

output_file = "folder_structure.txt"
start_path = "."   # current directory

def write_tree(folder, indent=""):
    entries = os.listdir(folder)

    for i, entry in enumerate(entries):
        path = os.path.join(folder, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        line = indent + connector + entry + "\n"
        file.write(line)

        if os.path.isdir(path):
            extension = "    " if i == len(entries) - 1 else "│   "
            write_tree(path, indent + extension)


with open(output_file, "w", encoding="utf-8") as file:
    file.write("Project Folder Structure\n\n")
    write_tree(start_path)

print("Folder structure saved to folder_structure.txt")