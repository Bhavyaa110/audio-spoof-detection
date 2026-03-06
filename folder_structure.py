import os

output_file = "folder_structure.txt"
start_path = "."

ignore = {".git", "__pycache__", ".venv", ".idea"}

def write_tree(folder, indent=""):
    entries = sorted(os.listdir(folder))

    for i, entry in enumerate(entries):

        if entry in ignore:
            continue

        path = os.path.join(folder, entry)

        connector = "└── " if i == len(entries) - 1 else "├── "

        file.write(indent + connector + entry + "\n")

        if os.path.isdir(path):
            extension = "    " if i == len(entries) - 1 else "│   "
            write_tree(path, indent + extension)


with open(output_file, "w", encoding="utf-8") as file:
    file.write("Project Folder Structure\n\n")
    write_tree(start_path)

print("Folder structure written to folder_structure.txt")