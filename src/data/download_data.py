import os
from download_fsdkaggle import download_fsdkaggle

def download_all(directory):
    return download_fsdkaggle(directory)

if __name__ == '__main__':
    # Get the path to the current file
    current_file_path = os.path.realpath(__file__)
    
    print(current_file_path)

    # Get the path to the project root directory
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    
    print(project_root_dir)

    # Append the 'data/external' path to the project root directory
    directory = os.path.join(project_root_dir, 'data')
    
    print(directory)

    download_all(directory)