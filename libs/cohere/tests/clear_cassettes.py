import os
import shutil

def delete_cassettes_directories(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == 'cassettes':
                dir_to_delete = os.path.join(dirpath, dirname)
                print(f"Deleting directory: {dir_to_delete}")
                shutil.rmtree(dir_to_delete)

if __name__ == "__main__":
    directory_to_clear = os.getcwd() + "/integration_tests"
    if not os.path.isdir(directory_to_clear):
        raise Exception("integration_tests directory not found in current working directory")
    delete_cassettes_directories(directory_to_clear)