import glob
import os

files = glob.glob("Nd_*.txt")

def change_filename(files):
    for file in files:
        if file[3:7].isdigit():
            pass
        elif file[3:6].isdigit():
            new_name = file[:3]+"0"+file[3:]
            os.rename(file, new_name)
        elif file[3:5].isdigit():
            new_name = file[:3]+"00"+file[3:]
            os.rename(file, new_name)
        elif file[3:4].isdigit():
            new_name = file[:3]+"000"+file[3:]
            os.rename(file, new_name)

    print("done!")

print(files)
for file in files:
    print(file, file[10:-4])
    # vapp = float(file[10:-4])
    # print(vapp)

