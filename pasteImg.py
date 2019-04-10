'''
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
'''
import os, random, shutil


def copyFile(fileDir):
    # 1
    pathDir = os.listdir(fileDir)

    # 2
    sample = random.sample(pathDir, 10000)
    print(sample)

    # 3
    for name in sample:
        shutil.copyfile(fileDir + name, tarDir + name)

def random_copyfile(srcPath,dstPath,numfiles):
    name_list=list(os.path.join(srcPath,name) for name in os.listdir(srcPath))
    random_name_list=list(random.sample(name_list,numfiles))
    if not os.path.exists(dstPath):
        os.mkdir(dstPath)
    for oldname in random_name_list:
        shutil.copyfile(oldname,oldname.replace(srcPath, dstPath))

srcPath='F:\\Event&NoEvent\\train\\Even_spec_224\\'
dstPath = "F:\\Event&NoEvent\\test\\Even_spec_224\\"
random_copyfile(srcPath,dstPath,100)

if __name__ == '__main__':
    fileDir = "F:\\Event&NoEvent\\train\\Even_spec_224\\"
    tarDir = "F:\\Event&NoEvent\\test\\Even_spec_224\\"
