'''
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
'''
import os, random, shutil

def random_cutFile(srcPath,dstPath,numfiles):
    name_list=list(os.path.join(srcPath,name) for name in os.listdir(srcPath))
    random_name_list=list(random.sample(name_list,numfiles))
    if not os.path.exists(dstPath):
        os.mkdir(dstPath)
    for oldname in random_name_list:
        shutil.move(oldname,oldname.replace(srcPath, dstPath))


nums_per_class = 250
srcPath='F:\\Event&NoEvent\\train\\Even_spec_224\\'
dstPath = "F:\\Event&NoEvent\\test\\Even_spec_224\\"
random_cutFile(srcPath,dstPath,nums_per_class)


srcPath='F:\\Event&NoEvent\\train\\No_event_spec_224\\'
dstPath = "F:\\Event&NoEvent\\test\\No_event_spec_224\\"
random_cutFile(srcPath,dstPath,nums_per_class)


