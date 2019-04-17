import os, random, shutil

def random_cutFile(srcPath,dstPath,numfiles):
    name_list=list(os.path.join(srcPath,name) for name in os.listdir(srcPath))
    random_name_list=list(random.sample(name_list,numfiles))
    if not os.path.exists(dstPath):
        os.mkdir(dstPath)
    for oldname in random_name_list:
        shutil.move(oldname,oldname.replace(srcPath, dstPath))


nums_per_class = 250
srcPath='D:\\Event&NoEvent\\train\\Even_spec_224\\'
dstPath = "D:\\Event&NoEvent\\validation\\Even_spec_224\\"
random_cutFile(srcPath,dstPath,nums_per_class)


srcPath='D:\\Event&NoEvent\\train\\No_event_spec_224\\'
dstPath = "D:\\Event&NoEvent\\validation\\No_event_spec_224\\"
random_cutFile(srcPath,dstPath,nums_per_class)


