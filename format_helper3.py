import os
import shutil
import argparse
import tqdm


def read_main_path():
    parser = argparse.ArgumentParser(description='Format Helper2')
    parser.add_argument('--dataset_path', '-d', type=str, help='dataset main path')
    args = parser.parse_args()
    return args.dataset_path

def copy_calib2():
    target_path = {'train': 'dataset/training/calib/', 'test': 'dataset/testing/calib/', 'val': 'dataset/training/calib/'}
    template_file = 'test_data/000001.txt'
    imagesets_path = {'train': 'dataset/ImageSets/train.txt', 'test': 'dataset/ImageSets/test.txt', 'val': 'dataset/ImageSets/val.txt'}

    # read imagesets file, read each line and rename the template file, then copy to target path
    for key in target_path:
        os.makedirs(target_path[key], exist_ok=True)
        with open(imagesets_path[key],'r') as f:
            lines = f.readlines()
            for line in lines:
                shutil.copy(template_file, os.path.join(target_path[key],line.replace('\n','')+'.txt'))

    return

def get_inner_frame(main_path, test_spilt, val_spilt):
    source_paths = []
    for entry in os.scandir(main_path):
        if entry.is_dir() and entry.name.startswith("vehicle"):
            source_paths.append(main_path+'/'+entry.name)

    train_set = set()
    test_set = set()
    val_set = set()
    for source_path in source_paths:
        lidar_path = source_path+'/velodyne'
        image_path = source_path+'/image_2'
        label_path = source_path+'/velodyne_semantic'

        lidar_set = set()
        image_set = set()
        label_set = set()

        flag = False

        for filename in os.listdir(lidar_path):
            if filename.endswith('.bin'):
                lidar_set.add(filename.replace('.bin',''))
        try:
            for filename in os.listdir(image_path):
                if filename.endswith('.png'):
                    image_set.add(filename.replace('.png',''))
        except:
            print('[WARNING] No image folder, the setting vehicle number is too large, copy the image from other folder')
            flag = True
        for filename in os.listdir(label_path):
            if filename.endswith('.txt'):
                with open(os.path.join(label_path,filename), 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 1:
                        print('[WARNING] No label in this frame, skip', filename)
                        continue
                    else:
                        label_set.add(filename.replace('.txt',''))

        if flag:
            for filename in os.listdir(lidar_path):
                if filename.endswith('.bin'):
                    image_set.add(filename.replace('.bin',''))


        train_set = train_set | (lidar_set & image_set & label_set)

        # random select test_spilt frames from train_set
        for i in range(len(train_set)):
            if i % test_spilt == 0:
                test_set.append(train_set[i])
                train_set[i] = 'none'
            if i % val_spilt == 0:
                val_set.append(train_set[i])
                train_set[i] = 'none'

        train_set = [x for x in train_set if x != 'none']

        # for i in range(len(train_set)//test_spilt):
        #     test_set.add(train_set.pop())
        # for i in range(len(train_set)//val_spilt):
        #     val_set.add(train_set.pop())

    return train_set, test_set, val_set, flag

def copy_lidar2(main_path, train_set, test_set, val_set, idx):
    target_path = {'train': 'dataset/training/velodyne/', 'test': 'dataset/testing/velodyne/'}
    source_paths = []
    for entry in os.scandir(main_path):
        if entry.is_dir() and entry.name.startswith("vehicle"):
            source_paths.append(main_path+'/'+entry.name+'/velodyne')
    
    for key in target_path:
        os.makedirs(target_path[key], exist_ok=True)

    # select the file in train_set, test_set, val_set
    for source_path in source_paths:
        for file in os.listdir(source_path):
            if file.endswith('.bin'):
                frame = file.replace('.bin','')
                if frame in train_set:
                    shutil.copy(os.path.join(source_path,file), os.path.join(target_path['train'],str(idx)+file[-9:]))
                elif frame in test_set:
                    shutil.copy(os.path.join(source_path,file), os.path.join(target_path['test'],str(idx)+file[-9:]))
                elif frame in val_set:
                    shutil.copy(os.path.join(source_path,file), os.path.join(target_path['train'],str(idx)+file[-9:]))

    return True

def copy_image2(main_path, train_set, test_set, val_set, flag, idx):
    target_path = {'train': 'dataset/training/image_2/', 'test': 'dataset/testing/image_2/'}
    source_paths = []



    # select the file in train_set, test_set, val_set
        

    if flag:
        for entry in os.scandir(main_path):
            if entry.is_dir() and entry.name.startswith("vehicle"):
                source_paths.append(main_path+'/'+entry.name+'/velodyne')


        for key in target_path:
            os.makedirs(target_path[key], exist_ok=True)


        image_copy_path = '/home/newDisk/tool/carla_dataset_tool/raw_data/record_2024_0104_1949/vehicle.tesla.model3.master/image_2/0000012836.png'
        # copy the image and rename it to frame name
        for source_path in source_paths:
            for file in os.listdir(source_path):
                if file.endswith('.bin'):
                    frame = file.replace('.bin','')
                    if frame in train_set:
                        shutil.copy(image_copy_path, os.path.join(target_path['train'],str(idx)+file[-9:-4]+'.png'))
                    elif frame in test_set:
                        shutil.copy(image_copy_path, os.path.join(target_path['test'],str(idx)+file[-9:-4]+'.png'))
                    elif frame in val_set:
                        shutil.copy(image_copy_path, os.path.join(target_path['train'],str(idx)+file[-9:-4]+'.png'))

    else: 
        for entry in os.scandir(main_path):
            if entry.is_dir() and entry.name.startswith("vehicle"):
                source_paths.append(main_path+'/'+entry.name+'/image_2')

        for key in target_path:
            os.makedirs(target_path[key], exist_ok=True)
    

        for source_path in source_paths:
            for file in os.listdir(source_path):
                if file.endswith('.png'):
                    frame = file.replace('.png','')
                    if frame in train_set:
                        shutil.copy(os.path.join(source_path,file), os.path.join(target_path['train'],str(idx)+file[-9:]))
                    elif frame in test_set:
                        shutil.copy(os.path.join(source_path,file), os.path.join(target_path['test'],str(idx)+file[-9:]))
                    elif frame in val_set:
                        shutil.copy(os.path.join(source_path,file), os.path.join(target_path['train'],str(idx)+file[-9:]))

    return True

def copy_label2(main_path, train_set, test_set, val_set, idx):
    target_path = {'train': 'dataset/training/label_2/', 'test': 'dataset/testing/label_2/'}
    ImageSets_path = 'dataset/ImageSets/'
    source_paths = []
    for entry in os.scandir(main_path):
        if entry.is_dir() and entry.name.startswith("vehicle"):
            source_paths.append(main_path+'/'+entry.name+'/velodyne_semantic')

    for key in target_path:
        os.makedirs(target_path[key], exist_ok=True)
    os.makedirs(ImageSets_path, exist_ok=True)

    # create train.txt and test.txt file if not exist
    # file_train = open(ImageSets_path+str(idx)+'train.txt','w')
    # file_test = open(ImageSets_path+str(idx)+'test.txt','w')
    # file_val = open(ImageSets_path+str(idx)+'val.txt','w')

    # select the file in train_set, test_set, val_set
    for source_path in source_paths:
        for file in os.listdir(source_path):
            if file.endswith('.txt'):
                with open(os.path.join(source_path,file), 'r') as f:
                    lines = f.readlines()
                    lines = lines[:-1]
                    fix_lines = ""
                    for line in lines:
                        elements = line.split(" ")
                        # print('This is the test version, when fix the recorder, update here')
                        # x, y, z, l, w, h, rot, lab, _, _, _ = elements
                        x, y, z, l, w, h, rot, lab, _, _ = elements
                        l = abs(round(float(l),4))
                        w = abs(round(float(w),4))
                        h = abs(round(float(h),4))
                        x = round(float(x),4)
                        y = round(float(y),4)
                        z = round(float(z),4)
                        rot = round(float(rot),4)
                        # if lab == 'Bus': lab = 'Van'
                        # x, y, z, dx, dy, dz, rot, lab, _, _, _ = elements
                        rot -= 3.14 if rot > 3.14 else 0
                        rot = round(float(rot),4)
                        fix_line = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(lab, '0', '0', '0', '0', '0', '0', '0', h, w, l, x, y, z, rot)
                        # fix_line = "{} {} {} {} {} {} {} {}\n".format(x,y,z,dx,dy,dz,rot,lab)
                        fix_lines += fix_line
                frame = file.replace('.txt','')
                if frame in train_set:
                    with open(os.path.join(target_path['train'],str(idx)+file[-9:]), 'w') as f:
                        f.writelines(fix_lines)
                elif frame in test_set:
                    with open(os.path.join(target_path['test'],str(idx)+file[-9:]), 'w') as f:
                        f.writelines(fix_lines)
                elif frame in val_set:
                    with open(os.path.join(target_path['train'],str(idx)+file[-9:]), 'w') as f:
                        f.writelines(fix_lines)

    return True


def generate_imagesets(main_path, train_set, test_set, val_set, idx):
    ImageSets_path = 'dataset/ImageSets/'
    os.makedirs(ImageSets_path, exist_ok=True)

    # create train.txt and test.txt file if not exist
    file_train = open(ImageSets_path+'train.txt','a')
    file_test = open(ImageSets_path+'test.txt','a')
    file_val = open(ImageSets_path+'val.txt','a')

    # print the file name in train_set, test_set, val_set, only the last 6 characters

    # sort the train_set, test_set, val_set
    train_set = sorted(train_set)
    test_set = sorted(test_set)
    val_set = sorted(val_set)

    for file in train_set:
        file_train.write(str(idx)+file[-5:]+'\n')
    for file in test_set:
        file_test.write(str(idx)+file[-5:]+'\n')
    for file in val_set:
        file_val.write(str(idx)+file[-5:]+'\n')

    file_train.close()
    file_test.close()
    file_val.close()

    return True



def read_dataset_dir():
    set_label = ['50-25','75-37','100-50','125-67','150-75']
    set_A_05 = ['1226_2120', '1226_2132', '1226_2146', '1226_2200', '1226_2214']
    set_B_02 = ['0104_2223', '0104_2309', '0104_2328', '0104_2343', '0105_0013']
    set_D_06 = ['0104_1949', '0104_2002', '0104_2016', '0104_2032', '0104_2056']
    
    for i in range(len(set_label)):
        subset = set_D_06[i]
        dir = '/home/newDisk/tool/carla_dataset_tool/raw_data/record_2024_'+subset
        for entry in os.scandir(dir):
            if entry.is_dir() and entry.name.startswith("vehicle"):
                print(i,"=",dir)
                train_set, test_set, val_set, flag = get_inner_frame(dir, 12, 15)
                copy_lidar2(dir, train_set, test_set, val_set, i)
                print('Lidar copy done')
                copy_image2(dir, train_set, test_set, val_set, flag, i)
                print('Image copy done')
                copy_label2(dir, train_set, test_set, val_set, i)
                print('Label copy done')
                generate_imagesets(dir, train_set, test_set, val_set, i)
                print('Imagesets generate done')
                copy_calib2()

    return



if __name__ == "__main__":
    os.makedirs('dataset', exist_ok=True)
    read_dataset_dir()
