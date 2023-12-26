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
                label_set.add(filename.replace('.txt',''))

        if flag:
            for filename in os.listdir(lidar_path):
                if filename.endswith('.bin'):
                    image_set.add(filename.replace('.bin',''))


        train_set = train_set | (lidar_set & image_set & label_set)

        for i in range(len(train_set)//test_spilt):
            test_set.add(train_set.pop())
        for i in range(len(train_set)//val_spilt):
            val_set.add(train_set.pop())

    return train_set, test_set, val_set, flag

def copy_lidar2(main_path, train_set, test_set, val_set):
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
                    shutil.copy(os.path.join(source_path,file), os.path.join(target_path['train'],file[-10:]))
                elif frame in test_set:
                    shutil.copy(os.path.join(source_path,file), os.path.join(target_path['test'],file[-10:]))
                elif frame in val_set:
                    shutil.copy(os.path.join(source_path,file), os.path.join(target_path['train'],file[-10:]))

    return True

def copy_image2(main_path, train_set, test_set, val_set, flag):
    target_path = {'train': 'dataset/training/image_2/', 'test': 'dataset/testing/image_2/'}
    source_paths = []



    # select the file in train_set, test_set, val_set
        

    if flag:
        for entry in os.scandir(main_path):
            if entry.is_dir() and entry.name.startswith("vehicle"):
                source_paths.append(main_path+'/'+entry.name+'/velodyne')


        for key in target_path:
            os.makedirs(target_path[key], exist_ok=True)


        image_copy_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1221_1951/vehicle.tesla.model3.master/image_2/0000147072.png'
        # copy the image and rename it to frame name
        for source_path in source_paths:
            for file in os.listdir(source_path):
                if file.endswith('.bin'):
                    frame = file.replace('.bin','')
                    if frame in train_set:
                        shutil.copy(image_copy_path, os.path.join(target_path['train'],file[-10:-4]+'.png'))
                    elif frame in test_set:
                        shutil.copy(image_copy_path, os.path.join(target_path['test'],file[-10:-4]+'.png'))
                    elif frame in val_set:
                        shutil.copy(image_copy_path, os.path.join(target_path['train'],file[-10:-4]+'.png'))

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
                        shutil.copy(os.path.join(source_path,file), os.path.join(target_path['train'],file[-10:]))
                    elif frame in test_set:
                        shutil.copy(os.path.join(source_path,file), os.path.join(target_path['test'],file[-10:]))
                    elif frame in val_set:
                        shutil.copy(os.path.join(source_path,file), os.path.join(target_path['train'],file[-10:]))

    return True

def copy_label2(main_path, train_set, test_set, val_set):
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
    file_train = open(ImageSets_path+'train.txt','w')
    file_test = open(ImageSets_path+'test.txt','w')
    file_val = open(ImageSets_path+'val.txt','w')

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
                    with open(os.path.join(target_path['train'],file[-10:]), 'w') as f:
                        f.writelines(fix_lines)
                elif frame in test_set:
                    with open(os.path.join(target_path['test'],file[-10:]), 'w') as f:
                        f.writelines(fix_lines)
                elif frame in val_set:
                    with open(os.path.join(target_path['train'],file[-10:]), 'w') as f:
                        f.writelines(fix_lines)

    return True

def generate_imagesets(main_path, train_set, test_set, val_set):
    ImageSets_path = 'dataset/ImageSets/'
    os.makedirs(ImageSets_path, exist_ok=True)

    # create train.txt and test.txt file if not exist
    file_train = open(ImageSets_path+'train.txt','w')
    file_test = open(ImageSets_path+'test.txt','w')
    file_val = open(ImageSets_path+'val.txt','w')

    # print the file name in train_set, test_set, val_set, only the last 6 characters

    # sort the train_set, test_set, val_set
    train_set = sorted(train_set)
    test_set = sorted(test_set)
    val_set = sorted(val_set)

    for file in train_set:
        print(file[-6:],file = file_train)
    for file in test_set:
        print(file[-6:],file = file_test)
    for file in val_set:
        print(file[-6:],file = file_val)

    return True


def read_dataset_dir():
    set_label = ['50-25','75-37','100-50','125-67','150-75']
    set_A_05 = ['1221_2116', '1223_1633', '1221_1951', '1223_1618', '1222_1620']
    set_B_01 = ['1223_1749', '1223_1805', '1223_1819', '1223_1836', '1223_1915']
    set_D_06 = ['1225_1955', '1225_2009', '1225_2024', '1225_2040', 'None']
    
    for i in range(len(set_label)):
        subset = set_A_05[i]
        dir_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'+subset
        print('Set: ', set_label[i], ' = ', subset)



    return



if __name__ == "__main__":
    os.makedirs('dataset', exist_ok=True)
    main_path = read_main_path()
    train_set, test_set, val_set, flag = get_inner_frame(main_path, 12, 15)
    print('Set preparation done')
    # print("train_set: ", train_set)

    copy_lidar2(main_path, train_set, test_set, val_set)
    print('Lidar copy done')
    copy_image2(main_path, train_set, test_set, val_set, flag)
    print('Image copy done')
    copy_label2(main_path, train_set, test_set, val_set)
    print('Label copy done')
    generate_imagesets(main_path, train_set, test_set, val_set)
    print('Imagesets generate done')
    copy_calib2()