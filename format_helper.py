import os
import shutil
import argparse

def split_files(source_dir, png_dir, txt_dir):
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            shutil.copy(os.path.join(source_dir, filename), os.path.join(png_dir, source_dir.split('/')[2]+"_"+filename))
        elif filename.endswith('.txt'):
            shutil.copy(os.path.join(source_dir, filename), os.path.join(txt_dir, source_dir.split('/')[2]+"_"+filename))

def split_lidar(source_dir, set_dir ,pyn_dir, txt_dir):
    os.makedirs(set_dir, exist_ok=True)
    os.makedirs(pyn_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    pyn_listdir = source_dir + '/velodyne'
    txt_listdir = source_dir + '/velodyne_semantic'
    file_cnt = 0
    # for filename in os.listdir(pyn_listdir):
    #     if filename.endswith('.npy'):
    #         file_cnt += 1
    #         shutil.copy(os.path.join(pyn_listdir,filename), os.path.join(pyn_dir, pyn_listdir.split('/')[2]+"_"+filename))
    for filename in os.listdir(txt_listdir):
        if filename.endswith('.txt'):
            file_cnt += 1
            shutil.copy(os.path.join(txt_listdir,filename), os.path.join(txt_dir, txt_listdir.split('/')[2]+"_"+filename))
            shutil.copy(os.path.join(pyn_listdir,filename.replace('.txt','.bin')), os.path.join(pyn_dir, pyn_listdir.split('/')[2]+"_"+filename.replace('.txt','.bin')))
            
    
    return file_cnt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Format Helper')
    parser.add_argument('--source', '-s', type=str, help='source directory')
    args = parser.parse_args()
    source = args.source
    data = []
    for entry in os.scandir(source):
        if entry.is_dir() and entry.name.startswith("vehicle"):
            data.append(source+"/"+entry.name)

    lidar_file_num = 0
    for source_dir in data:
        split_files(source_dir+"/image_2", 'dataset/image/images', 'dataset/image/labels')
        lidar_file_num += split_lidar(source_dir,'dataset/velodyne/ImageSets' ,'dataset/velodyne/points', 'dataset/velodyne/labels')
    
    file_train = open('dataset/velodyne/ImageSets/train.txt','w')
    file_val = open('dataset/velodyne/ImageSets/val.txt','w')
    idx = 0

    file_list = os.listdir('dataset/velodyne/labels')
    for points in file_list:
        if idx % 12 == 0:
            print(points.replace('.txt',''), file=file_val)
        else: print(points.replace('.txt',''), file=file_train)
        idx += 1
