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

def split_lidar(source_dir, set_dir ,bin_dir, txt_dir):
    os.makedirs(set_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    bin_listdir = source_dir + '/velodyne'
    txt_listdir = source_dir + '/velodyne_semantic'
    file_cnt = 0
    # for filename in os.listdir(pyn_listdir):
    #     if filename.endswith('.npy'):
    #         file_cnt += 1
    #         shutil.copy(os.path.join(pyn_listdir,filename), os.path.join(pyn_dir, pyn_listdir.split('/')[2]+"_"+filename))
    for filename in os.listdir(txt_listdir):
        if filename.endswith('.txt'):
            file_cnt += 1
            # read the txt file
            with open(os.path.join(txt_listdir,filename), 'r') as f:
                lines = f.readlines()
                lines = lines[:-1]
                fix_lines = ""

                for line in lines:
                    elements = line.split(" ")
                    # x, y, z, l, w, h, rot, lab, _, _, _ = elements
                    x, y, z, dx, dy, dz, rot, lab, _, _, _ = elements
                    rot = float(rot)
                    rot -= 3.14 if rot > 3.14 else 0
                # if lab != 'Bus':
                    # fix_line = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(lab, '0', '0', '0', '0', '0', '0', '0', h, w, l, x, y, z, rot)
                    # fix_line = "{} {} {} {} {} {} {} {}\n".format(x,y,z,l,w,h,rot,lab)
                    fix_line = "{} {} {} {} {} {} {} {}\n".format(x,y,z,dx,dy,dz,rot,lab)
                    fix_lines += fix_line

            with open(os.path.join(txt_dir, txt_listdir.split('/')[2]+"_"+filename), 'w') as f:
                f.writelines(fix_lines) 
            shutil.copy(os.path.join(bin_listdir,filename.replace('.txt','.bin')), os.path.join(bin_dir, bin_listdir.split('/')[2]+"_"+filename.replace('.txt','.bin')))

        if file_cnt % 12 == 0:
            print("Now processing: ", file_cnt, " files")

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
        # split_files(source_dir+"/image_2", 'dataset/image/images', 'dataset/image/labels')
        lidar_file_num += split_lidar(source_dir,'dataset/velodyne/ImageSets' ,'dataset/velodyne/points', 'dataset/velodyne/labels')
    
    # make file if not exist:dataset/velodyne/ImageSets/train.txt, dataset/velodyne/ImageSets/val.txt
    os.makedirs('dataset/velodyne/ImageSets', exist_ok=True)
    file_train = open('dataset/velodyne/ImageSets/train.txt','w')
    file_val = open('dataset/velodyne/ImageSets/val.txt','w')
    idx = 0



    # file_train = open('dataset/velodyne/ImageSets/train.txt','w')
    # file_val = open('dataset/velodyne/ImageSets/val.txt','w')
    # idx = 0

    file_list = os.listdir('dataset/velodyne/labels')
    for points in file_list:
        if idx % 12 == 0:
            print(points.replace('.txt',''), file=file_val)
        else: print(points.replace('.txt',''), file=file_train)
        idx += 1