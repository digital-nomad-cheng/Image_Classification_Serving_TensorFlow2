import os
import shutil

def main(data_path):
    classes = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    print("classes:{}".format(classes))
    for _ in classes:
        path = os.path.join(data_path, _)
        train_path = os.path.join(data_path, 'train/' + _)
        if os.path.exists(train_path):
            shutil.rmtree(train_path)
        os.makedirs(train_path)
        val_path = os.path.join(data_path, 'val/' + _)
        if os.path.exists(val_path):
            shutil.rmtree(val_path)
        os.makedirs(val_path)
        
        imgs = [f for f in os.listdir(path) if not f.startswith('.')]
        nums = len(imgs)
        print("there are {} imgs in {}.".format(nums, _))
        num_trains = nums*8 // 10
        for img in imgs[:num_trains]:
            shutil.copy(os.path.join(path, img), os.path.join(train_path, img))
        for img in imgs[num_trains:]:
            shutil.copy(os.path.join(path, img), os.path.join(val_path, img))


if __name__ == "__main__":
    main('/home/ubuntu/dataset/flower_photos')

