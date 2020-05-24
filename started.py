import os
from PIL import Image
from torchvision import transforms
root = './data/mnt/ramdisk/max/90kDICT32px'
datas = []
with open('./data/mnt/ramdisk/max/90kDICT32px/annotation_val.txt','r') as f:
    for i in range(10):
        print(f.readline())


# label = datas[0].split('/')[-1].split('.')[0].split('_')[-2]
# print(label)
# img_path = root + datas[0]
# print(img_path)
# img = Image.open(img_path)
# print(img.size)
# trans = transforms.Compose([transforms.Resize((32,100))])
# new_img = trans(img)
# new_img.save('./test.jpg')
# image = Image.open('./test.jpg')
# print(image.size)




