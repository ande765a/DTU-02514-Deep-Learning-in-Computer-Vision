import os, gdown, zipfile
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    digit_shape = 32

    if not os.path.exists(f'./SVHN'):
            url = 'https://drive.google.com/uc?id=1RWNq8JP5SXi07NLc1bgqN39FrsmB3KA7'
            gdown.download(url, './SVHN.zip', quiet=False)
            try:
                with zipfile.ZipFile('./SVHN.zip') as z:
                    z.extractall('SVHN')
                    print("Extracted", 'SVHN.zip')
            except:
                print('Invalid file')
    
    directories = ['SVHN/train/', 'SVHN/test/']
    for out_directory in directories:
        directory = os.path.join("SVHN", out_directory)
        background_dir = os.path.join(directory, "background/")
        if not os.path.exists(background_dir): 
            os.mkdir(background_dir)


        for filename in os.listdir(directory):
            if filename.endswith('.png'):
                f = os.path.join(directory, filename)

                if os.path.isfile(f):
                    f_name = filename[:-4] 
                    print(f_name)
                    bbox_path = directory + f_name + '.csv'
                    img = plt.imread(f)
                    if img.shape[0] < digit_shape or img.shape[1] < digit_shape:
                        print(f'Skipping {f_name}')
                        continue
                    bboxs = np.genfromtxt(bbox_path, delimiter=',')
                    _, left, _, width, _ = bboxs[1]
                    right = left+width
                    for j in range(1,len(bboxs)):
                        # left most point
                        if left > bboxs[j,1]:
                            left = bboxs[j,1]
                        if right < bboxs[j,1]+bboxs[j,3]:
                            right = bboxs[j,1]+bboxs[j,3]
                    left, right = int(left), int(right)

                    def loop():
                        i = 0
                        for k in range(0, img.shape[1]-digit_shape-1, digit_shape // 2):
                            for l in range(0, img.shape[0]-digit_shape-1, digit_shape // 2):
                                # if overlap with digit label - continue
                                if (k > left and k < right) or (k+digit_shape > left and k+digit_shape < right):
                                    continue

                                plt.imsave(f'{background_dir}{f_name}_{k}{l}.jpg', img[l:l+digit_shape,k:k+digit_shape])
                                i += 1
                                if i == 3:
                                    return
                    
                    loop()

