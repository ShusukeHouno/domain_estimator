import normalize
from PIL import Image, ImageOps, ImageDraw
import csv
import os

class DataAugument():
    def __init__(self, dataset):
        self.train_image_path, self.train_label_data = dataset.getData()
    # create data
    def augument(self, saveImagePath, CSVname, flip=False):
        os.makedirs(str(saveImagePath) + "/augumentImage/", exist_ok=True)
        augument_Image_Path = []
        augument_Label_Data = []
        if flip==True:
            self.flipImage()
        # cut 80%
        for index, image_path in enumerate(self.train_image_path):
            # image_path split
            split_image_path = image_path.split('/')
            image = Image.open(image_path)
            # get image size(clipping size)
            width, height = image.size
            width *= 0.8
            height *= 0.8
            width_df, height_df = image.size
            width_df *= 0.1
            height_df *= 0.1
            # after augument datapath
            augument_Image_Path.append(str(saveImagePath) + '/augumentImage/' 
                                        + split_image_path[-2] + "/"+ split_image_path[-1])
            augument_Label_Data.append(self.train_label_data[index])
            image.save(str(saveImagePath) + '/augumentImage/' 
                        + split_image_path[-2] + "/" + split_image_path[-1])
            # linear equation
            inclination, section = self.calcLine(self.train_label_data[index])
            # clipping image
            for repeat_height in range(0, 3):
                for repeat_width in range(0, 3):
                    # image clipping
                    # print([repeat_height, repeat_width])
                    # print([width_df * repeat_width, height_df * repeat_height, width + width_df * repeat_height, height + height_df * repeat_height])
                    image_crop = image.crop((width_df * repeat_width, # start point x
                                            height_df * repeat_height,   # start point y
                                            width + (width_df * repeat_width), # end point x
                                            height + (height_df * repeat_height))) # end point y
                    # calc start and end point
                    clip_label_list = [self.calcLinePoint(inclination, section, 0.8 + 0.1 * repeat_height, self.train_label_data[index][0]),
                                        0.8 + 0.1 * repeat_height,
                                        self.train_label_data[index][2],
                                        self.train_label_data[index][3]]
                    if clip_label_list[1] - clip_label_list[3] < 0.1:
                        clip_label_list[3] = 0.7 + 0.1 * repeat_height
                        clip_label_list[2] = self.calcLinePoint(inclination, section, 0.7 + 0.1 * repeat_height, self.train_label_data[index][2])
                    if self.areaDecision(clip_label_list, 
                                        0.1 * repeat_width, 
                                        0.8 + 0.1 * repeat_width,
                                        0.1 * repeat_height,
                                        0.8 + 0.1 * repeat_height):
                        # print("image crop failed")
                        continue
                    clip_label_list[0] = (clip_label_list[0] - 0.1 * repeat_width) / 0.8
                    clip_label_list[1] = (clip_label_list[1] - 0.1 * repeat_height) / 0.8
                    clip_label_list[2] = (clip_label_list[2] - 0.1 * repeat_width) / 0.8
                    clip_label_list[3] = (clip_label_list[3] - 0.1 * repeat_height) / 0.8
                    augument_Label_Data.append(clip_label_list)
                    # debug image
                    test_point = [clip_label_list[0] * width, clip_label_list[1] * height,
                                clip_label_list[2] * width, clip_label_list[3] * height]
                    self.paintLine(image_crop, test_point, "debugImage" + split_image_path[-1].replace('.png', '')
                                    + str(repeat_height + 1) + str(repeat_width + 1))
                    # save image
                    image_crop.save(str(saveImagePath) + '/augumentImage/' 
                                    + split_image_path[-2] + "/" + split_image_path[-1].replace('.png', '')
                                    + str(repeat_height + 1) + str(repeat_width + 1)
                                    + '.png')
                    # create image path list
                    augument_Image_Path.append(str(saveImagePath) + '/augumentImage/' 
                                    + split_image_path[-2] + "/" + split_image_path[-1].replace('.png', '')
                                    + str(repeat_height + 1) + str(repeat_width + 1)
                                    + '.png')
        self.createCSV(CSVname, augument_Image_Path, augument_Label_Data)
    
    def paintLine(self, image, point, image_path):
        os.makedirs("debugImage", exist_ok=True)
        canvas = image.copy()
        canvas_draw = ImageDraw.Draw(canvas)
        canvas_draw.line((point[0], point[1], point[2], point[3]))
        canvas.save("./debugImage/" + str(image_path) + ".png")

    def areaDecision(self, point, width_lower, width_upper, height_lower, height_upper):
        if (width_lower < point[0] <= width_upper) and (width_lower < point[2] <= width_upper):
            if (height_lower <= point[1] <= height_upper) and (height_lower <= point[3] <= height_upper):
                return False
        return True
    
    # calc x_point
    def calcLinePoint(self, inclination, section, point_height, if_x):
        if inclination != 1.5:
            return (point_height - section) / inclination
        else:
            return if_x

    # linear equation
    def calcLine(self, image_point):
        x_start, y_start, x_end, y_end = image_point
        if x_start != x_end:
            inclination = (y_start - y_end) / (x_start - x_end)
            section = y_end - x_end * inclination
            return inclination, section
        else: # ???
            return 1.5 ,0

    def flipImage(self):
        update_image_path, updata_label_data = []
        for index, image_path in enumerate(self.train_image_path):
            # check for name conflicts
            if image_path in "_flip":
                continue
            # create flip image
            image = Image.opne(image_path)
            widht, height = image.size
            image_flip = ImageOps.flip(image)
            # name flip image
            flip_image_name = image_path.replace(".png", "_flip.png")
            # save image
            image_flip.save(flip_image_name)
            # list update(before original data)
            update_image_path.appned(image_path)
            update_label_data.append(self.train_label_data[index])
            # list update(after update data)
            ## create new label data
            label_data = [width - self.train_label_data[index][0],
                        self.train_label_data[index][1],
                        widht - self.train_label_data[index][2],
                        self.train_label_data[index][3]]
            update_image_path.append(flip_image_name)
            update_label_data.append(label_data)

        # update train data(class variable)
        self.train_image_path = update_image_path
        self.train_label_data = update_label_data

    def createCSV(self, saveName, Image_path, Label_data):
        with open(str(saveName) + '.csv', 'w') as fileCSV:
            writerCSV = csv.writer(fileCSV)
            writerCSV.writerow(['image_path', 'x_start', 'y_start', 'x_end', 'y_end'])
            for index, image_path in enumerate(Image_path):
                writerCSV.writerow([image_path, 
                                    Label_data[index][0],
                                    Label_data[index][1],
                                    Label_data[index][2],
                                    Label_data[index][3]])
        print('save CSV file: ' + str(saveName) + '.csv')

if __name__ == "__main__":
    train_dataset = normalize.Dataset("sakaki_data_train.csv")
    data = DataAugument(train_dataset)
    data.augument('./data', 'Augument_train_sakaki')
