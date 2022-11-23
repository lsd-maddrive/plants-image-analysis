import cv2
import numpy as np
import cv2 as cv
import torch
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *
from torchvision.models import densenet121
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
hsv_min = np.array((21, 174, 0), np.uint8)
hsv_max = np.array((179, 255, 255), np.uint8)
LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']


# def get_data(fold):
#     datablock = DataBlock(
#         blocks=(ImageBlock, CategoryBlock(vocab=LABEL_COLS)),
#         getters=[
#             ColReader('image_id', pref=IMG_PATH, suff='.jpg'),
#             ColReader('label')
#         ],
#         splitter=IndexSplitter(train_df_bal.loc[train_df_bal.fold==fold].index),
#         item_tfms=Resize(IMG_SIZE),
#         batch_tfms=aug_transforms(size=IMG_SIZE, max_rotate=30., min_scale=0.75, flip_vert=True, do_flip=True)
#     )
#     return datablock.dataloaders(source=train_df_bal, bs=BS)
def get_contours(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # меняем цветовую модель с BGR на HSV
    thresh = cv.inRange(hsv, hsv_min, hsv_max)
    # применяем цветовой фильтр
    # ищем контуры и складируем их в переменную contours
    se = np.ones((1, 1), dtype='uint8')
    image_close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, se)

    contours, hierarchy = cv.findContours(image_close.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(hierarchy)
    return contours


def get_flower_contours(contours):
    # hierarchy хранит информацию об иерархии
    # Perform morphology
    new_countours = []
    for index in range(len(contours)):
        if cv2.contourArea(contours[index]) > 300:
            new_countours.append(contours[index])
    return new_countours

def colorRange(ax, ay, az, bx, by, bz, dist):
    def testPixel(cx, cy, cz):
        return (
            (cx-ax)**2 + (cy-ay)**2 + (cz-az)**2
            + (cx-bx)**2 + (cy-by)**2 + (cz-bz)**2
        ) < dist**2
    return testPixel


def loadmodel():
    # model = create_vision_model(densenet121, 4)
    # print(torch.load("model_fold_1.pth", map_location=torch.device('cpu')))
    # print(torch.load("model_fold_1.pth", map_location=torch.device('cpu')))
    import re
    model = densenet121(weights=torch.load("model_fold_1.pth", map_location=torch.device('cpu')))
    model.eval()
    print(model.__dir__())
    # checkpoint = torch.load("model_fold_1.pth", map_location=torch.device('cpu'), weights=DenseNet121_Weights.DEFAULT)

    # modify:
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls.
    # This pattern is used to find such keys.
    # pattern = re.compile(
    #     r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    # state_dict = checkpoint
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # model.load_state_dict(checkpoint)

    # model.load_state_dict(checkpoint['state_dict'])
    # start_epoch = checkpoint['epoch']
    # loaded = torch.load("model_fold_1.pth", map_location=torch.device('cpu'))
    # for item in loaded.keys():
    #     print(item)
    # model.load_state_dict(torch.load("model_fold_1.pth", map_location=torch.device('cpu')))
    # model.eval()
    # return model
    return model

def comp_metric(preds, targs, labels=range(len(LABEL_COLS))):
    # One-hot encode targets
    targs = np.eye(4)[targs]
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])

def healthy_roc_auc(*args):
    return comp_metric(*args, labels=[0])

def multiple_diseases_roc_auc(*args):
    return comp_metric(*args, labels=[1])

def rust_roc_auc(*args):
    return comp_metric(*args, labels=[2])

def scab_roc_auc(*args):
    return comp_metric(*args, labels=[3])

def main():
    img = cv.imread("merged-min.png")
    contours = get_contours(img)
    flower_contours = get_flower_contours(contours)

    mask = np.zeros(img.shape[:2], np.uint8)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    print(len(flower_contours))
    flower = 1
    for contour in flower_contours:
        count = 0
        discount = 0
        x, y, w, h = cv2.boundingRect(contour)
        for p_x in range(x, x + w):
            for p_y in range(y, y + h):
                if cv2.pointPolygonTest(contour, [p_x, p_y], False) == 1.0:

                    if hsv[p_y][p_x][0] > hsv_min[0] and hsv[p_y][p_x][0] < hsv_max[0] and hsv[p_y][p_x][1] > hsv_min[1] and hsv[p_y][p_x][1] < hsv_max[1] and hsv[p_y][p_x][2] > hsv_min[0] and hsv[p_y][p_x][2] < hsv_max[2]:
                        count += 1
                    else: discount += 1

        print("Цветочек", flower, " здоров на: ", count / (discount + count), " %")
        flower += 1
        # centered to template of model
        img_mod = img[y: y+h, x: x+w]

        # cv2.imwrite("1", img_mod)
        new_w = w * 1.33
        new_h = h * 1.33

        max_side_len = new_w
        if max_side_len < new_h:
            max_side_len = new_h

        bias_x = (w - max_side_len) / 2
        bias_y = (h - max_side_len) / 2
        print(bias_x, bias_y)
        crop_img = img[y + int(bias_y):y + int(new_h), x + int(bias_x):x + int(new_w)]

        cv2.imshow("cropped" + str(flower), crop_img)
        resized = cv2.resize(crop_img, (512, 512))

        cv2.imwrite("./images/Test_" + str(flower) + '.jpg', resized)
        transform = transforms.ToTensor()
        image1 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = transform(image1)
        model = loadmodel()
        print(model)
        pred = model(torch.unsqueeze(tensor, 0))
        print(len(pred[0]))
        sum1 = 0
        for item in range(len(pred[0])):
            if item % 4 == 0:
                sum1 += pred[0][item]
        print(sum1)
        # DataLoader(test_ds, batch_size=300, num_workers=2)

        # model.predict(resized)

    # cv.imshow("rr", mask)
    cv2.drawContours(mask, flower_contours[0], -1, (255, 0, 0), 2, cv.LINE_AA)
    cv.drawContours(img, flower_contours, -1, (255, 0, 0), 2, cv.LINE_AA)
    # cv.imshow('All_con', img)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()


