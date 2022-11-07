import cv2
import numpy as np
import cv2 as cv


hsv_min = np.array((21, 174, 0), np.uint8)
hsv_max = np.array((179, 255, 255), np.uint8)


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
    cv2.drawContours(mask, flower_contours[0], -1, (255, 0, 0), 2, cv.LINE_AA)
    cv.imshow("rr", mask)

    cv.drawContours(img, flower_contours, -1, (255, 0, 0), 2, cv.LINE_AA)
    cv.imshow('All_con', img)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()


