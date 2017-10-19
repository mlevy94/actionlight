from desktopmagic.screengrab_win32 import getDisplayRects, getRectAsImage
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process, Queue
import rgbxy
from phue import Bridge
import time

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
regions = 2
lights = ["Hue TV 1", "Hue TV 2"]


def regionprocess(light, imgQ, senderQ):
    converter = rgbxy.Converter(rgbxy.GamutC)
    while True:
        image = imgQ.get()
        pixles = np.float32(image)[0]
        ret, label, centers = cv2.kmeans(pixles, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        points = list(map(lambda x: pixles[label.ravel() == x], range(K)))
        colors = list(map(lambda y: "".join(list(map(lambda x: "{0:02X}".format((int(x))), centers[y]))), range(K)))
        weights = list(map(len, points))
        dominant = centers[weights.index(max(weights))]
        try:
            xy = converter.rgb_to_xy(*dominant)
        except ZeroDivisionError:
            continue
        brightness = (max(dominant) - min(dominant)) / (255 - abs(max(dominant) + min(dominant) - 255))
        if brightness in [0, np.nan]:
            brightness = max(dominant) / 255
        print(brightness)
        print(colors)
        print(list(map(lambda x: "{: 6}".format(x), weights)))
        print("Dominant: {}".format(dominant))
        # plotmeans(image, points, center, colors)
        senderQ.put((light, xy, brightness))


def senderprocess(bridge, queue):
    while True:
        light, color, brightness = queue.get()
        bridge.set_light(light, {"xy": color, "bri": int(brightness * 255)})


def makeprocess(light, senderQ):
    imgQ = Queue()
    process = Process(target=regionprocess, args=[light, imgQ, senderQ])
    process.start()
    return process, imgQ


def main(addr):
    bridge = Bridge(addr)
    senderQ = Queue()
    senderP = Process(target=senderprocess, args=[bridge, senderQ])
    senderP.start()
    processpool, queuepool = list(zip(*map(lambda x: makeprocess(*x), zip(lights, [senderQ] * regions))))
    display = getDisplayRects()[1]

    # start = time.time()
    # for _ in range(15):
    while True:
        start = time.time()
        imDisplay = getRectAsImage(display)
        list(map(lambda x: x.put(imDisplay), queuepool))
        sleeptime = 0.2 - (time.time() - start)
        if sleeptime > 0:
            time.sleep(sleeptime)
    # print("Time: ", time.time() - start)


def plotmeans(imDisplay, points, center, colors):
    # Plot the data
    fig = plt.figure()
    ax = Axes3D(fig)
    _ = list(map(lambda x: ax.scatter(points[x][:, 0], points[x][:, 1], points[x][:, 2], c=colors[x]), range(K)))
    ax.scatter(center[:, 0], center[:, 1], center[:, 2], s=80, c='yellow', marker='s')
    imDisplay.show()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main("192.168.7.5")
