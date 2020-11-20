import numpy as np
import os
import configure

# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", required=True, nargs="+", help="path to video clip")
# # Check whether to use clip, else use images
# args = vars(ap.parse_args())
# test = args["video"]
# print(test)
# for clip in args["video"]:
#     print(clip)

# a = numpy.array([[2, 3, 1], [2, 4, 5]])
# print(a.shape)
# b = a.reshape([1, a.shape[0], a.shape[1]])
# print(b)
# print(b.shape)

# c = numpy.array([[[1, 2, 3], [2, 3, 4]], [[4, 5, 6], [5, 6, 7]]])
# print(c.shape)
# d = c.reshape(c.shape[0], -1)
# print(d)
# print(d.shape)

test = list()
for i in range(4):
    test.append(i+1)

k = np.argmax(test)
print(test[k])


