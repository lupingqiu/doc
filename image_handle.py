from PIL import Image
import glob, os
import time

start = time.clock()

size = 64, 64

outputdir = "/home/rube/tf/resized/"
img_count = 0
for infile in glob.glob("/home/rube/tf/source/*.jpg"):
    file, ext = os.path.splitext(infile)
    filename = file.split("/")[-1]
    im = Image.open(infile)
    im.thumbnail(size, Image.ANTIALIAS)
    im.save(outputdir+filename + ext, "JPEG")
    img_count=img_count+1

end = time.clock()
print "read: %f s" % (end - start)
print img_count
import os

label_list = []
image_list = []
size=64,64
resized_path = "/data/rube/resized/"
source_file = open("/data/rube/000000_0")

for line in source_file:
	numiid,cat,vid,pic = line.split("\001")
	rfile = resized_path+numiid+.jpg
	if os.path.exists(rfile)
	    if vid == "20524":
            label_list.append([1,0,0,0,0,0,0])
        elif vid == "20525":
        	label_list.append([0,1,0,0,0,0,0])
        elif vid == "30271":
        	label_list.append([0,0,1,0,0,0,0])
        elif vid == "30272":
        	label_list.append([0,0,0,1,0,0,0])
        elif vid == "30465":
        	label_list.append([0,0,0,0,1,0,0])
        elif vid == "72202018":
        	label_list.append([0,0,0,0,0,1,0])
        elif vid == "9146553":
        	label_list.append([0,0,0,0,0,0,1])
        img= Image.open(rfile)
        ##img.thumbnail(size,Image.ANTIALIAS)
        imgnp = np.array(img)
        imgnp = imgnp.astype(np.float32)
        imgnp = np.multiply(imgnp,1.0/255.0)
        image_list.append(np.array(imgnp))

x_all,y_all = np.array(image_list),np.array(label_list)
x_train,y_train = x_all[0:45000],y_all[0:45000]
x_test,y_test = x_all[45000:],y_all[45000:]

train_data_size=45000
index_in_epoch = 0
epochs_completed = 0

def next_batch(batch_size):
	start = index_in_epoch
	index_in_epoch += batch_size
	if index_in_epoch > train_data_size:
		start = 0
		index_in_epoch = batch_size
    end = index_in_epoch
    return x_train[start:end],y_train[start,end]





