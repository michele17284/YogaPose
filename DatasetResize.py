import os, cv2, json


def getfilename(file_id):
    new_file_id = str(file_id)
    zeros = ['0' for i in range(7-len(new_file_id))]
    return "".join(zeros) + new_file_id

def rename(path, out):
    DATASET_PATH = path
    OUT_DATASET_PATH = out
    size=(256,192)
    file_id = 0 
    jsonfile = {}
    for idx, directory_class in enumerate(os.listdir(DATASET_PATH)):
        idxname_tuple = (idx, directory_class)
        classlist = []
        class_path = os.path.join(DATASET_PATH,directory_class)
        for image_file in os.listdir(class_path):
            if not image_file.endswith('.gif'):
                f = cv2.imread(os.path.join(class_path, image_file),cv2.IMREAD_COLOR)
                #print(f.shape,"before resize")
                height,width, channels = f.shape
                if height > width:
                    scale_percent = size[0]/height
                    f = cv2.resize(f,(int(width*scale_percent),size[0]))
                    if f.shape[1] > size[1]:
                        scale_percent = size[1]/width
                        f = cv2.resize(f,(size[1],int(height*scale_percent)))
                else:
                    scale_percent = size[1]/width
                    f = cv2.resize(f,(size[1],int(height*scale_percent)))
                    if f.shape[0] > size[1]:
                        scale_percent = size[0]/height
                        f = cv2.resize(f,(int(width*scale_percent),size[0]))

                height,width, channels = f.shape
                #print(f.shape,"after resize")
                f = cv2.copyMakeBorder(f, size[0]-height, 0, size[1]-width, 0, cv2.BORDER_CONSTANT,value=0)
                #print(f.shape,"after padding")
                filename = getfilename(file_id) + ".jpg"
                classlist.append(filename)
                outclass_path = os.path.join(OUT_DATASET_PATH, str(idx))
                if not (os.path.isdir(outclass_path)):
                    os.makedirs(outclass_path)
                
                cv2.imwrite(os.path.join(outclass_path,filename) , f)
                file_id +=1
        print("Class Done")
        jsonfile[idxname_tuple[0]] = {"class_name": idxname_tuple[1], "images_list": classlist}
    with open('annotation.json', 'w') as outfile:
        json.dump(jsonfile, outfile)


#rename("./dataset/", "./resized_dataset")