import os


if __name__ == '__main__':
   path = "/home/chen/chen_p/chen_yolo4/datasets/Material/data/4box_a/img"
   count = 0

   filelist = os.listdir(path)
   for file in filelist:
       Olddir = os.path.join(path, file)
       if os.path.isdir(Olddir):
           continue
       filename = os.path.splitext(file)[0]
       filetype = os.path.splitext(file)[1]
       # import---------------------------------------------------------------------------------------------------------------
       Newdir = os.path.join(path, 'sdgx' + str(count) + '.jpg')
       os.rename(Olddir, Newdir)

       count += 1

