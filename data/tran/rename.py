import os


if __name__ == '__main__':
   path = "/media/chen/U/xx"
   count = 0

   filelist = os.listdir(path)
   for file in filelist:
       Olddir = os.path.join(path, file)
       if os.path.isdir(Olddir):
           continue
       filename = os.path.splitext(file)[0]
       filetype = os.path.splitext(file)[1]
       # import---------------------------------------------------------------------------------------------------------------
       Newdir = os.path.join(path, 'ssss' + str(count) + filetype)
       os.rename(Olddir, Newdir)

       count += 1

