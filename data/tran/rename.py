import os


if __name__ == '__main__':
   path = "/home/chen/Desktop/抛撒物正报/补充b"
   count = 0

   filelist = os.listdir(path)
   for file in filelist:
       Olddir = os.path.join(path, file)
       if os.path.isdir(Olddir):
           continue
       filename = os.path.splitext(file)[0]
       filetype = os.path.splitext(file)[1]
       # import---------------------------------------------------------------------------------------------------------------
       Newdir = os.path.join(path, 'ffffff' + str(count) + filetype)
       os.rename(Olddir, Newdir)

       count += 1

