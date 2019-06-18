import os
import ftplib

yr_begin = 1982
yr_end = 2015
host_name = "ftp.cdc.noaa.gov"
username = "anonymous"
password = "xxxx"
path = "/Datasets/noaa.oisst.v2.highres/"
save_path = "Data/"

filename_begin = "sst.day.anom."
filenames = []
for i in range(yr_begin,yr_end+1):
    fn_ends.append(filename_begin+str(i)+".nc")

ftp = ftplib.FTP(host = host_name)
ftp.login(username,password)
ftp.cwd(path)

for j in filenames:
    print("loading "+j[-7:-3])
    ftp.retrbinary("RETR "+j,open(save_path+j[-7:],"wb").write)
