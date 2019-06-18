import os
import ftplib

yr_begin = 1982
yr_end = 2015
host_name = "ftp.cdc.noaa.gov"
username = "anonymous"
password = "xxxx"
path = "/Datasets/noaa.oisst.v2.highres/"
save_path = "Data/"

filenames = ["sst.day.anom."+str(i)+".nc" for i in list(range(yr_begin,yr_end+1))]

ftp = ftplib.FTP(host = host_name)
ftp.login(username,password)
ftp.cwd(path)

for j in filenames:
    print("loading "+j[-7:-3])
    ftp.retrbinary("RETR "+j,open(save_path+j[-7:],"wb").write)
