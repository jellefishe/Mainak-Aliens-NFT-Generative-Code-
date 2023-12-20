import glob
import os
import shutil
import json
import math
from PIL import Image
import string




# filepaths

#x represents the ID of the GIF 1 - 22,222
# directory = "C:/Users/localadmin/Documents/hashlips_art_engine-main/ManiakOut1/"
# count = 2248
# for f in os.listdir(directory):
    
#     if f.endswith('.mp4'):
#         os.rename("C:/Users/localadmin/Documents/hashlips_art_engine-main/ManiakOut1/"+str(f), "C:/Users/localadmin/Documents/hashlips_art_engine-main/ManiakOut1/"+str(count)+".mp4")
    #count = count + 1
for x in range(1,2014):

    #metadata location
    json_file = open("C:/Users/localadmin/Downloads/metadata_final/"+str(x)+".json","r")
    variables = json.load(json_file)

    

    
    # IMPORTANT IMPORANT IMPORTANT IMPORTANT #
    variables["image"] = "ipfs://QmTVJNsYWEhgCM7YnN1vduz3LqYWAUYeRyUZG2SVQMMm4L/"+str(x)+".mp4"
    # IMPORTANT IMPORANT IMPORTANT IMPORTANT #
    


    json_file = open("C:/Users/localadmin/Downloads/metadata_final/"+str(x)+".json","w")
    json.dump(variables,json_file)

    json_file.close()


