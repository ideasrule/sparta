import re
import requests
import sys
import os

#Change this if the Box link changes
top_folder = "d_154382715453"
shared_name = "tj1jnivn9ekiyhecl5up7mkg8xrd1htl"

def parse_names_and_IDs(text):
    typedIDs = []
    names = []
    locs = [m.start() for m in re.finditer('typedID', text)]
    for l in locs:
        typedID = text[l+10 : l+24]
        if typedID != top_folder:
            name_loc = text[l:].find('"name":') + l + 8
            name_end_loc = text[name_loc:].find(",") + name_loc - 1
            typedIDs.append(typedID)
            names.append(text[name_loc:name_end_loc])
    return names, typedIDs

def list_dir(folder_link):
    #Get all IDs on all pages
    page = 1
    all_names = []
    all_IDs = []

    while True:    
        response = requests.get('{}?page={}'.format(folder_link, page))
        names, IDs = parse_names_and_IDs(response.text)
        all_names += names
        all_IDs += IDs
        if len(IDs) == 0:
            break   
        print("Read page {} and found IDs {} with names {}".format(page, IDs, names))
        page += 1
    return all_names, all_IDs

def download_file(file_ID, download_location):
    direct_link = "https://app.box.com/index.php?rm=box_download_shared_file&shared_name={}&file_id={}".format(shared_name, file_ID)
    r = requests.get(direct_link)
    filename = re.findall("filename=(.+)", r.headers["Content-Disposition"])[0].split(";")[0].strip('"')
    print("Downloaded file {} with ID {}".format(filename, file_ID))
    with open(download_location + filename, "wb") as f:
        f.write(r.content)

def download_folder(folder_link, download_location):
    print("Downloading folder", folder_link)
    names, IDs = list_dir(folder_link)

    for i in range(len(names)):
        if IDs[i].startswith("f_"):
            download_file(IDs[i], download_location)
        elif IDs[i].startswith("d_"):
            link = "https://stsci.app.box.com/s/{}/folder/{}".format(shared_name, IDs[i][2:])
            os.mkdir(download_location + names[i])
            download_folder(link, download_location + names[i] + "/")
        else:
            assert(False)
            
    
if len(sys.argv) != 2:
    print("Give me a folder link, e.g. python download_files.py https://stsci.app.box.com/s/tj1jnivn9ekiyhecl5up7mkg8xrd1htl/folder/155356707447")
    exit(-1)
    
folder_link = sys.argv[1]
implied_shared_name = folder_link.split("/")[-3]
if shared_name != implied_shared_name:
    print("I got a shared name of {}.  It should be {} for our original JWST Box.".format(implied_shared_name, shared_name))

download_folder(folder_link, "./")    

