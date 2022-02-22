import re
import requests
import sys

def find_file_IDs(text):
    '''Very hacky way to parse webpage to get IDs of all files on page'''
    locs = [m.start() for m in re.finditer('typedID', text)]
    all_typedIDs = []
    for l in locs:
        typedID = text[l+10 : l+24]
        if typedID[0] == "f": 
            all_typedIDs.append(typedID)  
    return all_typedIDs


if len(sys.argv) != 2:
    print("Give me a folder link, e.g. python download_files.py https://stsci.app.box.com/s/tj1jnivn9ekiyhecl5up7mkg8xrd1htl/folder/155356707447")
    exit(-1)
    
folder_link = sys.argv[1]
shared_name = folder_link.split("/")[-3]
print("I got a shared name of {}.  It should be tj1jnivn9ekiyhecl5up7mkg8xrd1htl for our original JWST Box.".format(shared_name))

#Get all file IDs on all pages
page = 1
all_file_IDs = []

while True:    
    response = requests.get('{}?page={}'.format(folder_link, page))
    file_IDs = find_file_IDs(response.text)
    all_file_IDs += file_IDs
    if len(file_IDs) == 0:
        break   
    print("Read page {} and found file IDs {}".format(page, file_IDs))
    page += 1

print("Downloading...")
#Download all files on all pages
for file_ID in all_file_IDs:
    direct_link = "https://app.box.com/index.php?rm=box_download_shared_file&shared_name={}&file_id={}".format(shared_name, file_ID)
    r = requests.get(direct_link)
    filename = re.findall("filename=(.+)", r.headers["Content-Disposition"])[0].split(";")[0].strip('"')
    print("Downloaded file {} with ID {}".format(filename, file_ID))
    with open(filename, "wb") as f:
        f.write(r.content)
