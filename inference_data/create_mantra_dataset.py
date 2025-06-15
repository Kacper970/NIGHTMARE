# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 13:03:55 2025

@author: kacpe
"""
import os
from PIL import Image
import json


with open("Data/Mantra/annotation.json", 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    
list_comics = os.listdir("Data/Mantra/images")

names = {
    1:'tencho_isoro',
    2:'tojime_no_siora',
    3:'rasetugari',
    4:'balloon_dream',
    5:'boureisougi'
    }


lookup_dict = {}
    

for comic in list_comics:
    
    img_list = []
    
    list_images = os.listdir(f"Data/Mantra/images/{comic}")
    
    for image_ in list_images:
        
        image_file = Image.open(f"Data/Mantra/images/{comic}/{image_}")
        page_num = int(image_.split('.')[0]) 
        
        for com in json_data:
           
            if com['book_title'] == names[int(comic)]:
                
                for page in com['pages']:
                    
                    if page['page_index'] == (page_num+1):
                        
                        panel_num = 0
                        
                        for frame in page['frame']:
                            
                            left = frame["x"]
                            top = frame["y"]
                            right = left + frame["w"]
                            bottom = top + frame["h"]
                            
                            cropped_image = image_file.crop((left, top, right, bottom))
        
                            cropped_image.save(f"Data/Mantra/cut_images/{comic}/{page_num}_{panel_num}.jpg")
                            
                            key = (int(comic), page_num, panel_num) 
                            
                            for translation in page['text']:
                                
                                left_ = translation["x"]
                                top_ = translation["y"]
                                right_ = left + translation["w"]
                                bottom_ = top + translation["h"]
                                
                                if (left_ >= left and right_ <= right and top_ >= top and bottom_ <= bottom):
                                
                                    text = translation['text_zh']
                                    
                                    if key not in lookup_dict:
                                        
                                        lookup_dict[key] = text 
                                        
                                    else:
                                        
                                        lookup_dict[key] += (' ' + text)
                            
                            panel_num+=1

file_path = 'Data/Mantra/zh_lookup_dict_mantra.json'

data_str_keys = {str(k): v for k, v in lookup_dict.items()}

# Open the file in write mode and save the dictionary as JSON
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(data_str_keys, file, ensure_ascii=False, indent=4)