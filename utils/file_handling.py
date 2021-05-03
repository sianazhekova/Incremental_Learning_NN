# A file defining useful and common file handling functionality

def save_array_txt(file_path, array_type, arr_to_write):
    
    time_str = str(datetime.datetime.now())
            
    with open(file_path + "/" + array_type + " " + time_str + '.txt', 'w') as file_handle:
        file_handle.writelines("%s\n" % label for label in arr_to_write)


def txt_load_array(file_str, array_type):
    return_arr = []
    with open('file_str', 'r') as file_handle:
        filecontents = filehandle.readlines()

    for line in filecontents:
        # Remove the last character of the string representing a line break
        current_place = line[:-1]

        places.append(current_place)
    
    return return_arr

def save_array_json(file_path, array_type, arr_to_write):
    
    time_str = str(datetime.datetime.now())
            
    with open(file_path + "/" + array_type + " " + time_str + '.txt', 'w') as file_handle:
        file_handle.writelines("%s\n" % label for label in arr_to_write)


def load_json(file_str, file_path):
    
    with open(file_path + "/" + file_str + ".txt", 'r') as file_handle:
        return_arr = json.load(filehandle)
    
    return return_arr

def save_json(file_path, array_type, obj_to_write):

    with open(file_path + "/" + array_type + time_str + ".txt", 'w') as file_handle:
        json.dump(obj_to_write, file_handle)
