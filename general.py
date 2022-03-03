import os 
import json

def wrapperargs(func, args):
    '''Takes in input a function and a list of arguments,
    return func(*args)'''
    return func(*args)

def json_save(data, file_name, dir_name=''):
    dir_name = [dir_name] if ((type(dir_name) != list) and (dir_name)) else dir_name
    assert type(file_name) == str
    file_name = file_name + '.json' if not file_name.endswith('.json') else file_name
    full_path = os.path.join(*dir_name, file_name)
    with open(full_path, 'w') as fp:
        json.dump(data, fp)