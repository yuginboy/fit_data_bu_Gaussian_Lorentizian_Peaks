import sys
import os
from io import StringIO

runningScriptDir = os.path.dirname(os.path.abspath(__file__))
# get root project folder name:
runningScriptDir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

def create_out_data_folder(main_folder_path, first_part_of_folder_name = ''):
    '''
    create out data directory like 0005 or 0004
    :param main_folder_path: path to the main project folder
    :return: full path to the new directory
    return folder path like: main_folder_path + first_part_of_folder_name + '%04d' % i
    '''
    checkFile = 1
    i = 1

    # check, if first_part_of_folder_name is not absent then add '_' symbol to the end
    if len(first_part_of_folder_name) > 0:
        first_part_of_folder_name += '_'

    while checkFile > 0:

        out_data_folder_path = os.path.join( main_folder_path, first_part_of_folder_name + '%05d' % i )
        if  not (os.path.isdir(out_data_folder_path)):
            checkFile = 0
            os.makedirs(out_data_folder_path, exist_ok=True)
        i+=1
    return  out_data_folder_path

def create_unique_out_data_file(main_folder_path, first_part_of_file_name = '', ext = 'txt'):
    '''
    create out data directory like 0005 or 0004
    :param main_folder_path: path to the main project folder
    :return: full path to the new directory
    return folder path like: main_folder_path + first_part_of_folder_name + '%04d' % i
    '''
    checkFile = 1
    i = 1

    # check, if first_part_of_folder_name is not absent then add '_' symbol to the end
    if len(first_part_of_file_name) > 0:
        first_part_of_file_name += '_'

    while checkFile > 0:

        out_data_file_path = os.path.join( main_folder_path, first_part_of_file_name + '%05d' % i + '.' + ext)
        if  not (os.path.isfile(out_data_file_path)):
            checkFile = 0
        i+=1
    return  out_data_file_path


def create_all_dirs_in_path_if_their_not_exist(folder_path):
    # create all dirs in path if their not exist
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    return folder_path


# def create_out_data_folder(main_folder_path):
#     '''
#     create out data directory like 0005 or 0004
#     :param main_folder_path: path to the main project folder
#     :return: full path to the new directory
#     '''
#     checkFile = 1
#     i = 1
#     while checkFile > 0:
#
#         out_data_folder_path = os.path.join( main_folder_path, '%04d' % i )
#         if  not (os.path.isdir(out_data_folder_path)):
#             checkFile = 0
#             os.makedirs(out_data_folder_path, exist_ok=True)
#         i+=1
#     return  out_data_folder_path


def listOfFiles(dirToScreens):
    '''
    return only the names of the files in directory
    :param folder:
    :return:
    '''
    '''
    :param dirToScreens: from which directory you want to take a list of the files
    :return:
    '''
    files = [f for f in os.listdir(dirToScreens) if os.path.isfile(os.path.join(dirToScreens,f))]
    return files

def listOfFilesFN(folder):
    '''
    Return list of full pathname of files in the directory

    '''
    files = listOfFiles(folder)
    return [os.path.join(folder,f) for f in os.listdir(folder)]

def listOfFilesFN_with_selected_ext(folder, ext = 'png'):
    '''
    Return list of full pathname of files in the directory

    '''
    files = listOfFiles(folder)
    return [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(ext)]

def deleteAllFilesInFolder(folder):
    # delete all files in the current directory:
    filelist = [ f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]
    for f in filelist:
        os.remove(f)
    return None

def listdirs(folder):
    '''
    return only the names of the subdirectories
    :param folder:
    :return:
    '''
    '''
    :param folder:
    :return:
    '''
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def listdirsFN(folder):
    '''
    Return list of full pathname subdirectories
    '''
    return [
        d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
        if os.path.isdir(d)
    ]

def createFolder (folder_name):
    '''
    check for exist and if it is then create folder_name
    :param folder_name: full folder path name
    :return: folder_name of created directory
    '''
    if  not (os.path.isdir(folder_name)):
        os.mkdir(folder_name)
    return folder_name

def get_upper_folder_name(file_path):
    # return only a directory name when file is placed
    return os.path.split(os.path.split(os.path.normpath(file_path))[0])[1]

def get_folder_name(file_path):
    # return only a directory name when file is placed
    return os.path.split(os.path.normpath(file_path))[1]
if __name__ == "__main__":
    print ('-> you run ',  __file__, ' file in a main mode' )