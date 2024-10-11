# app/utils.py
import configparser
import os

def save_last_image_path(name, path):
    config = configparser.ConfigParser()
    config[name] = {'Path': path}
    
    # Get the absolute path to the 'resources' directory
    resources_dir = os.path.join(os.path.dirname(__file__), '../resources')
    
    # Make sure the directory exists
    os.makedirs(resources_dir, exist_ok=True)
    
    # Define the path to the config file
    config_path = os.path.join(resources_dir, 'config.ini')
    
    # Write to the config file
    with open(config_path, 'w') as configfile:
        config.write(configfile)

def load_last_image_path(name):
    config = configparser.ConfigParser()
    config.read('resources/config.ini')
    return config.get(name, 'Path', fallback='')
