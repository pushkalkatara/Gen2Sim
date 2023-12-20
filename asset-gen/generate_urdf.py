import xml.etree.ElementTree as ET
import os 
import random
import sys 

import shutil

from llm_utils import query


class AssetToURDF():
    def __init__(self):
        self.prompt = '''
            Average mass of {} in grams:
            Average length, width, height of {} in centimeters:

            Following are some examples:
            1. 
            Average mass of apple in grams: 100-200
            Average length, width, height of apple in centimeters: 6-7, 6-7, 6-7

            2. 
            Average mass of lemon in grams: 50-100
            Average length, width, height of lemon in centimeters: 5-7, 4-5, 4-5
        '''

    def parse(self, response):
        response = response.split('\n')
        first, second = response[0], response[1]
        attr = dict()
        mass_range = first.split(": ")[-1]
        attr["min_mass"] = int(mass_range.split("-")[0])
        attr["max_mass"] = int(mass_range.split("-")[1])
        
        size = second.split(": ")[-1]
        for r, n in zip(size.split(", "), ["length", "width", "height"]):
            attr[f"min_{n}"] = int(r.split("-")[0])
            attr[f"max_{n}"] = int(r.split("-")[1])
        return attr

    def generate_urdf(self, name, path="../output"):
        prompt = self.prompt.format(name, name)
        try:
            response = query(prompt, path)
            attr_dict = self.parse(response)
            self.generate_urdf_file(name, attr_dict, path)
        except Exception as err:
            print(err)

    def generate_urdf_file(self, name, attr_dict, save_path="../output"):
        et = ET.parse("urdf_template.urdf")
        robot = et.getroot()
        robot.set("name", name)
        link = robot[0]
        link.set("name", name)
        p = link[0][1][0]
        p.set("filename", "mesh.obj")
        p.set("scale", "1.0")
        p = link[1][1][0]
        p.set("filename", "mesh.obj")
        # later update scale based on length, width, height.
        p.set("scale", "1.0")
        p = link[2][0]
        p.set("value", str(random.randint(attr_dict["min_mass"], attr_dict["max_mass"])/1000))
        et.write(f"{save_path}/mesh/urdf.urdf")
    
    def copy_asset(self, name, path, dest):
        mesh_folder = os.path.join(path, "mesh")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(mesh_folder, dest)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python generate_urdf.py <assets_path> <dest>, Using default paths")
        assets_path = "/projects/katefgroup/gen2sim/output/"
        dest = "/projects/katefgroup/gen2sim/assets/"
    else:
        assets_path = sys.argv[1]
        dest = sys.argv[2]

    asset_to_urdf = AssetToURDF()
    for name in os.listdir(assets_path):
        print("Processing:", name)
        if not name.endswith("dmtet"):
            continue
        asset_category = name[:-6]
        asset_to_urdf.generate_urdf(asset_category, os.path.join(assets_path, name))
        asset_to_urdf.copy_asset(asset_category, os.path.join(assets_path, name), os.path.join(dest, asset_category))