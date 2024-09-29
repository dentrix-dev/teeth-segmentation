import open3d as o3d
import numpy as np
def extract_coords(path:str) -> list :

    with open(path, 'r') as file:
        file_content = file.readlines()

    points = []
    for line in file_content[1:-1]:
        points.append([float(coord) for coord in line.split()])

    return points


def display_point_cloud(path: str):
    
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points)==0:
        print("file had extra tokens")
        pcd = extract_coords(path)
    coords_np = np.array(pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_np) 
    print(pcd)
    o3d.visualization.draw_geometries([pcd])



file_path = './data_samples/margin_line_ADA4.pts'
display_point_cloud(file_path)
