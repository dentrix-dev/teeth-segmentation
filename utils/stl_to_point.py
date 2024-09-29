import open3d as o3d
import numpy as np


def extract_coords(path: str) -> list:

    with open(path, "r") as file:
        file_content = file.readlines()

    points = []
    for line in file_content[1:-1]:
        points.append([float(coord) for coord in line.split()])

    return points


def read_pcloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        print("file had extra tokens")
        pcd = extract_coords(path)
    coords_np = np.array(pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_np)
    return pcd


def read_mesh(path):
    #'./data_samples/2022-10-23_01866-000-lowerjaw.ply'
    mesh = o3d.io.read_triangle_mesh(path)
    print(mesh)
    print("Vertices:")
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.vertices))
    print("Triangles:")
    print(np.asarray(mesh.triangles))
    return mesh


def display_point_cloud(pcd: o3d.geometry.PointCloud):
    print(pcd)
    o3d.visualization.draw_geometries([pcd])


def visualize_mesh(mesh):
    print(mesh)
    o3d.visualization.draw_geometries([mesh])


def pcloud_mesh_rollingball(path):  # pcd: o3d.geometry.PointCloud
    # TODO: integrate after finishing testing
    pcd = o3d.io.read_point_cloud(path)
    pcd.estimate_normals()

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 2 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )
    visualize_mesh(mesh)


# TODO:finetune the hyperparams
def pcloud_mesh_poission(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=50)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
    vertices_to_remove = mesh[1] < np.quantile(mesh[1], 0.1)
    mesh[0].remove_vertices_by_mask(vertices_to_remove)
    visualize_mesh(mesh[0])


file_path = "./data_samples/bunny.pcd"
# display_point_cloud(file_path)
pcloud_mesh_poission(path=file_path)
