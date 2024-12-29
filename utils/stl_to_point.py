import open3d as o3d
import numpy as np


def extract_coords(path: str) -> list:

    with open(path, "r") as file:
        file_content = file.readlines()

    points = []
    for line in file_content[1:-1]:
        points.append([float(coord) for coord in line.split()])

    coords_np = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_np)

    return pcd


def read_xyz(path):

    point_cloud = np.loadtxt(path, delimiter=";", skiprows=1)
    print("here 1")
    print("Loaded point cloud shape:", point_cloud.shape)
    pcd = o3d.geometry.PointCloud()
    print("here 2")
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    print("here 2")
    reflectance = point_cloud[:, 4]
    colors = np.tile(reflectance[:, None], (1, 3)) / np.max(reflectance)
    print("here 2")
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def read_pcloud(path: str) -> o3d.geometry.PointCloud:
    file_ext = path.split(".")
    if file_ext[-1] == "pts":

        print("file had extra tokens")
        pcd = extract_coords(path)

        return pcd
    if file_ext[-1] == "xyz":
        print(path)
        return read_xyz(path)

    pcd = o3d.io.read_point_cloud(path)
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


def visualize_pcloud(pcd: o3d.geometry.PointCloud):
    print(pcd)
    o3d.visualization.draw_geometries([pcd])


def visualize_mesh(mesh):
    print(mesh)
    o3d.visualization.draw_geometries([mesh])


def pcloud_mesh_rollingball(pcd):  # pcd: o3d.geometry.PointCloud
    # TODO: integrate after finishing testing

    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 4.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 4.5])
    )
    dec_mesh = mesh.simplify_quadric_decimation(100000)
    dec_mesh.remove_degenerate_triangles()
    dec_mesh.remove_duplicated_triangles()
    dec_mesh.remove_duplicated_vertices()
    dec_mesh.remove_non_manifold_edges()
    if pcd.has_colors():
        vertex_colors = np.asarray(pcd.colors)
        dec_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    else:
        # Assign red color to all vertices
        red_color = np.array([[1.0, 0.0, 0.0]] * len(dec_mesh.vertices))
        dec_mesh.vertex_colors = o3d.utility.Vector3dVector(red_color)
    return dec_mesh


# TODO:finetune the hyperparams
def pcloud_mesh_poission(pcd):

    pcd.estimate_normals()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, width=0, scale=1.1, linear_fit=False
    )[0]
    # vertices_to_remove = mesh[1] < np.quantile(mesh[1], 0.1)
    # mesh[0].remove_vertices_by_mask(vertices_to_remove)
    dec_mesh = mesh.simplify_quadric_decimation(100000)
    dec_mesh.remove_degenerate_triangles()
    dec_mesh.remove_duplicated_triangles()
    dec_mesh.remove_duplicated_vertices()
    dec_mesh.remove_non_manifold_edges()
    visualize_mesh(dec_mesh)


file_path = "./data_samples/margin_line_ADA4.pts"

pcd = read_pcloud(file_path)
print("here 2")
# visualize_pcloud(pcd=pcd)
mesh = pcloud_mesh_rollingball(pcd=pcd)
visualize_mesh(mesh=mesh)
