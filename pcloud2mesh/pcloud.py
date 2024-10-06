from ..utils import stl_to_point as s2p


class pcloud_figure:
    def __init__(self, path):
        self.path = path
        self.mesh = None
        self.pcloud = None

    def visualize():
        s2p.visualize_pcloud()

    def get_pcloud(self):
        self.pcloud = s2p.read_pcloud(self.path)

    def to_mesh_ball(self):
        self.mesh = s2p.pcloud_mesh_rollingball(self.pcloud)
