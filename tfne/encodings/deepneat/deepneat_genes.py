class DeepNEATNode:
    """"""

    def __init__(self, gene_id, node, layer):
        self.gene_id = gene_id
        self.node = node
        self.layer = layer


class DeepNEATConn:
    """"""

    def __init__(self, gene_id, conn_start, conn_end, enabled=True):
        self.gene_id = gene_id
        self.conn_start = conn_start
        self.conn_end = conn_end
        self.enabled = enabled

    def set_enabled(self, enabled):
        """"""
        self.enabled = enabled
