"""
Computes Loss map
"""


class LossMap:
    def __init__(self, labels):
        self.labels = labels
        self.mean = labels.mean()

    def _closet_cell(self, i, j):
        """
        This function is define as d1 in the U-Net paper
        """
        raise NotImplementedError("This function should return the closet cell")

    def _second_closet_cell(self, i, j):
        """
        This function is define as d2 in the U-Net paper
        """
        raise NotImplementedError("This function should return"
                                  "the second closet cell")

    def _pixel_weight(self, i, j):
        """
        This function is define as w in the U-Net paper
        """
        raise NotImplementedError("This function should return the weight of the pixel")
