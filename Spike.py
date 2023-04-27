import Draw

'''
Subclass of Draw
'''
class Spike(Draw):
    """spike"""
    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)
