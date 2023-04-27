import Draw

'''
Subclass of Draw
'''

class Platform(Draw):
    """block"""
    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)
