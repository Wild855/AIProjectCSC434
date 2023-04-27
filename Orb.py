import Draw

'''
Subclass of Draw
'''

class Orb(Draw):
    """orb. click space or up arrow while on it to jump in midair"""
    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)
