import Draw

'''
Subclass of Draw
'''

class Coin(Draw):
    """coin. get 6 and you win the game"""
    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)
