from Draw import Draw

'''
Subclass of Draw
'''

class End(Draw):
    "place this at the end of the level"

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)
