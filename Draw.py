from main import pygame

'''
Parent class for all obstacle classes (Orb, Platform, Spike, Coin, Trick, End); Sprite class
'''

class Draw(pygame.sprite.Sprite):
    def __init__(self, image, pos, *groups):
        super().__init__(*groups)
        self.image = image
        self.rect = self.image.get_rect(topleft=pos)
