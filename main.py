#  filename: main.py
#  author: Yonah Aviv
#  date created: 2020-11-10 6:21 p.m.
#  last modified: 2020-11-18
#  Pydash: Similar to Geometry Dash, a rhythm based platform game, but programmed using the pygame library in Python


"""CONTROLS
Anywhere -> ESC: exit
Main menu -> 1: go to previous level. 2: go to next level. SPACE: start game.
Game -> SPACE/UP: jump, and activate orb
    orb: jump in midair when activated
If you die or beat the level, press SPACE to restart or go to the next level

"""

import csv
import os
import random
import math

# import the pygame module
import pygame

# will make it easier to use pygame functions
from pygame.math import Vector2
from pygame.draw import rect

# initializes the pygame module
pygame.init()

# creates a screen variable of size 800 x 600
screen = pygame.display.set_mode([800, 600])

# controls whether or not to start the game from the main menu
# sets the frame rate of the program
clock = pygame.time.Clock()

global done, start
# DEBUG
print("start initialized here")
start = False

# DEBUG
print(start)
done = False

attempts = 0

"""
CONSTANTS
"""
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


# Parent class
class Draw(pygame.sprite.Sprite):
    """parent class to all obstacle classes; Sprite class"""

    def __init__(self, image, pos, *groups):
        super().__init__(*groups)
        self.image = image
        self.pos = pos
        self.rect = self.image.get_rect(topleft=pos)

#  ====================================================================================================================#
#  classes of all obstacles. this may seem repetitive but it is useful(to my knowledge)
#  ====================================================================================================================#
# children
class Platform(Draw):
    """block"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class Spike(Draw):
    """spike"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)



class Coin(Draw):
    """coin. get 6 and you win the game"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class Orb(Draw):
    """orb. click space or up arrow while on it to jump in midair"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class Trick(Draw):
    """block, but its a trick because you can go through it"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class End(Draw):
    "place this at the end of the level"

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


"""
Main player class
"""

class Player(pygame.sprite.Sprite):
    """Class for player. Holds update method, win and die variables, collisions and more."""
    win: bool
    died: bool

    def __init__(self, image, platforms, pos, *groups):

        """
        :param image: block face avatar
        :param platforms: obstacles such as coins, blocks, spikes, and orbs
        :param pos: starting position
        :param groups: takes any number of sprite groups.
        """
        # DEBUG
        print("in Player init function")

        super().__init__(*groups)

        # DEBUG
        print("test")

        self.onGround = False  # player on ground?
        self.platforms = platforms  # obstacles but create a class variable for it
        self.died = False  # player died?
        self.win = False  # player beat level?

        self.image = pygame.transform.smoothscale(image, (32, 32))
        self.rect = self.image.get_rect(center=pos)  # get rect gets a Rect object from the image
        self.x_pos = math.floor(self.rect.left / 32)
        self.pos_x = pos[0]
        self.jump_amount = 10  # jump strength
        self.particles = []  # player trail
        self.isjump = False  # is the player jumping?
        self.vel = Vector2(0, 0)  # velocity starts at zero
        self.collided = False

    def draw_particle_trail(self, x, y, color=(255, 255, 255)):
        """draws a trail of particle-rects in a line at random positions behind the player"""

        self.particles.append(
            [[x - 5, y - 8], [random.randint(0, 25) / 10 - 1, random.choice([0, 0])],
             random.randint(5, 8)])

        for particle in self.particles:
            particle[0][0] += particle[1][0]
            particle[0][1] += particle[1][1]
            particle[2] -= 0.5
            particle[1][0] -= 0.4
            rect(alpha_surf, color,
                 ([int(particle[0][0]), int(particle[0][1])], [int(particle[2]) for i in range(2)]))
            if particle[2] <= 0:
                self.particles.remove(particle)

    def reward(self):
        new_x_pos = math.floor(self.rect.left / 32)

        if (self.died):
            return -10
        else:
            # DEBUG
            #print("Got farther than last record! Current new_x_pos: ", new_x_pos, "\n")
            #print("Old x_pos: ", self.pos_x)
            # self.pos_x = new_x_pos
            #print("New x_pos: ", self.pos_x)
            return 10

    def collide(self, yvel, platforms):
        global coins, attempts

        self.canJump = False

        # DEBUG
        print("Begin collide(): Player dead ? ", self.died)
        print("Collide(): Collided? ", self.collided)

        for p in platforms:
            if pygame.sprite.collide_rect(self, p):
                """pygame sprite builtin collision method,
                sees if player is colliding with any obstacles"""

                self.collided = True
                #DEBUG
                print("Collide(): Inside if statement...Collided? ", self.collided)

                if isinstance(p, Orb):
                    pygame.draw.circle(alpha_surf, (255, 255, 0), p.rect.center, 18)
                    screen.blit(pygame.image.load("images/editor-0.9s-47px.gif"), p.rect.center)
                    self.jump_amount = 12  # gives a little boost when hit orb
                    self.canJump = True
                    self.jump()
                    self.jump_amount = 10  # return jump_amount to normal

                elif isinstance(p, End):
                    self.win = True

                elif isinstance(p, Spike):
                    self.died = True  # die on spike

                    #DEBUG
                    print("Collided with Spike. Player dead? ", self.died)

                    attempts += 1

                    #reset()

                    # DEBUG
                    #print("Player position when died: ", player.x_pos)

                elif isinstance(p, Coin):
                    # keeps track of all coins throughout the whole game(total of 6 is possible)
                    coins += 1

                    # erases a coin
                    p.rect.x = 0
                    p.rect.y = 0

                elif isinstance(p, Platform):  # these are the blocks (may be confusing due to self.platforms)

                    if yvel > 0:
                        """if player is going down(yvel is +)"""
                        self.rect.bottom = p.rect.top  # dont let the player go through the ground
                        self.vel.y = 0  # rest y velocity because player is on ground

                        # set self.onGround to true because player collided with the ground
                        self.onGround = True

                        # reset jump
                        self.isjump = False

                    elif yvel < 0:
                        """if yvel is (-),player collided while jumping"""
                        self.rect.top = p.rect.bottom  # player top is set the bottom of block like it hits it head

                        # DEBUG
                        print("Collided with Platform while jumping. Player dead? ", self.died)
                    else:
                        """otherwise, if player collides with a block, he/she dies."""
                        self.vel.x = 0
                        self.rect.right = p.rect.left  # dont let player go through walls
                        self.died = True
                        attempts += 1

                        # DEBUG
                        #print("Player position when died: ", player.x_pos)

                        # DEBUG
                        print("Collided with Platform/Block. Player dead? ", self.died)

                        reset()


    def jump(self):
        # DEBUG
        print("in jump function rn")
        self.vel.y = -self.jump_amount  # players vertical velocity is negative so ^

    def move_player(self):
        self.pos_x += self.vel.x

    def update(self, final_move):
        global start, angle
        # DEBUG
        print("updating")
        keys = pygame.key.get_pressed()
        # start = false at the beginning of the game so that the title screen shows up. Pushing should start the game.
        if not start:
            wait_for_key()
            reset()
        # start = true because the game has begun. also prevents the game from showing the title screen again
        start = True

        """Move player"""
        self._move(final_move)
        self._update_ui()

        # DEBUG - Trying this out to stop the infinite loop from agent:train() - SW
        # self.died = True
        # print("Final move value:")
        # print(final_move[0])

        """check if game over """

        # do x-axis collisions
        #self.collide(0, self.platforms)
        self.collide(self.vel.x, self.platforms)

        # increment.update in y direction
        self.rect.top += self.vel.y

        # assuming player in the air, and if not it will be set to inversed after collide
        self.onGround = False

        # do y-axis collisions
        self.collide(self.vel.y, self.platforms)

        self._update_ui()

        if not self.onGround:  # only accelerate with gravity if in the air
            self.vel += GRAVITY  # Gravity falls

            # max falling speed
            if self.vel.y > 100: self.vel.y = 100

        # check if we won or if player won
        # all this function does is show the win and die screens, don't need it
        # eval_outcome(self.win, self.died)

        reward = self.reward()

        # DEBUG
        print("reward is: ", reward)

        """update ui and clock"""
        pygame.display.update()
        pygame.display.flip()
        clock.tick(60)

        # DEBUG
        print("end of update")

        # DEBUG
        #print("velocity is:", self.vel.x)

        # Self.died or self.win will determine if we are done (for agent.train() loop)
        return reward, (self.died or self.win), self.rect.left

    def _move(self, final_move):
        global angle

        # Trying to change x_pos
        self.move_player()

        if final_move[0] == 1:
            self.isjump = True

        if self.isjump:
            # removed line from if statement: or self.canJump
            if self.onGround:
                """if player wants to jump and player is on the ground: only then is jump allowed"""
                self.jump()

            """rotate the player by an angle and blit it if player is jumping"""
            angle -= 8.1712  # this may be the angle needed to do a 360 deg turn in the length covered in one jump by player
            blitRotate(screen, self.image, self.rect.center, (16, 16), angle)
        else:
            """if player.isjump is false, then just blit it normally(by using Group().draw() for sprites"""
            player_sprite.draw(screen)  # draw player sprite group
            # DEBUG
            print("Player is false")

    def _update_ui(self):
        # map, player movement update
        # velocity the player moves at throughout the game
        # print(self.vel.x)
        self.vel.x = 6
        # Reduce the alpha of all pixels on this surface each frame.
        # Control the fade2 speed with the alpha value.
        # draw background
        alpha_surf.fill((255, 255, 255, 1), special_flags=pygame.BLEND_RGBA_MULT)
        # apply player speed to camera
        CameraX = self.vel.x  # for moving obstacles
        move_map()  # apply CameraX to all elements
        screen.blit(bg, (0, 0))  # Clear the screen(with the bg)

        self.draw_particle_trail(self.rect.left - 1, self.rect.bottom + 2,
                                 WHITE)
        screen.blit(alpha_surf, (0, 0))  # Blit the alpha_surf onto the screen.
        draw_stats(screen, coin_count(coins))

        self.draw_elements(screen)  # draw all other obstacles
        self.draw_player(screen)
        #player_sprite.draw(screen)

    def draw_player(self, screen):
        player_sprite.draw(screen)

    def draw_elements(self, screen):
        elements.draw(screen)

    def check_if_dead(self) -> bool:
        return self.died

"""
Functions
"""


def init_level(map):
    """this is similar to 2d lists. it goes through a list of lists, and creates instances of certain obstacles
    depending on the item in the list"""
    x = 0
    y = 0

    for row in map:
        for col in row:

            if col == "0":
                Platform(block, (x, y), elements)

            if col == "Coin":
                Coin(coin, (x, y), elements)

            if col == "3":
                Spike(spike, (x, y), elements)
            if col == "Orb":
                orbs.append([x, y])

                Orb(orb, (x, y), elements)

            if col == "T":
                Trick(trick, (x, y), elements)

            if col == "End":
                End(avatar, (x, y), elements)
            x += 32
        y += 32
        x = 0


def blitRotate(surf, image, pos, originpos: tuple, angle: float):
    """
    rotate the player
    :param surf: Surface
    :param image: image to rotate
    :param pos: position of image
    :param originpos: x, y of the origin to rotate about
    :param angle: angle to rotate
    """
    # calcaulate the axis aligned bounding box of the rotated image
    w, h = image.get_size()
    box = [Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]

    # make sure the player does not overlap, uses a few lambda functions(new things that we did not learn about number1)
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
    # calculate the translation of the pivot
    pivot = Vector2(originpos[0], -originpos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move = pivot_rotate - pivot

    # calculate the upper left origin of the rotated image
    origin = (pos[0] - originpos[0] + min_box[0] - pivot_move[0], pos[1] - originpos[1] - max_box[1] + pivot_move[1])

    # get a rotated image
    rotated_image = pygame.transform.rotozoom(image, angle, 1)

    # rotate and blit the image
    surf.blit(rotated_image, origin)


def won_screen():
    """show this screen when beating a level"""
    #global attempts, level, fill
    global attempts, level, fill
    attempts = 0
    player_sprite.clear(player.image, screen)
    screen.fill(pygame.Color("yellow"))
    txt_win1 = txt_win2 = "Nothing"
    if level == 1:
        if coins == 6:
            txt_win1 = f"Coin{coins}/6! "
            txt_win2 = "the game, Congratulations"
    else:
        txt_win1 = f"level{level}"
        txt_win2 = f"Coins: {coins}/6. "
    txt_win = f"{txt_win1} You beat {txt_win2}! Press SPACE to restart, or ESC to exit"

    won_game = font.render(txt_win, True, BLUE)

    screen.blit(won_game, (200, 300))
    level += 1

    wait_for_key()
    reset()


def death_screen():
    """show this screenon death"""
    global attempts, fill
    fill = 0
    player_sprite.clear(player.image, screen)
    attempts += 1
    game_over = font.render("Game Over. [SPACE] to restart", True, WHITE)

    screen.fill(pygame.Color("sienna1"))
    screen.blits([[game_over, (100, 100)], [tip, (100, 400)]])

    wait_for_key()
    reset()



def eval_outcome(won: bool, died: bool):
    """simple function to run the win or die screen after checking won or died"""
    if won:
        won_screen()
    if died:
        death_screen()

    #return

def block_map(level_num):
    """
    :type level_num: rect(screen, BLACK, (0, 0, 32, 32))
    open a csv file that contains the right level map
    """
    lvl = []
    with open(level_num, newline='') as csvfile:
        trash = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in trash:
            lvl.append(row)
    global platformList
    platformList = lvl
    return lvl


def start_screen():
    """main menu. option to switch level, and controls guide, and game overview."""
    global level, start
    if not start:
        # DEBUG
        # print("screen filled black")
        screen.fill(BLACK)
        if pygame.key.get_pressed()[pygame.K_1]:
            level = 0
        if pygame.key.get_pressed()[pygame.K_2]:
            level = 1

        welcome = font.render(f"Welcome to Pydash. choose level({level + 1}) by keypad", True, WHITE)

        controls = font.render("Controls: jump: Space/Up exit: Esc", True, GREEN)

        screen.blits([[welcome, (100, 100)], [controls, (100, 400)], [tip, (100, 500)]])

        level_memo = font.render(f"Level {level + 1}.", True, (255, 255, 0))
        screen.blit(level_memo, (100, 200))


def reset():
    """resets the sprite groups, music, etc. for death and new level"""
    global player, elements, player_sprite, level

    # DEBUG
    player.died = False
    print("In reset(): Player Dead? ", player.died)

    # DEBUG
    print("in reset function here")

    if level == 1:
        #DEBUG
        print("level 1 music playing")
        #pygame.mixer.music.load(os.path.join("music", "castle-town.mp3"))
    #pygame.mixer_music.play()
    player_sprite = pygame.sprite.Group()
    elements = pygame.sprite.Group()
    player = Player(avatar, elements, (100, 150), player_sprite)

    # DEBUG
    print("Player initialized here")

    init_level(
            block_map(
                    level_num=levels[level]))


def move_map():
    CameraX = 5
    """moves obstacles along the screen"""
    for sprite in elements:
        sprite.rect.x -= CameraX
        #player.x_pos -= CameraX


        # DEBUG
        #print("Player x position is: ", player.x_pos)


def draw_stats(surf, money=0):
    """
    draws progress bar for level, number of attempts, displays coins collected, and progressively changes progress bar
    colors
    """
    global fill
    progress_colors = [pygame.Color("red"), pygame.Color("orange"), pygame.Color("yellow"), pygame.Color("lightgreen"),
                       pygame.Color("green")]

    tries = font.render(f" Attempt {str(attempts)}", True, WHITE)
    BAR_LENGTH = 600
    BAR_HEIGHT = 10
    for i in range(1, money):
        screen.blit(coin, (BAR_LENGTH, 25))
    if player.check_if_dead():
        fill = 0
    else:
        fill += 0.5
    outline_rect = pygame.Rect(0, 0, BAR_LENGTH, BAR_HEIGHT)
    fill_rect = pygame.Rect(0, 0, fill, BAR_HEIGHT)
    col = progress_colors[int(fill / 100)]
    rect(surf, col, fill_rect, 0, 4)
    rect(surf, WHITE, outline_rect, 3, 4)
    screen.blit(tries, (BAR_LENGTH, 0))


def wait_for_key():
    """separate game loop for waiting for a key press while still running game loop
    """
    global level, start
    waiting = True
    while waiting:
        #clock.tick(60)
        pygame.display.flip()

        if not start:
            start_screen()

        for event in pygame.event.get():
            #DEBUG
            print("Entered event for loop")
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    start = True
                    waiting = False
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()


def coin_count(coins):
    """counts coins"""
    if coins >= 3:
        coins = 3
    coins += 1
    return coins


def resize(img, size=(32, 32)):
    """resize images
    :param img: image to resize
    :type img: im not sure, probably an object
    :param size: default is 32 because that is the tile size
    :type size: tuple
    :return: resized img

    :rtype:object?
    """
    resized = pygame.transform.smoothscale(img, size)
    return resized


"""
Global variables
"""
#DEBUG
print("global vars assigned")
font = pygame.font.SysFont("lucidaconsole", 20)

# square block face is main character the icon of the window is the block face
avatar = pygame.image.load(os.path.join("images", "avatar.png"))  # load the main character
pygame.display.set_icon(avatar)
#  this surface has an alpha value with the colors, so the player trail will fade away using opacity
alpha_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

# sprite groups
player_sprite = pygame.sprite.Group()
elements = pygame.sprite.Group()

player = Player(avatar, elements, (145, 150), player_sprite)
# images
spike = pygame.image.load(os.path.join("images", "obj-spike.png"))
spike = resize(spike)
coin = pygame.image.load(os.path.join("images", "coin.png"))
coin = pygame.transform.smoothscale(coin, (32, 32))
block = pygame.image.load(os.path.join("images", "block_1.png"))
block = pygame.transform.smoothscale(block, (32, 32))
orb = pygame.image.load((os.path.join("images", "orb-yellow.png")))
orb = pygame.transform.smoothscale(orb, (32, 32))
trick = pygame.image.load((os.path.join("images", "obj-breakable.png")))
trick = pygame.transform.smoothscale(trick, (32, 32))

#  ints
fill = 0
num = 0
CameraX = 0
coins = 0
global angle
angle = 0
# DEBUG

level = 0

# list
particles = []
orbs = []
win_cubes = []

# initialize level with
levels = ["level_1.csv", "level_2.csv"]
level_list = block_map(levels[level])
level_width = (len(level_list[0]) * 32)
level_height = len(level_list) * 32
init_level(level_list)

# set window title suitable for game
pygame.display.set_caption('Pydash: Geometry Dash in Python')

# initialize the font variable to draw text later
text = font.render('image', False, (255, 255, 0))

# music
music = pygame.mixer_music.load(os.path.join("music", "bossfight-Vextron.mp3"))
#pygame.mixer_music.play()

# bg image
bg = pygame.image.load(os.path.join("images", "bg.png"))

# create object of player class


# show tip on start and on death
tip = font.render("tip: tap and hold for the first few seconds of the level", True, BLUE)



"""lambda functions are anonymous functions that you can assign to a variable.
e.g.
1. x = lambda x: x + 2  # takes a parameter x and adds 2 to it
2. print(x(4))
>>6
"""
color = lambda: tuple([random.randint(0, 255) for i in range(3)])  # lambda function for random color, not a constant.
GRAVITY = Vector2(0, 0.86)  # Vector2 is a pygame


"""
Obstacle classes
"""








#player = Player(avatar, elements, (150, 150), player_sprite)



