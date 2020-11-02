
'''
Flappy Bird
'''
import pygame
from pygame import mixer
import random
import os
import time
from setting import *
import neat

class Bird:
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("assets","bird" + str(x) + ".png"))) for x in range(1,4)]
        self.IMGS = self.bird_images
        self.tilt = 0  # degrees to tilt
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        # for downward acceleration
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2  # calculate displacement

        # terminal velocity
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:  # tilt up
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:  # tilt down
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        # For animation of bird, loop through three images
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        # so when bird is nose diving it isn't flapping
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        # tilt the bird
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe():
    GAP = 200
    VEL = 5

    def __init__(self, x,img):
        self.x = x
        self.height = 0
        self.pipe_img = img

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = pygame.transform.flip(self.pipe_img, False, True)
        self.PIPE_BOTTOM = self.pipe_img

        self.passed = False

        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        # draw top
        win.blit(self.PIPE_TOP, (self.x, self.top))
        # draw bottom
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)

        if b_point or t_point:
            return True

        return False

class Base:
    VEL = 5

    def __init__(self, y,img):
        self.y = y
        self.base_img = img
        self.IMG = self.base_img        
        self.WIDTH = self.base_img.get_width()

        self.x1 = 0
        self.x2 = self.WIDTH
        
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)

    surf.blit(rotated_image, new_rect.topleft)

global gen
gen = 0

class GameClass:

    def __init__(self):
        pygame.init()
        pygame.font.init()
        pygame.mixer.init()

        self.screen = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption('AI Learns to Play Flappy Bird')

        pygame.mixer.music.load(os.path.join('assets','music.mp3'))
        pygame.mixer.music.play(-1)
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.SysFont("comicsans", 50,False,True)

        self.pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("assets","pipe.png")).convert_alpha())
        self.bg_img = pygame.transform.scale(pygame.image.load(os.path.join("assets","bg.png")).convert_alpha(), (600, 900))
        self.base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("assets","base.png")).convert_alpha())
        
        self.run = True

    def gameloop(self,genomes, config):
        '''
        runs the simulation of the current population of
        birds and sets their fitness based on the distance they
        reach in the game.
        '''
        global gen
        gen += 1

        # start by creating lists holding the genome itself, the
        # neural network associated with the genome and the
        # bird object that uses that network to play
        nets = []
        birds = []
        ge = []
        for genome_id, genome in genomes:
            genome.fitness = 0  # start with fitness level of 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            birds.append(Bird(230,350))
            ge.append(genome)
     
        base = Base(FLOOR,self.base_img)
        pipes = [Pipe(700,self.pipe_img)]
        score = 0
        
        while self.run and len(birds) > 0:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.run = False
                    pygame.quit()
                    quit()

            pipe_ind = 0
            if len(birds) > 0:
                if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  # determine whether to use the first or second
                    pipe_ind = 1                                                                 # pipe on the screen for neural network input

            for x, bird in enumerate(birds):  # give each bird a fitness of 0.1 for each frame it stays alive
                ge[x].fitness += 0.1
                bird.move()

                # send bird location, top pipe location and bottom pipe location and determine from network whether to jump or not
                output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

                if output[0] > 0.5:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
                    bird.jump()

            base.move()

            rem = []
            add_pipe = False
            for pipe in pipes:
                pipe.move()
                # check for collision
                for bird in birds:
                    if pipe.collide(bird, self.screen):
                        ge[birds.index(bird)].fitness -= 1
                        nets.pop(birds.index(bird))
                        ge.pop(birds.index(bird))
                        birds.pop(birds.index(bird))

                if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                    rem.append(pipe)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if add_pipe:
                score += 1
                # print(score)
                # can add this line to give more reward for passing through a pipe (not required)
                for genome in ge:
                    genome.fitness += 5
                pipes.append(Pipe(WIDTH,self.pipe_img))

            for r in rem:
                pipes.remove(r)

            for bird in birds:
                if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            self.draw(self.screen, birds, pipes, base, score, gen, pipe_ind)

    def draw(self,win, birds, pipes, base, score, gen, pipe_ind):
        if gen == 0:
            gen = 1
        win.blit(self.bg_img, (0,0))

        for pipe in pipes:
            pipe.draw(win)

        base.draw(win)
        for bird in birds:
            # draw lines from bird to pipe
            if DRAW_LINES:
                try:
                    pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                    pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
                except:
                    pass
            # draw bird
            bird.draw(win)

        # score
        score_label = self.font.render(str(score),1,(0,0,0))
        # win.blit(score_label, (WIDTH - score_label.get_width() - 15, 10))
        win.blit(score_label, (WIDTH//2,10))

        # generations
        score_label = self.font.render('Gen: ' + str(gen-1),1,(255,255,255))
        win.blit(score_label, (10, 10))

        # alive
        score_label = self.font.render("Alive: " + str(len(birds)),1,(255,255,255))
        win.blit(score_label, (WIDTH-150, 10))

        pygame.display.update()
