import pygame



def init():
    pygame.init()
    win = pygame.display.set_mode((400,400))
    

def getKey(keyName):
    result = False;
    for events in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    if keyInput[myKey]:
        result = True
    pygame.display.update()
    return result

def main():
    print(getKey("a"))

if __name__=='__main__':
    init()
    


