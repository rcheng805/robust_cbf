import pygame

pygame.init()


def make_video(screen, iteration, type):
    _image_num = 0

    while True:
        _image_num += 1
        str_num = "000" + str(_image_num)
        file_name = str(iteration) + "_image_" + str_num[-4:] + str(type) + ".jpg"
        pygame.image.save(screen, file_name)
        yield
