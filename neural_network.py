import pygame as pg
from PIL import Image, ImageEnhance
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Users/dakal/AppData/Local/Tesseract-OCR/tesseract.exe"

pg.init()
sc = pg.display.set_mode((400, 300))
WHITE = (255, 255, 255)
BLACK = (0,0,0)
sc.fill(WHITE)
pg.display.update()
f1 = pg.font.Font(None, 36)

clock = pg.time.Clock()
FPS = 60

play = True

if __name__ == '__main__':
    while play:
        for i in pg.event.get():
            if i.type == pg.QUIT:
                pg.quit()
                play = False
            if i.type == pg.MOUSEBUTTONDOWN:
                if i.button == 3:
                    pg.image.save(sc, 'picture.jpg')
                    image = Image.open('picture.jpg')

                    enhancer = ImageEnhance.Contrast(image)
                    img = enhancer.enhance(2)

                    thresh = 200
                    fn = lambda x: 255 if x > thresh else 0
                    res = img.convert('L').point(fn, mode='1')
                    res.save("bw.png", "png")
                    text = pytesseract.image_to_string(res, lang='eng',
                                                       config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                    print(text)

                if i.button == 2:
                    sc.fill(WHITE)
                    pg.display.update()
        if play:
            airbrush = True
            cur = pg.mouse.get_pos()
            click = pg.mouse.get_pressed()
            if click[0] == True:
                pg.draw.circle(sc, BLACK, (cur[0], cur[1]), 10)
            pg.display.update()
            clock.tick(FPS)