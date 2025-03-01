from ultralytics import YOLO
import pygame
import time
import cv2

pygame.init()

class button():
    def __init__(self,x,y,width,height,color,text):
        self.rect =pygame.Rect(x,y,width,height)
        self.text = text
        self.color = color
        self.font = pygame.font.SysFont(None, 50)
        self.text = text
        self.text_pos = (x,y)
        self.latest_mouseclick = False

    def draw(self,screen):
        pygame.draw.rect(screen,self.color,self.rect)
        screen.blit(self.font.render(f"{self.text}", True, (255,255,255)), self.text_pos)
    def click(self):
        mouseClick = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        if mouseClick[0] and not self.latest_mouseclick:
            if self.rect.collidepoint(mouse_pos):
                self.latest_mouseclick = mouseClick[0]
                return True
        self.latest_mouseclick = mouseClick[0]
        

def main():
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))

    model = YOLO("yolov8n.pt")

    model.to("cuda")

    target_class_ids = [0]


    clock = pygame.time.Clock() 

    tick = 0
    font = pygame.font.SysFont(None, 50)
    tuuka = font.render("tuuka", True, (255,255,255))
    start = font.render("start", True, (255,255,255))
    stop = font.render("stop", True, (255,255,255))
    delay = 100
    laps = [0]
    x1=0
    y1=0
    y2=0
    x2=0
    isrunning = False
    # WEBカメラからリアルタイム検出
    results = model(0 , show=False, stream = True)
    
    start = button(380,0,100,40,(0,255,0),"start")
    stop = button(380,0,100,40,(255,0,0),"stop")
    
    for result in results:

        frame = result.orig_img

        for detection in result.boxes:
            class_id = detection.cls  # クラスID
            confidence = detection.conf  # 信頼度
            if class_id in target_class_ids:
                pos = detection.xyxy[0]  # 座標
                print(f"座標: {pos}")
                x1, y1, x2, y2 = map(int, pos)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 青色の四角

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 色を変換
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = pygame.surfarray.make_surface(frame)

        screen.blit(frame, (0, 0))

        if isrunning:
            if x1 < width/2 and x2 > width/2 and delay > 30:
                delay = 0
                laps.append(tick)
                screen.blit(tuuka,(100,0))
            stop.draw(screen)
            if stop.click():
                isrunning = False
            tick += 1
        else:
            start.draw(screen)
            if start.click():
                isrunning = True
                tick = 0
                laps = [0]

        text = font.render(f"{tick/10}s", True, (255,255,255))
        screen.blit(text, (0, 0))

        lap_txt = font.render(f"{laps[-1]/10}s", True, (255,255,255))
        screen.blit(lap_txt,(0,200))
        pygame.draw.line(screen, (0,95,0), (width/2,0), (width/2,height), 5)
        pygame.display.flip()  # 画面を更新

        delay += 1
        
        clock.tick(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
if __name__ == '__main__':
    main()
#pytorchでメモリを32GB=>8GB
#https://yolov8-objectdetection.glitch.me/
# でWEB動作
#fletみたいなやつでpythonでスマホのローカルで動作
#11.stopsine使う