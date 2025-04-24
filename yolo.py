from ultralytics import YOLO
import pygame
import time
import cv2

pygame.init()
#startなどボタンのクラス
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
    width, height = 800, 540
    screen = pygame.display.set_mode((width, height))

    model = YOLO("yolov8n.pt")

    model.to("cuda")

    target=0

    target_class_ids = [target]


    clock = pygame.time.Clock() 

    tick = 0
    font = pygame.font.SysFont(None, 50)
    tuuka = font.render("tuuka", True, (255,255,255))
    start = font.render("start", True, (255,255,255))
    stop = font.render("stop", True, (255,255,255))
    background =pygame.Rect(0,0,width,height)
    delay = 100
    laps = []
    x1=0
    y1=0
    y2=0
    x2=0
    line_color=(0,0,0)
    isrunning = False
    ready = False

    # WEBカメラからリアルタイム検出
    results = model(0 , show=False, stream = True,classes = [target])#11 stopsign
    
    start = button(5,5,100,40,(0,255,0),"start")
    stop = button(5,5,100,40,(255,0,0),"stop")
    
    for result in results:

        #OpenCV部分
        frame = result.orig_img

        for detection in result.boxes:
            class_id = detection.cls  # クラスID
            confidence = detection.conf  # 信頼度
            print(class_id)
            if class_id in target_class_ids:
                #print(result.boxes)
                pos = detection.xyxy[0]  # 座標
                print(f"座標: {pos}")
                x1, y1, x2, y2 = map(int, pos)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 青色の四角

        #Pygameに変換・描画
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 色を変換
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = pygame.surfarray.make_surface(frame)

        screen.fill((255,255,255), background)
        screen.blit(frame, (0, height-480))

        #通過判定、スタートボタン表示
        pygame.draw.line(screen, line_color, (640/2,0), (640/2,height), 5)
        if ready:
            line_color = (0,255,255)
            if x1 < 640/2 and x2 > 640/2:
                isrunning = True
                ready = False
        elif isrunning:
            line_color = (0,255,0)
            if x1 < 640/2 and x2 > 640/2 and delay > 30:
                delay = 0
                laps.append(tick)
                screen.blit(tuuka,(100,0))
            stop.draw(screen)
            if stop.click():
                isrunning = False
            tick += 1
        else:
            line_color = (255,0,0)
            start.draw(screen)
            if start.click():
                ready = True
                tick = 0
                laps = []

        #現在タイム、ラップ表示
        text = font.render(f"{tick/10}s", True, (0,0,0))
        screen.blit(text, (640, 5))

        if 0 in laps:
            lap_txt = font.render(f"{laps[-1]/10}s", True, (255,255,255))
            for lap in laps:
                if laps.index(lap)==0:
                    continue
                lap_txt = font.render(f"{lap/10}s /{laps.index(lap)}", True, (0,0,0))
                screen.blit(lap_txt,(640,10+laps.index(lap)*30))
        pygame.display.flip()  # 画面を更新
        print(laps)

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