from ultralytics import YOLO
import pygame
import time
import cv2

pygame.init()
width, height = 640, 480
screen = pygame.display.set_mode((width, height))

model = YOLO("yolov8n.pt")

model.to("cuda")

target_class_ids = [0]


clock = pygame.time.Clock() 

tick = 0
font = pygame.font.SysFont(None, 50)
tuuka = font.render("通過", True, (255,255,255))
delay = 100
laps = [0]
# WEBカメラからリアルタイム検出
results = model(0 , show=False, stream = True)
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

    if x1 < width/2 and x2 > width/2 and delay > 30:
        delay = 0
        laps.append(time)
        screen.blit(tuuka,(100,0))

    text = font.render(f"{tick}", True, (255,255,255))
    screen.blit(text, (0, 0))

    lap_txt = font.render(f"{laps[-1]}", True, (255,255,255))
    screen.blit(lap_txt,(0,200))
    pygame.draw.line(screen, (0,95,0), (width/2,0), (width/2,height), 5)
    pygame.display.flip()  # 画面を更新

    tick += 1
    delay += 1
    
    clock.tick(10)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

#pytorchでメモリを32GB=>8GB
#https://yolov8-objectdetection.glitch.me/
# でWEB動作
#flepみたいなやつでpythonでスマホのローカルで動作