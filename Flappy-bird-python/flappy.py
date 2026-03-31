import os
from pathlib import Path
import random
import threading
import time
import urllib.request

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import pygame
from pygame.locals import K_SPACE, K_UP, KEYDOWN, QUIT

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    cv2 = None
    mp = None
    mp_python = None
    mp_vision = None


BASE_DIR = Path(__file__).resolve().parent
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
)
GESTURE_MODEL_PATH = BASE_DIR / "assets" / "models" / "gesture_recognizer.task"


def asset_path(*parts):
    return str(BASE_DIR.joinpath("assets", *parts))


def ensure_gesture_model():
    GESTURE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not GESTURE_MODEL_PATH.exists():
        urllib.request.urlretrieve(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)
    return str(GESTURE_MODEL_PATH)


SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 15

SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 15

GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100

PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.images = [
            pygame.image.load(asset_path("sprites", "bluebird-upflap.png")).convert_alpha(),
            pygame.image.load(asset_path("sprites", "bluebird-midflap.png")).convert_alpha(),
            pygame.image.load(asset_path("sprites", "bluebird-downflap.png")).convert_alpha(),
        ]
        self.speed = SPEED
        self.current_image = 0
        self.image = self.images[self.current_image]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH // 6
        self.rect.y = SCREEN_HEIGHT // 2

    def update(self):
        self.current_image = (self.current_image + 1) % len(self.images)
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect.y += int(self.speed)

    def bump(self):
        self.speed = -SPEED

    def begin(self):
        self.current_image = (self.current_image + 1) % len(self.images)
        self.image = self.images[self.current_image]


class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        super().__init__()
        self.inverted = inverted
        self.passed = False
        self.image = pygame.image.load(asset_path("sprites", "pipe-green.png")).convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect.x = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.y = -(self.rect.height - ysize)
        else:
            self.rect.y = SCREEN_HEIGHT - ysize

        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect.x -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        super().__init__()
        self.image = pygame.image.load(asset_path("sprites", "base.png")).convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.x = xpos
        self.rect.y = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect.x -= GAME_SPEED


class ThumbController:
    WINDOW_NAME = "Thumb Control"
    JUMP_COOLDOWN = 0.8

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.running = False
        self.jump_requested = False
        self.status = f"Dang khoi dong camera {camera_index}..."
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        if cv2 is None or mp is None or mp_python is None or mp_vision is None:
            self.status = "Thieu OpenCV/MediaPipe"
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    def consume_jump(self):
        with self.lock:
            if self.jump_requested:
                self.jump_requested = False
                return True
        return False

    def _capture_loop(self):
        capture = cv2.VideoCapture(self.camera_index)
        recognizer = None

        try:
            self.status = f"Dang mo camera {self.camera_index}..."
            if not capture.isOpened():
                self.running = False
                return

            model_path = ensure_gesture_model()
            options = mp_vision.GestureRecognizerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            recognizer = mp_vision.GestureRecognizer.create_from_options(options)

            last_jump = 0.0

            while self.running:
                ok, frame = capture.read()
                if not ok:
                    self.status = "Khong doc duoc frame camera"
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = recognizer.recognize(mp_image)

                thumbs_up = self._contains_thumbs_up(result)
                self._draw_landmarks(frame, result.hand_landmarks)

                now = time.time()
                if thumbs_up and now - last_jump >= self.JUMP_COOLDOWN:
                    with self.lock:
                        self.jump_requested = True
                    last_jump = now
                elif not thumbs_up:
                    self.status = "Dang nhan dien..."

                cv2.putText(
                    frame,
                    self.status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(self.WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False
                    break
        except Exception as exc:
            self.status = f"Camera loi: {type(exc).__name__}, dung phim Space"
            self.running = False
        finally:
            if recognizer is not None:
                recognizer.close()
            capture.release()
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except cv2.error:
                pass

    @staticmethod
    def _contains_thumbs_up(result):
        for hand_gestures in result.gestures:
            if hand_gestures and hand_gestures[0].category_name == "Thumb_Up":
                return True
        return False

    @staticmethod
    def _draw_landmarks(frame, hand_landmarks):
        for landmarks in hand_landmarks:
            points = []
            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                points.append((x, y))
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

            for start, end in (
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17),
            ):
                cv2.line(frame, points[start], points[end], (255, 0, 255), 2)


def is_off_screen(sprite):
    return sprite.rect.right < 0


def get_random_pipes(xpos):
    size = random.randint(100, 300)
    bottom_pipe = Pipe(False, xpos, size)
    top_pipe = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return bottom_pipe, top_pipe


def draw_text(screen, font, text, position, color=WHITE):
    shadow = font.render(text, True, BLACK)
    label = font.render(text, True, color)
    screen.blit(shadow, (position[0] + 2, position[1] + 2))
    screen.blit(label, position)


def draw_scene(
    screen,
    background,
    begin_image,
    bird_group,
    pipe_group,
    ground_group,
    hud_font,
    started,
    center_message=None,
    center_color=(255, 0, 0),
):
    screen.blit(background, (0, 0))
    pipe_group.draw(screen)
    ground_group.draw(screen)
    bird_group.draw(screen)

    if not started:
        screen.blit(begin_image, (120, 140))

    if center_message:
        message_surface = hud_font.render(center_message, True, center_color)
        shadow_surface = hud_font.render(center_message, True, BLACK)
        message_rect = message_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        shadow_rect = shadow_surface.get_rect(center=(SCREEN_WIDTH // 2 + 2, SCREEN_HEIGHT // 2 + 2))
        screen.blit(shadow_surface, shadow_rect)
        screen.blit(message_surface, message_rect)

    pygame.display.update()


def create_game_objects():
    bird_group = pygame.sprite.Group()
    bird = Bird()
    bird_group.add(bird)

    ground_group = pygame.sprite.Group()
    for index in range(2):
        ground_group.add(Ground(GROUND_WIDTH * index))

    pipe_group = pygame.sprite.Group()
    for index in range(2):
        pipes = get_random_pipes(SCREEN_WIDTH * index + 800)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

    return bird_group, bird, ground_group, pipe_group


def recycle_ground(ground_group):
    if is_off_screen(ground_group.sprites()[0]):
        ground_group.remove(ground_group.sprites()[0])
        ground_group.add(Ground(GROUND_WIDTH - 20))


def recycle_pipes(pipe_group):
    if is_off_screen(pipe_group.sprites()[0]):
        pipe_group.remove(pipe_group.sprites()[0])
        pipe_group.remove(pipe_group.sprites()[0])
        pipes = get_random_pipes(SCREEN_WIDTH * 2)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])


def jump_requested(events, controller):
    pressed = any(
        event.type == KEYDOWN and event.key in (K_SPACE, K_UP)
        for event in events
    )
    return pressed or controller.consume_jump()


def main():
    pygame.init()
    pygame.mixer.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - Thumbs Up")

    background = pygame.image.load(asset_path("sprites", "background-day.png"))
    background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
    begin_image = pygame.image.load(asset_path("sprites", "message.png")).convert_alpha()

    wing_sound = pygame.mixer.Sound(asset_path("audio", "wing.wav"))
    hit_sound = pygame.mixer.Sound(asset_path("audio", "hit.wav"))

    hud_font = pygame.font.SysFont("consolas", 28, bold=True)

    controller = ThumbController(camera_index=0)
    controller.start()

    clock = pygame.time.Clock()
    running = True

    try:
        while running:
            bird_group, bird, ground_group, pipe_group = create_game_objects()
            started = False
            crashed = False

            while running and not crashed:
                clock.tick(FPS)
                events = pygame.event.get()

                if any(event.type == QUIT for event in events):
                    running = False
                    continue

                wants_jump = jump_requested(events, controller)

                if not started:
                    if wants_jump:
                        bird.bump()
                        wing_sound.play()
                        started = True
                    bird.begin()
                    ground_group.update()
                    recycle_ground(ground_group)
                    draw_scene(
                        screen,
                        background,
                        begin_image,
                        bird_group,
                        pipe_group,
                        ground_group,
                        hud_font,
                        started,
                    )
                    continue

                if wants_jump:
                    bird.bump()
                    wing_sound.play()

                bird_group.update()
                ground_group.update()
                pipe_group.update()

                recycle_ground(ground_group)
                recycle_pipes(pipe_group)

                crashed = bool(
                    pygame.sprite.groupcollide(
                        bird_group,
                        ground_group,
                        False,
                        False,
                        pygame.sprite.collide_mask,
                    )
                    or pygame.sprite.groupcollide(
                        bird_group,
                        pipe_group,
                        False,
                        False,
                        pygame.sprite.collide_mask,
                    )
                )

                draw_scene(
                    screen,
                    background,
                    begin_image,
                    bird_group,
                    pipe_group,
                    ground_group,
                    hud_font,
                    started,
                )

                if crashed:
                    hit_sound.play()

            if not running:
                break

            restart_deadline = time.time() + 0.8
            waiting_restart = True
            while waiting_restart and running:
                clock.tick(FPS)
                events = pygame.event.get()

                if any(event.type == QUIT for event in events):
                    running = False
                    continue

                wants_jump = jump_requested(events, controller)
                if time.time() >= restart_deadline and wants_jump:
                    waiting_restart = False
                    continue

                draw_scene(
                    screen,
                    background,
                    begin_image,
                    bird_group,
                    pipe_group,
                    ground_group,
                    hud_font,
                    True,
                    center_message="GAME OVER",
                    center_color=(255, 60, 60),
                )
                pygame.display.update()
    finally:
        controller.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
