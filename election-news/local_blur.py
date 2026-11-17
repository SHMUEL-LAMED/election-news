"""Free, on-runner replacement for the paid frame analysis used by women_blur.

Nothing leaves the GitHub runner and no AI service is called. Three small open models run on the CPU:

1. YOLOX-s finds every person in a sampled frame (full body, partial body, back view).
2. YuNet finds faces, so close-ups and people the body detector missed are still covered.
3. CLIP ViT-B/32 embeds each person (whole figure, and the head when it is large enough). A linear
   read-out of that embedding, trained on labelled pedestrians and faces, gives one number per look:
   how strongly the figure reads as male. The embedding sees hair, face, beard, head covering,
   clothing and build together.

One look is never trusted on its own. Looks are linked into a track per person inside a scene, and the
person stays visible only when the whole track reads as a man with a wide margin. Everything else is
blurred: women, girls, figures too small or too unclear to judge, and anyone the models disagree on.
A doubt always ends in a blur, never in a raw figure.
"""
import base64
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

POLICY = 'women-body-v1'
HERE = Path(__file__).resolve().parent

# Pinned files: a changed or truncated download is refused, and then no video is produced at all.
MODELS = {
    'person': ('https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx',
               'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'),
    'face': ('https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx',
             '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
    'clip': ('https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx',
             'fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40'),
}

# ---- decision rule (pure Python, covered by unit tests without any model) -------------------------
# Scores are log-odds that a look shows a male: 0 is undecided, positive is male, negative is female.
CLEAR_SCORE = 2.2        # average a track needs before its person may stay visible
CLEAR_SINGLE = 3.2       # a person seen in one sample only needs much stronger evidence
# A person whose face was never read (back turned, far away, covered) is judged by the body alone, and the
# body is the weaker witness: on held-out pedestrians about 3 women in 100 pass 2.2, about 3 in 1000 pass 3.8.
CLEAR_FACELESS = 3.8
FEMALE_LOOK = -0.5       # a look this low counts as evidence of a woman
MAX_FEMALE_SHARE = 0.12  # more than this share of such looks blurs the whole track
SCORE_CAP = 6.0          # one extreme look cannot outvote several ordinary ones
MIN_SIDE = 26            # pixels, in the analysed frame; smaller figures cannot be judged
MATCH_IOU = 0.25
MAX_GAP = 3              # samples a track may miss (occlusion) before a new track starts
MAX_CLASSIFIED = 12      # largest figures judged per frame; the rest are blurred unjudged
MAX_BOXES = 50


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_tracks(frames):
    """frames: [{'scene': n, 'people': [{'box': (x0, y0, x1, y1), 'score': float | None}]}].

    Returns tracks as lists of (frame_index, person_index). Tracks never cross a scene cut.
    """
    tracks, open_tracks, scene = [], [], None
    for index, frame in enumerate(frames):
        if frame['scene'] != scene:
            scene, open_tracks = frame['scene'], []
        open_tracks = [t for t in open_tracks if index - t['last'] <= MAX_GAP]
        pairs = []
        for p, person in enumerate(frame['people']):
            for t, track in enumerate(open_tracks):
                overlap = iou(person['box'], track['box'])
                if overlap >= MATCH_IOU:
                    pairs.append((overlap, p, t))
        taken_people, taken_tracks = set(), set()
        for overlap, p, t in sorted(pairs, reverse=True):
            if p in taken_people or t in taken_tracks:
                continue
            taken_people.add(p)
            taken_tracks.add(t)
            open_tracks[t]['items'].append((index, p))
            open_tracks[t].update(box=frame['people'][p]['box'], last=index)
        for p, person in enumerate(frame['people']):
            if p not in taken_people:
                track = {'items': [(index, p)], 'box': person['box'], 'last': index}
                open_tracks.append(track)
                tracks.append(track)
    return [t['items'] for t in tracks]


def track_is_clearly_male(scores, faceless=None):
    """scores: one entry per look, None when the figure could not be judged.

    faceless: per look, True when only the body was read. A track with no face reading at all needs
    the stricter CLEAR_FACELESS average, from at least two looks.
    """
    judged = [max(-SCORE_CAP, min(SCORE_CAP, s)) for s in scores if s is not None]
    if not judged or len(judged) * 2 < len(scores):
        return False
    if sum(1 for s in judged if s <= FEMALE_LOOK) > MAX_FEMALE_SHARE * len(judged):
        return False
    average = sum(judged) / len(judged)
    if faceless is not None and all(flag for flag, s in zip(faceless, scores) if s is not None):
        return len(judged) >= 2 and average >= CLEAR_FACELESS
    return average >= (CLEAR_SINGLE if len(judged) < 2 else CLEAR_SCORE)


def decide(frames):
    """Returns, per frame, the pixel boxes that must be blurred."""
    result = [[] for _ in frames]
    for track in link_tracks(frames):
        looks = [frames[f]['people'][p] for f, p in track]
        if track_is_clearly_male([look['score'] for look in looks], [look.get('faceless', False) for look in looks]):
            continue
        for f, p in track:
            result[f].append(frames[f]['people'][p]['box'])
    return result


def to_policy_boxes(boxes, width, height):
    """Pixel (x0, y0, x1, y1) -> the [y0, x0, y1, x1] 0..1000 form women_blur expects."""
    output = []
    for x0, y0, x1, y1 in boxes:
        x0, x1 = max(0.0, min(width, x0)), max(0.0, min(width, x1))
        y0, y1 = max(0.0, min(height, y0)), max(0.0, min(height, y1))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        output.append([math.floor(y0 * 1000 / height), math.floor(x0 * 1000 / width),
                       min(1000, math.ceil(y1 * 1000 / height)), min(1000, math.ceil(x1 * 1000 / width))])
    # Largest first; with an absurd crowd the frame is covered whole rather than partly.
    output.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return output if len(output) <= MAX_BOXES else [[0, 0, 1000, 1000]]


# ---- models ---------------------------------------------------------------------------------------
def model_path(name):
    url, digest = MODELS[name]
    folder = Path(os.environ.get('BLUR_MODEL_DIR') or Path.home() / '.cache' / 'news-blur-models')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + '-' + digest[:16] + '.onnx')

    def valid():
        if not path.exists():
            return False
        check = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(1 << 20):
                check.update(block)
        return check.hexdigest() == digest
    for attempt in range(3):
        if valid():
            return str(path)
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'news-blur/1'})
            partial = path.with_suffix('.part')
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
                while block := response.read(1 << 20):
                    output.write(block)
            partial.replace(path)
        except OSError:
            time.sleep(3 + attempt * 5)
    if valid():
        return str(path)
    path.unlink(missing_ok=True)
    raise ValueError('blur_model_unavailable: ' + name)


def runtime():
    """onnxruntime is installed on first use, so the workflow file needs no change to switch engines."""
    try:
        import onnxruntime
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', 'onnxruntime==1.20.1'],
                       check=True, timeout=300, stdout=subprocess.DEVNULL)
        import onnxruntime
    return onnxruntime


class LocalAnalyzer:
    """Callable like the remote analyzer, plus analyze_samples() for whole-clip tracking."""

    def __init__(self, threads=None):
        import numpy as np
        import cv2
        ort = runtime()
        self.np, self.cv2 = np, cv2
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or max(1, os.cpu_count() or 1)
        providers = ['CPUExecutionProvider']
        self.person = ort.InferenceSession(model_path('person'), options, providers=providers)
        self.clip = ort.InferenceSession(model_path('clip'), options, providers=providers)
        self.face_model = model_path('face')
        self.faces = {}
        probe = json.loads((HERE / 'gender_probe.json').read_text())
        self.weights = np.array(probe['weights'], dtype=np.float32)
        self.bias = float(probe['bias'])
        if self.weights.shape != (512,):
            raise ValueError('blur_probe_invalid')
        self.stats = {'frames': 0, 'people': 0, 'blurred': 0, 'clear': 0}

    # -- detection --
    def detect_people(self, image, threshold=0.25):
        np, cv2 = self.np, self.cv2
        size = 640
        height, width = image.shape[:2]
        ratio = min(size / height, size / width)
        resized = cv2.resize(image, (max(1, round(width * ratio)), max(1, round(height * ratio))), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        output = self.person.run(None, {self.person.get_inputs()[0].name: blob})[0][0]
        grids, strides = [], []
        for stride in (8, 16, 32):
            side = size // stride
            xs, ys = np.meshgrid(np.arange(side), np.arange(side))
            grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
            strides.append(np.full((side * side, 1), stride))
        grid, stride = np.concatenate(grids), np.concatenate(strides)
        centre = (output[:, :2] + grid) * stride
        extent = np.exp(np.clip(output[:, 2:4], -10, 10)) * stride
        score = output[:, 4] * output[:, 5]  # objectness x "person"
        keep = score >= threshold
        if not keep.any():
            return []
        centre, extent, score = centre[keep] / ratio, extent[keep] / ratio, score[keep]
        boxes = np.concatenate((centre - extent / 2, centre + extent / 2), 1)
        picked = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
                                  [float(s) for s in score], threshold, 0.5)
        result = []
        for i in np.array(picked).reshape(-1):
            x0, y0, x1, y1 = boxes[i]
            x0, y0, x1, y1 = max(0.0, float(x0)), max(0.0, float(y0)), min(float(width), float(x1)), min(float(height), float(y1))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                result.append((x0, y0, x1, y1))
        return result

    def detect_faces(self, image, threshold=0.6):
        cv2 = self.cv2
        height, width = image.shape[:2]
        # Small broadcast frames are enlarged so that distant faces reach the detector's working size.
        scale = 2.0 if max(height, width) <= 480 else 1.0
        if scale != 1.0:
            image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_CUBIC)
        key = image.shape[:2]
        if key not in self.faces:
            self.faces[key] = cv2.FaceDetectorYN.create(self.face_model, '', (key[1], key[0]), threshold, 0.3, 200)
        _, found = self.faces[key].detect(image)
        result = []
        for row in (found if found is not None else []):
            x, y, w, h = (float(v) / scale for v in row[:4])
            if w >= 6 and h >= 6:
                result.append((max(0.0, x), max(0.0, y), min(float(width), x + w), min(float(height), y + h)))
        return result

    # -- classification --
    def embed(self, crops):
        np, cv2 = self.np, self.cv2
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        deviation = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        batch = []
        for crop in crops:
            height, width = crop.shape[:2]
            ratio = 224 / max(height, width)
            resized = cv2.resize(crop, (max(1, round(width * ratio)), max(1, round(height * ratio))), interpolation=cv2.INTER_CUBIC)
            canvas = np.full((224, 224, 3), 124, dtype=np.uint8)
            top, left = (224 - resized.shape[0]) // 2, (224 - resized.shape[1]) // 2
            canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
            batch.append(((canvas[:, :, ::-1].astype(np.float32) / 255 - mean) / deviation).transpose(2, 0, 1))
        vectors = self.clip.run(['image_embeds'], {'pixel_values': np.stack(batch)})[0]
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def male_scores(self, crops):
        if not crops:
            return []
        return [float(v) for v in self.embed(crops) @ self.weights + self.bias]

    def look(self, image):
        """One frame -> people with a box and a male score (None when the figure cannot be judged)."""
        height, width = image.shape[:2]
        bodies, faces = self.detect_people(image), self.detect_faces(image)
        people = [{'box': box, 'face': None} for box in bodies]
        for face in faces:
            fx, fy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
            owners = [p for p in people if p['box'][0] <= fx <= p['box'][2] and p['box'][1] <= fy <= p['box'][3]]
            if owners:
                # The tightest surrounding body owns the face.
                owner = min(owners, key=lambda p: (p['box'][2] - p['box'][0]) * (p['box'][3] - p['box'][1]))
                if owner['face'] is None:
                    owner['face'] = face
                continue
            # A face without a detected body: assume the body under it so the whole figure is covered.
            w, h = face[2] - face[0], face[3] - face[1]
            people.append({'box': (max(0.0, fx - 1.6 * w), max(0.0, face[1] - 0.7 * h), min(float(width), fx + 1.6 * w), min(float(height), face[3] + 6.5 * h)),
                           'face': face})
        order = sorted(range(len(people)), key=lambda i: -(people[i]['box'][2] - people[i]['box'][0]) * (people[i]['box'][3] - people[i]['box'][1]))
        crops, owners = [], []
        for rank, i in enumerate(order):
            person = people[i]
            person['score'] = None
            x0, y0, x1, y1 = person['box']
            if rank >= MAX_CLASSIFIED or max(x1 - x0, y1 - y0) < MIN_SIDE:
                continue
            crops.append(image[int(y0):math.ceil(y1), int(x0):math.ceil(x1)])
            owners.append((i, 'body'))
            face = person['face']
            if face and min(face[2] - face[0], face[3] - face[1]) >= 14:
                w, h = face[2] - face[0], face[3] - face[1]
                hx0, hy0 = max(0, int(face[0] - 0.45 * w)), max(0, int(face[1] - 0.55 * h))
                hx1, hy1 = min(width, math.ceil(face[2] + 0.45 * w)), min(height, math.ceil(face[3] + 0.45 * h))
                crops.append(image[hy0:hy1, hx0:hx1])
                owners.append((i, 'head'))
        scores = {}
        for (i, kind), score in zip(owners, self.male_scores(crops)):
            scores.setdefault(i, {})[kind] = score
        for i, found in scores.items():
            body, head = found.get('body'), found.get('head')
            people[i]['faceless'] = head is None
            if head is None:
                people[i]['score'] = body
            elif min(body, head) <= FEMALE_LOOK:
                # Either view reading as a woman is enough.
                people[i]['score'] = min(body, head)
            else:
                people[i]['score'] = (body + head) / 2
        return [{'box': p['box'], 'score': p['score'], 'faceless': p.get('faceless', True)} for p in people]

    # -- interface used by women_blur.sanitize_video --
    def analyze_samples(self, samples, deadline=None):
        """samples: [{'scene': n, 'image': base64 JPEG}], in time order. Returns policy boxes per sample."""
        np, cv2 = self.np, self.cv2
        frames, sizes = [], []
        for sample in samples:
            if deadline is not None and time.monotonic() > deadline:
                raise ValueError('blur_analysis_budget_exceeded')
            image = cv2.imdecode(np.frombuffer(base64.b64decode(sample['image']), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('frame_decode_failed')
            sizes.append((image.shape[1], image.shape[0]))
            frames.append({'scene': sample.get('scene', 0), 'people': self.look(image)})
        decided = decide(frames)
        self.stats['frames'] += len(frames)
        for frame, boxes in zip(frames, decided):
            self.stats['people'] += len(frame['people'])
            self.stats['blurred'] += len(boxes)
            self.stats['clear'] += len(frame['people']) - len(boxes)
        return [to_policy_boxes(boxes, width, height) for boxes, (width, height) in zip(decided, sizes)]

    def __call__(self, images):
        # Batch form of the remote analyzer: without the clip around them, the frames form one scene.
        boxes = self.analyze_samples([{'scene': 0, 'image': image} for image in images])
        return {'policy': POLICY, 'frames': [{'index': i, 'boxes': b} for i, b in enumerate(boxes)]}
