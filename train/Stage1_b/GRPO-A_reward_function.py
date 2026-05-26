import re
import pickle
from typing import Any
import requests


# Metadata
REWARD_NAME = "aesthetic_reward"
REWARD_TYPE = "batch"


# ----------------------------
# Reward Server Config
# ----------------------------
SERVER_URL = "http://127.0.0.1:12341/similarity"
REQUEST_TIMEOUT = 10000


# ----------------------------
# Format Reward
# ----------------------------
def format_reward(response: str) -> float:
    headers = [
        r"\(1\)\s*Aspect Ratio\s*:",
        r"\(2\)\s*Framing and Composition\s*:",
        r"\(3\)\s*Camera Position and Viewpoint\s*:",
        r"\(4\)\s*Subject Position\s*:",
        r"\(5\)\s*Subject Pose and Action Details\s*:",
        r"\(6\)\s*Depth of Field and Focus Control\s*:",
        r"\(7\)\s*Color and Lighting Adjustment\s*:",
    ]
    pos = 0
    for h in headers:
        m = re.search(h, response[pos:], flags=re.DOTALL)
        if not m:
            return 0.0
        pos += m.end()
    return 1.0


# ----------------------------
# Ground truth parsing utils
# ----------------------------
_IMG_PATH_RE = re.compile(r"<IMG_PATH>(.*?)</IMG_PATH>", re.DOTALL)
_TAGS_TO_SPACE = [
    "<Ratio>", "</Ratio>",
    "<Composition>", "</Composition>",
    "<Camera>", "</Camera>",
    "<Position>", "</Position>",
    "<Pose>", "</Pose>",
    "<Focus>", "</Focus>",
    "<Color>", "</Color>",
]


def parse_ground_truth_with_img(gt_raw: str) -> tuple[str, str]:
    m = _IMG_PATH_RE.search(gt_raw)

    if not m:
        raise ValueError("ground_truth does not contain <IMG_PATH>...</IMG_PATH>")
    
    img_path = m.group(1).strip()
    gt_text = gt_raw[m.end():].strip()
    return img_path, gt_text


def sanitize_gt_for_reward_server(gt_text: str) -> str:
    for t in _TAGS_TO_SPACE:
        gt_text = gt_text.replace(t, " ")
    return gt_text


def call_reward_server_batch(imgs: list[str], texts_1: list[str], texts_2: list[str]) -> tuple[list[float], list[float]]:
    payload = {"imgs": imgs, "texts_1": texts_1, "texts_2": texts_2}
    resp = requests.post(
        SERVER_URL,
        data=pickle.dumps(payload),
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"\n\nReward server HTTP {resp.status_code}: {resp.text[:800]}\n\n")
    result = pickle.loads(resp.content)
    return result["img_scores"], result["text_scores"]


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
) -> list[dict[str, float]]:

    imgs: list[str] = []
    texts_1: list[str] = []
    texts_2: list[str] = []
    format_scores: list[float] = []

    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        f_score = format_reward(response)
        format_scores.append(f_score)

        img_path, gt_text = parse_ground_truth_with_img(reward_input["ground_truth"])
        gt_text = sanitize_gt_for_reward_server(gt_text)

        imgs.append(img_path)
        texts_1.append(response)   
        texts_2.append(gt_text)    

    try:
        img_scores, text_scores = call_reward_server_batch(imgs, texts_1, texts_2)
    except Exception as e:
        print(f"\n\n[ERROR] Reward server call failed: {e}. Fallback img_s/text_s=0.\n\n")
        img_scores = [0.0] * len(reward_inputs)
        text_scores = [0.0] * len(reward_inputs)

    scores: list[dict[str, float]] = []
    for i in range(len(reward_inputs)):
        img_s = float(img_scores[i])
        text_s = float(text_scores[i])
        f = float(format_scores[i])

        overall = img_s * 0.4 + text_s * 0.5 + format_weight * f

        scores.append(
            {
                "overall": float(overall),
                "format": f,
                "img_s": img_s,
                "text_s": text_s,
            }
        )

    return scores