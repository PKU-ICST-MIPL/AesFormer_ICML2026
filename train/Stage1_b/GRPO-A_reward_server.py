import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

import pickle
import traceback
import asyncio
from typing import List
import torch
import ray
import vllm
from vllm import LLM, SamplingParams
from flask import Flask, request
from PIL import Image


MODEL_PATH = "AesFormer_ICML2026/pretrained_weights/Qwen2.5-VL-32B-Instruct"
NUM_GPUS = 2
NUM_TP = 2
PORT = 12341


score_idx = [15, 16, 17, 18, 19, 20]
MAX_SCORE = 5


PROMPT = """
Here are two sets of image-editing instructions for the same photo: one is the model output (model_output), and the other is the reference answer (ground_truth). Please rate how well model_output matches ground_truth on a scale of 0–5. “Match” means whether the instructions convey the same editing intent and key operations as the ground truth (paraphrases and synonyms are allowed). Do not judge based on wording similarity, length, or writing style.

You need to rate the match between model_output and ground_truth from 0 to 5.
0: Not a match—multiple key operations conflict with the ground truth, or there is almost no meaningful, comparable content.
5: Highly matched—key operations and important details are essentially consistent.

Response Format (Directly respond the score number):
0-5
"""

PROMPT_2 = """
Input: A photo + an image-editing instruction for that photo.
Task: Give the instruction a 0–5 score. A higher score means the instruction is more specific, executable, and likely to produce a clear and noticeable aesthetic improvement.
You MUST reward STRUCTURAL edits: composition/cropping, camera viewpoint, subject placement, and subject pose.
You MUST strongly penalize: generic/vague advice, overly conservative instructions (edited result would look almost the same as the input), or color-only adjustments.

[Hard Constraint (violation => low score)]
- The instruction may ONLY operate on subjects and scene elements already present in the photo. It must NOT introduce, assume, or describe any new objects, new people, new background elements, or nonexistent scene details.
- If this is clearly violated: give 0.

[Definition of “Structural Edits” (key for high score)]
A “structural edit” includes at least ONE of the following, and the instruction MUST clearly specify how to change it:
- Composition/Cropping: where to crop, how much to crop, what to keep/remove (e.g., remove ground, remove bystanders, reduce sky, crop to waist-up).
- Camera Viewpoint: distance change, camera height, up-tilt/down-tilt, angle change.
- Subject Placement: where to place the subject in the frame (e.g., right third, centered symmetrical, leave negative space).
- Subject Pose: head/eye direction, arm/hand actions, body orientation, standing/sitting, prop actions (e.g., remove selfie stick).

[High-score signals (prefer giving high scores)]
- Clear and executable composition/cropping goals: crop target (waist/chest/full-body), which regions to remove (ground/sky/bystanders), which key elements must remain (e.g., pagoda top fully visible).
- Clear camera/viewpoint guidance: closer/farther, lower/higher camera, slight upward/downward angle—explicitly stating what changes.
- Clear subject placement and use of negative space: right third/left negative space/center symmetry—explicitly stating where to place the subject.
- Clear pose details: head tilt direction, gaze/eyes closed, arm/hand placement, body orientation, remove selfie stick / lower raised arm—explicitly stating what to do.
- Sufficient detail and low ambiguity: the edit is likely to create a noticeable change, not just tiny, barely visible tweaks.

[Low-score signals (prefer giving low scores)]
- Generic advice without actionable operations: e.g., “improve aesthetics / optimize composition / make it premium / add atmosphere,” but not stating how to crop, place, pose, or shoot.
- Mostly color/exposure/filter changes (brightness/contrast/saturation/temperature, etc.) with almost no structural edits (crop/viewpoint/placement/pose).
- Overly conservative: the edited result would likely be almost identical to the input (changes hard to perceive).
- Internal contradictions / impossible constraints (e.g., asking to crop away a large part of the building while also requiring the building to remain fully visible), making execution difficult or forcing “no change.”

[Forced caps (to suppress generic/conservative answers)]
- If the instruction is mostly generic and lacks concrete operations: score MUST be at most 2.
- If the instruction is mostly color/exposure/filter with almost no structural edits: score MUST be at most 2.
- If the instruction is overly conservative and likely produces an almost unchanged output: score MUST be at most 2.
- If the hard constraint is clearly violated (introduces/assumes new elements): give 0 or 1 (typically 0).

[Scoring rubric (0–5)]
5: Contains at least TWO OR MORE types of structural edits (any two or more among crop/composition, viewpoint, placement, pose), with concrete, executable, checkable details; likely to produce a noticeable improvement; does not introduce new elements.
4: Contains at least ONE strong structural edit with sufficient supporting details; change likely noticeable; overall executable.
3: Contains structural edits but they are weak or underspecified; improvement uncertain; change may be small.
2: Mostly generic / mostly color / overly conservative; few or unclear structural edits; change likely minimal.
1: Very vague, not executable, or mismatched/contradictory; unlikely to improve.
0: Clearly introduces/assumes new elements or changes subject identity, or provides almost no meaningful editing instruction.

Response Format (Directly respond the score number):
0-5
"""


if vllm.__version__ != "0.9.2":
    raise ValueError("vLLM version must be exactly 0.9.2")

os.environ["VLLM_USE_V1"] = "0"


# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)


# ============================================================
# Logits Spy
# ============================================================

class LogitsSpy:
    def __init__(self):
        self.processed_logits: List[torch.Tensor] = []

    def __call__(self, token_ids, logits: torch.Tensor):
        self.processed_logits.append(logits)
        return logits


# ============================================================
# Ray Worker
# ============================================================

def load_image_fast(img_path: str, max_side: int = 512):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = max(w, h)
        if s > max_side:
            scale = max_side / s
            im = im.resize((int(w * scale), int(h * scale)))
        return im


@ray.remote(num_gpus=NUM_TP)
class ModelWorker:
    def __init__(self):
        self.llm = None
        self._load_model()

    def _load_model(self):
        self.llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=NUM_TP,
            max_model_len=4096,
        )

    def evaluate_pair(self, img_path: str, text1: str, text2: str):
        text_score = self.evaluate_text_pair(text1, text2)
        img_score = self.evaluate_image_text_pair(img_path, text1)
        return img_score, text_score

    def evaluate_text_pair(self, text1: str, text2: str) -> float:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"model_output: {text1}\n"
                            f"ground_truth: {text2}\n\n"
                            f"{PROMPT}"
                        ),
                    }
                ],
            }
        ]
        return self._vllm_evaluate(conversation)
    
    def evaluate_image_text_pair(self, img_path: str, text: str) -> float:
        image = load_image_fast(img_path, max_side=512)
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_pil",
                        "image_pil": image,
                    },
                    {
                        "type": "text",
                        "text": (
                            f"editing instruction: {text}\n\n"
                            f"{PROMPT_2}"
                        ),
                    },
                ],
            }
        ]
        return self._vllm_evaluate(conversation)

    def _vllm_evaluate(self, conversation, max_tokens: int = 3) -> float:
        logits_spy = LogitsSpy()
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            logits_processors=[logits_spy],
        )
        self.llm.chat(conversation, sampling_params=sampling_params)
        try:
            if not logits_spy.processed_logits:
                return 0.0
            logits = logits_spy.processed_logits[0]
            probs = torch.softmax(logits[score_idx], dim=-1)
            score = (
                torch.sum(
                    probs * torch.arange(len(score_idx), device=probs.device)
                ).item()
                / MAX_SCORE
            )
            return float(score)
        except Exception as e:
            print("Reward calculation failed:", e)
            return 0.0


# ============================================================
# Ray Worker
# ============================================================

workers = []
def initialize_ray_workers():
    global workers

    if not ray.is_initialized():
        ray.init()

    workers = []
    for _ in range(NUM_GPUS // NUM_TP):
        workers.append(ModelWorker.remote())

    print(f"[INFO] Initialized {len(workers)} Ray workers")


# ============================================================
# Batch
# ============================================================

async def evaluate_text_pairs_async(imgs, texts_1, texts_2):
    tasks = []
    for i, (img, t1, t2) in enumerate(zip(imgs, texts_1, texts_2)):
        worker = workers[i % len(workers)]
        tasks.append(worker.evaluate_pair.remote(img, t1, t2))
    return ray.get(tasks)


def evaluate_text_pairs(imgs, texts_1, texts_2):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(
            evaluate_text_pairs_async(imgs, texts_1, texts_2)
        )
        if not results:
            return [], []
        img_scores, text_scores = zip(*results)
        return list(img_scores), list(text_scores)
    finally:
        loop.close()


@app.route("/similarity", methods=["POST"])
def similarity_api():
    try:
        data = pickle.loads(request.get_data())
        imgs = data["imgs"] 
        texts_1 = data["texts_1"] 
        texts_2 = data["texts_2"] 

        assert len(imgs) == len(texts_1) == len(texts_2)

        img_scores, text_scores = evaluate_text_pairs(imgs, texts_1, texts_2)
        return pickle.dumps({"img_scores": img_scores, "text_scores": text_scores,}), 200

    except Exception:
        return traceback.format_exc().encode("utf-8"), 500


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    initialize_ray_workers()
    print(f"[INFO] Reward Server running on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
