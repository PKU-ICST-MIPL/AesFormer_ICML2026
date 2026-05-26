import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import json
from tqdm import tqdm

model_path = "/path/to/your/Stage1_b_model"
test_json_path = "AesFormer_ICML2026/AesRecon_dataset/jsons/test/test.json"
poor_image_root = "AesFormer_ICML2026/AesRecon_dataset/images/poor_images"
answer_path = "AesFormer_ICML2026/test/Stage1/Editing_Actions.json"


model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto",)
processor = AutoProcessor.from_pretrained(model_path)


with open(test_json_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)


question = "Given a photo captured in a real-world scene, please generate executable photo reconstruction instructions to guide an image editing model to improve the aesthetic quality of the photo. The instructions should not change the subject’s appearance or the scene identity, and should improve aesthetics only by adjusting composition and other photography-related factors. You are only allowed to operate on subjects and scene elements already present in the photo, and must not introduce, assume, or describe any new objects or background elements."
answer = {}
for image_name, poor_image_name in tqdm(test_data.items(), desc="Running benchmark"):
    poor_image = os.path.join(poor_image_root, poor_image_name)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": poor_image,
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    
    # ===== remove <|im_end|> =====
    response_text = output_text[0]
    if response_text.endswith("<|im_end|>"):
        response_text = response_text[:-len("<|im_end|>")]
    
    answer[poor_image_name] = response_text

    with open(answer_path, "w", encoding="utf-8") as f:
        json.dump(answer, f, ensure_ascii=False, indent=2)

print(f"Total answers: {len(answer)}")
