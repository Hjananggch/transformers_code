from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    r"C:\Users\AN\Desktop\qw\qw_0.5_instruct",  # 模型路径
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\AN\Desktop\qw\qw_0.5_instruct")

@app.route("/chat", methods=["POST"])
def vllm_infer():
    data = request.json

    prompt = f"{data['content']}"
    messages = [
        {"role": "system", "content": "你是一个智能AI助手"},  # 系统角色消息
        {"role": "user", "content": prompt}  # 用户角色消息
    ]

    text = tokenizer.apply_chat_template(
        messages,  # 要格式化的消息
        tokenize=False,  # 不进行分词
        add_generation_prompt=True  # 添加生成提示
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return jsonify({"response": response}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=6000)


