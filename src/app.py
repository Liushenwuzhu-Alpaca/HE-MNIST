"""
Flask Web应用 - 基于同态加密的手写数字识别演示
"""

import base64
import io
import os
import sys

import numpy as np
import tenseal as ts
import torch
from flask import Flask, jsonify, render_template, request, send_file

sys.path.append(os.path.dirname(__file__))

from ciphertext_inference import CiphertextInference
from encrypt import Encoder
from keygen import KeyGenerator

app = Flask(__name__, template_folder="../templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

CONTEXT_PATH = "./keys/context.bin"
MODEL_PATH = "./models/mnist_net.pth"

context = None
inference_engine = None


def initialize_app():
    """初始化应用，加载密钥和模型"""
    global context, inference_engine

    print("初始化应用...")

    if not os.path.exists(CONTEXT_PATH):
        print("密钥上下文不存在，正在生成...")
        keygen = KeyGenerator()
        keygen.generate()
        keygen.save_keys(
            public_key_path="./keys/public_key.bin",
            secret_key_path="./keys/secret_key.bin",
            context_path=CONTEXT_PATH,
            galois_keys_path="./keys/galois_keys.bin",
        )

    context = KeyGenerator.load_context(CONTEXT_PATH)
    print("密钥上下文已加载")

    if not os.path.exists(MODEL_PATH):
        print("警告: 模型文件不存在，请先运行训练")

    try:
        model = torch.load(MODEL_PATH, map_location="cpu")
        if hasattr(model, "state_dict"):
            weights = model.state_dict()
        else:
            weights = model

        inference_engine = CiphertextInference(context, weights)
        print("模型已加载")
    except Exception as e:
        print(f"模型加载失败: {e}")
        inference_engine = None


initialize_app()


@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


@app.route("/api/generate-keys", methods=["POST"])
def generate_keys():
    """生成新的密钥对"""
    try:
        keygen = KeyGenerator()
        keygen.generate()
        keygen.save_keys(
            public_key_path="./keys/public_key.bin",
            secret_key_path="./keys/secret_key.bin",
            context_path=CONTEXT_PATH,
            galois_keys_path="./keys/galois_keys.bin",
        )

        global context, inference_engine
        context = KeyGenerator.load_context(CONTEXT_PATH)

        return jsonify({"success": True, "message": "密钥生成成功"})
    except Exception as e:
        return jsonify({"success": False, "message": f"密钥生成失败: {str(e)}"}), 500


@app.route("/api/download-key/<key_type>")
def download_key(key_type):
    """下载密钥文件"""
    key_map = {
        "public": "./keys/public_key.bin",
        "secret": "./keys/secret_key.bin",
        "context": "./keys/context.bin",
    }

    key_path = key_map.get(key_type)
    if not key_path or not os.path.exists(key_path):
        return jsonify({"error": "密钥文件不存在"}), 404

    return send_file(key_path, as_attachment=True, download_name=f"{key_type}_key.bin")


@app.route("/api/predict", methods=["POST"])
def predict():
    """密文预测API"""
    try:
        if inference_engine is None:
            return jsonify({"success": False, "message": "模型未加载"}), 500

        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"success": False, "message": "未提供图片数据"}), 400

        image_array = np.array(image_data, dtype=np.float64)

        # 前端已经做了颜色反转(255-avg)，这里不需要再反转
        # image_array = 1.0 - image_array

        encoder = Encoder()
        encoded_input = encoder.encode_image(image_array)

        encrypted_input = ts.ckks_vector(context, encoded_input.tolist())

        prediction, probabilities = inference_engine.predict_encrypted(encrypted_input)

        return jsonify(
            {
                "success": True,
                "prediction": int(prediction),
                "probabilities": probabilities.tolist(),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "message": f"预测失败: {str(e)}"}), 500


@app.route("/api/predict-plain", methods=["POST"])
def predict_plain():
    """明文预测API（用于对比）"""
    try:
        if inference_engine is None:
            return jsonify({"success": False, "message": "模型未加载"}), 500

        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"success": False, "message": "未提供图片数据"}), 400

        image_array = np.array(image_data, dtype=np.float64)

        # 前端已经做了颜色反转(255-avg)，这里不需要再反转
        # image_array = 1.0 - image_array

        encoder = Encoder()
        encoded_input = encoder.encode_image(image_array)

        prediction, probabilities = inference_engine.predict(encoded_input)

        return jsonify(
            {
                "success": True,
                "prediction": int(prediction),
                "probabilities": probabilities.tolist(),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "message": f"预测失败: {str(e)}"}), 500


@app.route("/api/status")
def status():
    """获取系统状态"""
    return jsonify(
        {
            "model_loaded": inference_engine is not None,
            "context_loaded": context is not None,
            "keys_exist": os.path.exists(CONTEXT_PATH),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
