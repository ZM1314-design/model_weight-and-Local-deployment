# Farmer AI Assistant (Continued Pretraining)

这是一个在基座模型上基于农业语料继续预训练后得到的本地模型服务示例，目标是提供农业场景问答与建议。
Agricultural_Fine_tuning-LLM为微调后的模型权重

## Project Highlights

- 农业语料继续预训练定位
- 本地 FastAPI 推理接口
- 极简前端页面调用接口

## File Structure

- `server.py`：后端推理 API（`POST /generate`）
- `index.html`：前端演示页面
- `generation_config.json`：默认生成参数
- `model.safetensors`：模型权重
- `tokenizer.json` / `tokenizer_config.json` / `vocab.json` / `merges.txt`：分词器相关文件

## Strict System Hiding

后端默认启用：

- `STRICT_HIDE_SYSTEM = True`
- 不在 API 响应中返回 `system`
- 不在日志中打印 `system` 原文（仅显示 `<hidden>`）
- 返回文本会清洗 `system/user/assistant` 回显，仅展示回答正文

## Local Run

1. 安装依赖：

```bash
python -m pip install torch transformers fastapi uvicorn safetensors
```

2. 启动后端：

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

3. 启动静态页面（可选）：

```bash
python -m http.server 5174
```

4. 浏览器访问：

- `http://127.0.0.1:5174/index.html`

## API Example

Endpoint:

- `POST http://127.0.0.1:8000/generate`

Request JSON:

```json
{
  "prompt": "小麦返青期叶片发黄怎么办？",
  "max_new_tokens": 220,
  "temperature": 0.25,
  "top_p": 0.85,
  "top_k": 30,
  "repetition_penalty": 1.2,
  "no_repeat_ngram_size": 4
}
```

## Upload to GitHub

模型权重文件较大，请使用 Git LFS：

```bash
git lfs install
git lfs track "*.safetensors" "*.bin"
git add .gitattributes
git add .
git commit -m "Add farmer AI assistant with strict hidden system"
```

然后推送：

```bash
git push origin <your-branch>
```

## Agricultural_Fine_tuning-LLM

本研究微调后的模型权重较大，超过github限制的2GB，如需完整权重文件，请发送邮箱至
zhao_myc1@163.com


## Reproducibility Notes

- 使用同一套模型与 tokenizer 文件
- 保持 `generation_config.json` 参数一致
- 保持 `server.py` 中 `STRICT_HIDE_SYSTEM = True`
- 模型来自：在基座模型上使用农业领域数据集继续预训练得到

