import json
import time
from pathlib import Path

import gradio as gr
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    get_twilio_turn_credentials,
)
from fastrtc.utils import audio_to_bytes
from gradio.utils import get_space
from pydantic import BaseModel
from speech import SpeechClient
from settings import get_settings
from openai import OpenAI

settings = get_settings()
speech_client = SpeechClient(
    stt_base_url=settings.STT_BASE_URL,
    stt_model=settings.STT_MODEL,
    stt_api_key=settings.STT_API_KEY,
    stt_response_format=settings.STT_RESPONSE_FORMAT,
    tts_base_url=settings.TTS_BASE_URL,
    tts_api_key=settings.TTS_API_KEY,
    tts_model=settings.TTS_MODEL,
    tts_voice=settings.TTS_VOICE,
    tts_backend=settings.TTS_BACKEND,
    language=settings.LANGUAGE,
)
llm_client = OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
tts_client = OpenAI(api_key=settings.TTS_API_KEY, base_url=settings.TTS_BASE_URL)
curr_dir = Path(__file__).parent

def response(
    audio: tuple[int, np.ndarray],
    chatbot: list[dict] | None = None,
):
    chatbot = chatbot or []
    messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]
    prompt = speech_client.speech_to_text(("audio-file.mp3", audio_to_bytes(audio)))
    chatbot.append({"role": "user", "content": prompt})
    yield AdditionalOutputs(chatbot)
    messages.append({"role": "user", "content": prompt})
    completion = llm_client.chat.completions.create(
        model=settings.LLM_MODEL,
        max_tokens=512,
        messages=messages,  # type: ignore
    )
    response_text = completion.choices[0].message.content
    chatbot.append({"role": "assistant", "content": response_text})

    start = time.time()

    print("starting tts", start)
    print("response_text", response_text)

    with tts_client.audio.speech.with_streaming_response.create(
          model=settings.TTS_MODEL,
          voice=settings.TTS_VOICE,
          input=response_text,
          response_format=settings.TTS_AUDIO_FORMAT,
          extra_body={"backend": settings.TTS_BACKEND, "language": settings.LANGUAGE},
    ) as stream_response:
        for i, chunk in enumerate(stream_response.iter_bytes(chunk_size=1024)):
            print("chunk", i, time.time() - start)
            audio_array = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)
            yield (24000, audio_array)
            print("finished tts", time.time() - start)
            yield AdditionalOutputs(chatbot)


chatbot = gr.Chatbot(type="messages")
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(response),
    additional_outputs_handler=lambda a, b: b,
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
)


class Message(BaseModel):
    role: str
    content: str


class InputData(BaseModel):
    webrtc_id: str
    chatbot: list[Message]


app = FastAPI()
stream.mount(app)


@app.get("/")
async def _():
    rtc_config = get_twilio_turn_credentials() if get_space() else None
    html_content = (curr_dir / "index.html").read_text()
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/input_hook")
async def _(body: InputData):
    stream.set_input(body.webrtc_id, body.model_dump()["chatbot"])
    return {"status": "ok"}


@app.get("/outputs")
def _(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            chatbot = output.args[0]
            yield f"event: output\ndata: {json.dumps(chatbot[-1])}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    if (mode := settings.MODE) == "UI":
        stream.ui.launch(server_port=7860, server_name="0.0.0.0")
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)
