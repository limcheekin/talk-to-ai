from io import BytesIO
from openai import OpenAI, AsyncOpenAI
from pydantic import SecretStr

class SpeechClient:
  def __init__(self,  
               stt_base_url: str,
               stt_api_key: SecretStr, 
               stt_model: str,
               stt_response_format: str,  
               tts_base_url: str,
               tts_api_key: SecretStr, 
               tts_model: str,
               tts_voice: str,
               tts_backend: str,
               language: str = 'en'):
    self.__stt_base_url = stt_base_url
    self.__stt_model = stt_model
    self.__stt_api_key = stt_api_key
    self.__stt_response_format = stt_response_format
    self.__language = language
    self.__tts_base_url = tts_base_url
    self.__tts_api_key = tts_api_key
    self.__tts_model = tts_model
    self.__tts_voice = tts_voice
    self.__tts_backend = tts_backend
    

  def speech_to_text(self, audio_file: tuple) -> str:
    client = OpenAI(api_key=self.__stt_api_key.get_secret_value(), base_url=self.__stt_base_url)
    response = client.audio.transcriptions.create(
        model=self.__stt_model,
        file=audio_file,
        language=self.__language,
        response_format=self.__stt_response_format,
    )
    return response.text

  def text_to_speech(self, text: str) -> bytes:
    client = OpenAI(api_key=self.__tts_api_key.get_secret_value(), base_url=self.__tts_base_url)
    response = client.audio.speech.create(
                model=self.__tts_model,
                voice=self.__tts_voice,
                input=text,
                extra_body={"backend": self.__tts_backend, "language": self.__language},
            )
    return response.read()

