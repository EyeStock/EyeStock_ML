from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    user_id: str
    chat_id: str
    text: str

class PredictResponse(BaseModel):
    response: str
    # command_vector: List[float]


class EndChatPreferenceRequest(BaseModel):
    user_id: str
    chat_id: str
    preference_vector: List[float] = []

class EndChatPreferenceResponse(BaseModel):
    user_question: List[str]

class EndChatCommandRequest(BaseModel):
    user_id: str
    chat_id: str
    command_vector: List[float] = []


class UsePrevVectorRequest(BaseModel):
    user_id: str
    use_previous: bool
    preference_vector: List[float] = []
    command_vector: List[float] = []

class UsePrevVectorResponse(BaseModel):
    response: str
