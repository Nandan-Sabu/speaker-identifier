import numpy as np

SIMILARITY_THRESHOLD = 0.65
SAMPLE_RATE = 16000
encoder = None


def get_encoder():
    global encoder
    if encoder is None:
        from resemblyzer import VoiceEncoder
        encoder = VoiceEncoder()
    return encoder


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(audio: np.ndarray) -> np.ndarray:
    from resemblyzer import preprocess_wav
    enc = get_encoder()
    wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
    return enc.embed_utterance(wav)


def identify(audio: np.ndarray, profiles: dict):
    if not profiles:
        return None, 0.0

    embedding = get_embedding(audio)
    best_name = None
    best_score = -1

    for name, profile_embedding in profiles.items():
        score = float(cosine_similarity(embedding, profile_embedding))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= SIMILARITY_THRESHOLD:
        return best_name, best_score
    return None, best_score