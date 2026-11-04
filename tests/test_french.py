import shutil
from types import SimpleNamespace

import pytest

from glados.core.engine import FRENCH_REPLY_INSTRUCTION, GladosConfig
from glados.TTS.espeak import EspeakPhonemizer

needs_espeak = pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")


def _config(**updates: object) -> GladosConfig:
    config = GladosConfig.from_yaml("configs/glados_extended_config.yaml")
    return GladosConfig.model_validate({**config.model_dump(), **updates})


def test_french_profile_asks_for_french_replies_and_keeps_french_speech() -> None:
    config = _config(language="fr", routing={"addressed_language": "English"})
    assert config.to_chat_messages()[-1] == {"role": "system", "content": FRENCH_REPLY_INSTRUCTION}
    assert config.routing.addressed_language == "French"
    assert FRENCH_REPLY_INSTRUCTION not in str(_config().to_chat_messages())


@needs_espeak
def test_espeak_keeps_clause_punctuation_and_reads_names_and_times() -> None:
    sentences = EspeakPhonemizer("fr")("Bonjour, GLaDOS. Il est 14 h 30 ?")
    assert len(sentences) == 2 and sentences[0].endswith(".") and sentences[1].endswith("?")
    assert "," in sentences[0] and "ɡladˈɔs" in sentences[0]
    assert "ˈaʃ" not in sentences[1] and "ˈœʁ" in sentences[1], "“h” is read as “heures”, not the letter"


def test_only_english_voices_use_the_english_text_converter() -> None:
    from glados.core.tts_synthesizer import TextToSpeechSynthesizer

    converter = SimpleNamespace(text_to_spoken=lambda text: "fourteen")
    for language, expected in (("en", "fourteen"), ("fr", "14")):
        synth = SimpleNamespace(tts_model=SimpleNamespace(language=language), stc=converter)
        assert TextToSpeechSynthesizer._spoken(synth, "14") == expected


def test_glados_effect_raises_pitch_keeps_shape_and_never_clips() -> None:
    import numpy as np

    from glados.TTS.voice_effect import GladosEffect

    rate = 22050
    t = np.arange(rate * 4 // 10) / rate
    tone = (0.99 * np.sin(2 * np.pi * 200 * t)).astype(np.float32).reshape(-1, 1)  # Piper returns (n, 1)
    out = GladosEffect(flutter_depth=0).apply(tone, rate)
    assert out.ndim == 2 and out.shape[1] == 1 and out.dtype == np.float32
    assert abs(len(out) - len(tone) / 1.06) < 2, "Pitch is raised by resampling down"
    assert np.abs(out).max() <= 0.95 + 1e-6
    spectrum = np.abs(np.fft.rfft(out.reshape(-1)[2000:-2000]))
    peak_hz = np.argmax(spectrum) * rate / (len(out) - 4000)
    assert abs(peak_hz - 212) < 5, peak_hz


@needs_espeak
@pytest.mark.skipif(not __import__("pathlib").Path("models/TTS/fr_FR-siwis-medium.onnx").exists(),
                    reason="French voice not downloaded")
def test_french_uses_the_native_voice_with_the_glados_effect() -> None:
    from glados.TTS import get_speech_synthesizer

    voice = get_speech_synthesizer("glados", language="fr")
    assert voice.effect is not None and "siwis" in str(voice.ort_sess._model_path)
    assert get_speech_synthesizer("glados").effect is None
    assert voice.generate_speech_audio("Bonjour.").size > 1000


def test_wake_word_ignores_punctuation_and_allows_a_follow_up_window() -> None:
    from glados.core.speech_listener import SpeechListener

    listener = SimpleNamespace(wake_word="glados", SIMILARITY_THRESHOLD=SpeechListener.SIMILARITY_THRESHOLD)
    detected = lambda text: SpeechListener._wakeword_detected(listener, text)  # noqa: E731
    assert detected("Dis-moi, GLaDOS, quelle heure est-il ?") and detected("« Gladoss » ?")
    assert not detected("Il s'est caché derrière l'arbre")

    def follow_up(window: float, since: float | None) -> bool:
        state = SimpleNamespace(seconds_since_assistant=lambda: since)
        return SpeechListener._in_follow_up(SimpleNamespace(WAKE_FOLLOW_UP_S=window, _interaction_state=state))

    assert follow_up(15, 4.0) and not follow_up(15, 40.0) and not follow_up(0, 1.0) and not follow_up(15, None)
