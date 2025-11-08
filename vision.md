# Glados vision module

Glados is able to capture the world with a camera and react to what it sees.  
 The vision module is disabled by default, use the following command to start glados with vision:

```
uv run glados start --config .\configs\glados_vision_config.yaml
```

See [vision_config.py](./src/glados/vision/vision_config.py) for help with custom configuration.

Some usage notes:

- You can have both the LLM and the VLM (vision model) running on the same Ollama instance: choose models that can both fit in VRAM at the same time, otherwise it will be very slow as ollama keeps unloading them between vision and conversation tasks. The default config fits in 12GB VRAM.
- The default LLM (qwen3:4b-instruct) is good enough to follow the personality while correctly handling the visual descriptions. Sometimes it will reply to the user multiple times in my experience. You can try a bigger LLM, such as gpt-oss:20b for better results.
- The default VLM (qwen3-vl:2b-instruct) is more than sufficient for the vision task despite its small size. Unfortunately it's not viable for the conversational task so we need the 2nd model for the LLM.
- The image `resolution` can be small - glados doesn't need vision in full hd, a small 256px image is sufficient for good results and the processing is fast.
- `capture_interval_seconds` - Glados' vision is not continuous - photos are captured periodically by this interval. Increase if you see timeout errors, which means the vision processing takes longer.
- `camera_index` - this is usually 0 which will use the default / first available camera. You may need to change this index if you have multiple webcams connected. See OpenCV's docs for details.
- You can remove the whole `vision` section from the config to disable vision.

Implementation details:

- Vision runs in its own thread similar to other processors. I made so that glados is able to react to changes in the environment.
- the result of the VLM is a description of the image (eg. "a yellow rubber duck"). I add the `[vision]` prefix to this and pass it to the LLM queue as a user message. I tried sending it before as a `system` and custom roles but none of the VLMs seem to be prepared for this. Instead, when vision is configured, a few extra instructions are added to the LLM that prepares it to correctly handle these prefixed messages. This way Glados has the option to react or ignore subsequent vision observations. The instructions are at [constants.py](/src/glados/vision/constants.py)
- current problems: it looks like the LLM sometimes repeats previous answers after a few vision observations, even when it's instructed not to. Not sure whether this is a problem with qwen3:4b or coding error.
