---
library_name: transformers.js
license: apple-amlr
pipeline_tag: image-text-to-text
tags:
- fastvlm
base_model:
- apple/FastVLM-0.5B
---

# FastVLM: Efficient Vision Encoding for Vision Language Models

FastVLM was introduced in
**[FastVLM: Efficient Vision Encoding for Vision Language Models](https://www.arxiv.org/abs/2412.13303). (CVPR 2025)**

Try it out using the [online demo](https://huggingface.co/spaces/apple/fastvlm-webgpu), which runs 100% locally in your browser with Transformers.js!
<video muted autoplay loop controls src="https://cdn-uploads.huggingface.co/production/uploads/61b253b7ac5ecaae3d1efe0c/u7dQO66Fx8XR5e0D-fArs.mp4" width=800></video>


## Usage

### Transformers.js

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

You can then caption images as follows:

```js
import {
  AutoProcessor,
  AutoModelForImageTextToText,
  load_image,
  TextStreamer,
} from "@huggingface/transformers";

// Load processor and model
const model_id = "onnx-community/FastVLM-0.5B-ONNX";
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await AutoModelForImageTextToText.from_pretrained(model_id, {
  dtype: {
    embed_tokens: "fp16",
    vision_encoder: "q4",
    decoder_model_merged: "q4",
  },
});

// Prepare prompt
const messages = [
  {
    role: "user",
    content: "<image>Describe this image in detail.",
  },
];
const prompt = processor.apply_chat_template(messages, {
  add_generation_prompt: true,
});

// Prepare inputs
const url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg";
const image = await load_image(url);
const inputs = await processor(image, prompt, {
  add_special_tokens: false,
});

// Generate output
const outputs = await model.generate({
  ...inputs,
  max_new_tokens: 512,
  do_sample: false,
  streamer: new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false,
    // callback_function: (text) => { /* Do something with the streamed output */ },
  }),
});

// Decode output
const decoded = processor.batch_decode(
  outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
  { skip_special_tokens: true },
);
console.log(decoded[0]);
```

<details>

<summary>See here for example output</summary>

```
The image depicts a vibrant and colorful scene featuring a variety of flowers and plants. The main focus is on a striking pink flower with a dark center, which appears to be a type of petunia. The petals are a rich, deep pink, and the flower has a classic, slightly ruffled appearance. The dark center of the flower is a contrasting color, likely a deep purple or black, which adds to the flower's visual appeal.

In the background, there are several other flowers and plants, each with their unique colors and shapes. To the left, there is a red flower with a bright, vivid hue, which stands out against the pink flower. The red flower has a more rounded shape and a lighter center, with petals that are a lighter shade of red compared to the pink flower.

To the right of the pink flower, there is a plant with red flowers, which are smaller and more densely packed. The red flowers are a deep, rich red color, and they have a more compact shape compared to the pink flower.

In the foreground, there is a green plant with a few leaves and a few small flowers. The leaves are a bright green color, and the flowers are a lighter shade of green, with a few petals that are slightly open.

Overall, the image is a beautiful representation of a garden or natural setting, with a variety of flowers and plants that are in full bloom. The colors are vibrant and the composition is well-balanced, with the pink flower in the center drawing the viewer's attention.
```

</details>

