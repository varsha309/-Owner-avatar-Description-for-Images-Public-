# Image-Description-Generation-using-VGG19-and-Dense-LSTM

An encoder-decoder-based description generation. Existing papers used only objects for descriptions, but the relationship between them is equally important. Which in turn requires context information. For which technique like Long Short-term Memory (LSTM) is required. This paper proposes an encoder-decoder-based methodology to generate human-like textual descriptions. Dense-LSTM is presented for better description as a decoder with modified VGG19 as an encoder
to capture information to describe the scene. Standard datasets Flickr8K and Flickr30k are used for testing and training purposes. BLEU (Bilingual Evaluation Understudy) score is used to evaluate a generated text.

VGG19 is used as an encoder with slight modification to get the desired dimensions. A novel Dense-LSTM is proposed as a decoder for the textual part.

To view the paper, https://drive.google.com/drive/u/1/folders/1ae9XeWYdPUGpOgir_dC8zMxjhhf_tHor
or https://content.iospress.com/articles/journal-of-intelligent-and-fuzzy-systems/ifs222358
