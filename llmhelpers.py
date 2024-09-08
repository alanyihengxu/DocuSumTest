"""EXAMPLES"""
import apihelper
from transformers import AutoProcessor, SeamlessM4Tv2Model
import json
import base64

class huggingFaceAPis:

    def __init__(self):
        self.helper = apihelper.apiHelper()

    def sumQuery(self, model, text, max_length, min_length):
        APIURL, modelKey = self.helper.base(model)
        payload = {"inputs": text}
        if modelKey != "FLAN-T5":
            payload.update({
                "max_length": max_length,
                "min_length": min_length
            })
        if modelKey in ["FLAN-T5"]:
            payload["inputs"] = f"Summarize: {text}"

        return self.helper.closure(APIURL, payload)

    def transQuery(self, model, text, langTTF, langTTT):
        APIURL, modelKey = self.helper.base(model)
        src_lang_code = langTTF
        tgt_lang_code = langTTT
        payload = {
            "inputs": text,
            "parameters": {
                "src_lang": src_lang_code,
                "tgt_lang": tgt_lang_code
            }
        }
        print(payload)
        print("DONE")
        return self.helper.closure(APIURL, payload)

    def askQuery(self, model, text, question):
        APIURL, modelKey = self.helper.base(model)
        payload = {
            "inputs": {
                "question": question,
                "context": text
            }
        }

        print(payload)
        return self.helper.closure(APIURL, payload)

    def sentimentQuery(self, model, text):
        APIURL, modelKey = self.helper.base(model)
        payload = {
            "inputs": f"What is the sentiment of this? {text}"
        }
        return self.helper.closure(APIURL, payload)

    #def readQuery(self, model, text):
        print("Test2")
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        tts_model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
        print("Test3")
        text_inputs = processor(text = text, src_lang="eng", return_tensors="pt")
        audio_array_from_text = tts_model.generate(**text_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()
        print("Test4")

        payload = {}
        payload['content'] = base64.b64encode(audio_array_from_text)
        payload['sample_rate'] = base64.b64encode(model.config.sampling_rate)

        out = json.dumps(payload)
        print("Test5")

        return out