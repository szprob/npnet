from sprite import BertToxicDetector

model = BertToxicDetector()
model.from_pretrained("toxic_detector_bert_51m")

# socre toxic
model.score("f u")

# sensitive detect
model.add_sensitive_words(["free", "fire"])
model.sensitive_words_detect("i like free fire")

# detect all
model.detect("i like free fire")
