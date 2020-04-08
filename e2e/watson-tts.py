import json

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import ApiException, TextToSpeechV1

authenticator = IAMAuthenticator("qBXy-CjkC2hqcv2kBbIUuDviMje8145AnerVImvoIa8e")
text_to_speech = TextToSpeechV1(authenticator=authenticator)

text_to_speech.set_service_url(
    "https://api.us-east.text-to-speech.watson.cloud.ibm.com/instances/f4ccd88f-5db9-452f-af5f-f0d002d8c131"
)

try:
    pass
except ApiException as ex:
    print("Method failed with status code " + str(ex.code) + ": " + ex.message)

voices = text_to_speech.list_voices().get_result()
print(json.dumps(voices, indent=2))
